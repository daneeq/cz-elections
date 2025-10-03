# CZ Elections Live — Design Document
**Last updated:** 2025-10-03  
**Owner:** You (and contributors)

## 0. Goal & Scope
Build a lightweight, **live** dashboard for the 2025 Czech parliamentary elections that:
- Ingests **live partial results** (volby.cz official open data; XML).
- Runs a **Monte Carlo simulator** to project final **seat distributions**.
- Computes **odds that predefined coalitions** reach a governing majority (≥101 of 200).
- Presents a **simple Streamlit UI** with a table of current partials and coalition odds.

**Non-goals (v0):**
- Fancy visualizations (maps/needles).
- Full legal seat allocation nuances beyond the baseline (we start with a fast D’Hondt approximation).
- Historical comparison UI (beyond using 2021 data as a **fixture** for testing).

---

## 1. Users & Questions
**Primary user:** technically savvy observer (e.g., data practitioner, journalist) following election night.  
**Key questions:**
1. What are the **current partial results** by party?
2. Given uncertainty, what are the **odds that Coalition X** reaches ≥101 seats?
3. Which coalitions are trending up/down as counting progresses?

---

## 2. High-level Architecture
```
Streamlit UI  ─────▶  Monte Carlo Engine  ─────▶  Coalition Odds
      ▲                         ▲
      │                         │
      └──── Data Adapter ◀──────┘
            (volby2025 XML OR 2021 fixture)
```

**Components**
- **Data Adapters (`src/cz_elections_live/data_adapters/`)**
  - `volby2025.py` — fetch & parse live **XML** → normalized DataFrame.
  - `ps2021_fixture.py` — generate **synthetic partials** from 2021-like final totals for local testing.
- **Model (`src/cz_elections_live/model/`)**
  - `monte_carlo.py` — Dirichlet–multinomial draws for remaining votes, N simulations, aggregation.
  - `seats.py` — fast **approximate D’Hondt** seat allocator (national approximation for v0).
  - `thresholds.py` — constants for **5/7/11%** thresholds.
- **Config (`config/`)**
  - `coalitions.yaml` — named coalition presets (editable).
  - `parties.yaml` — codes → names + coalition membership meta (e.g., SPOLU members).
- **UI (`src/cz_elections_live/streamlit_app.py`)**
  - Two modes: `ps2021_fixture` (default) or `ps2025_live`.
  - Controls: N sims, fixture profile, manual refresh.  
  - Tables: partials; coalition odds.

---

## 3. Data Sources & Formats
- **Live source (2025)**: official volby.cz **XML** endpoints (topline + optional regional).  
  - Expected: national totals listing parties with vote counts/percentages, timestamp, and progress indicators.
  - **Namespaces** present; XPath uses `NS = {"ps": "http://www.volby.cz/ps/"}` (verify once live).
  - **Polling cadence**: typically updated ~every 60s (verify exact feed notes on the day).
  - We keep endpoint(s) configurable via `.env` (`VOLBY2025_TOPLINE`).
- **Fixture source (2021-like)**:
  - `data/raw/ps2021_national_totals.csv`
  - `data/raw/ps2021_region_totals.csv`
  - We **simulate counting order** profiles (e.g., rural-first vs urban-late) for realistic uncertainty.

**Normalization target (per adapter)**  
DataFrame columns (minimum):
```
party_code, party_name, votes_reported, pct_reported, [timestamp], [region_code?]
```

---

## 4. Modeling Approach
### 4.1 Remaining votes & uncertainty
- From partial totals, estimate **remaining votes**. For v0 fixture we use a scalar multiplier (`est_total_factor≈1.8`).  
  In live mode, prefer official progress indicators (e.g., precincts counted or turnout) when available.
- For each **region** (v1) or **national** (v0), draw remaining votes using **Dirichlet–multinomial**:
  - Mean = current share; Variance scaled (`alpha_scale`) based on how incomplete the region is.
  - Optional **biasing** for late-reporting urban regions (Prague/large cities) using 2021 patterns (v1).

### 4.2 Thresholds
- Apply legal vote thresholds on **national share** per party/coalition:
  - Single party **5%**
  - Two-party coalition **7%**
  - Three-or-more **11%**
- v0: implement single-party 5% rule; v1: coalition-aware thresholds via `config/parties.yaml`.

### 4.3 Seat Allocation
- **v0 (baseline)**: national **D’Hondt approximation** to distribute 200 seats to eligible parties.
- **v1 (improvement)**: per-region D’Hondt, then (if required by 2025 rules) national top-up stage.
- Accuracy vs. simplicity trade-off: v0 yields fast, stable signals during fluid counting; v1 for closer to legal exactness.

### 4.4 Simulation Specs
- `N` = 5k–10k simulations per refresh (UI slider).
- Outputs:
  - Party seats per sim → distribution.
  - For each coalition: `P(majority) = P(total_seats ≥ 101)` and **expected seats**.
- Performance:
  - NumPy vectorization where possible.
  - Keep party count small (only parties above some **visibility** threshold for speed, but retain for thresholds).

---

## 5. Coalition Modeling
- Presets in `config/coalitions.yaml`:
  - **Gov bloc:** SPOLU + Pirates + STAN
  - **Opposition core:** ANO + SPD
  - Variants: SPOLU + Pirates, SPOLU + STAN
  - ANO + ČSSD (centrist partner scenario)
  - Grand coalition: SPOLU + ANO
- SPOLU handling: either
  - Treat as a **single line** if feed reports coalition combined; or
  - Sum member parties (ODS, KDU-ČSL, TOP09) from feed. Configure in `config/parties.yaml`.

---

## 6. UI/UX Notes
- **Simple first**: tables for partials & coalition odds.
- Sidebar controls:
  - Number of simulations
  - Fixture counting profile
  - Manual refresh (cache TTL ~60s; live cadence aligned to feed update)
- Future (nice-to-have):
  - Sparkline of odds over time (write odds snapshots to `data/processed/`).
  - Per-coalition histograms or fan charts.

---

## 7. Dev Setup & Commands
- **Python**: ≥3.11 (Poetry-managed)
- **Install**:
  ```
  poetry env use 3.11
  poetry install
  cp .env.example .env    # choose DATA_MODE
  ```
- **Run**:
  ```
  poetry run streamlit run src/cz_elections_live/streamlit_app.py
  # or
  poetry run cz-live
  ```
- **Quality**:
  ```
  poetry run pytest -q
  poetry run ruff check .
  poetry run black .
  poetry run mypy src
  ```

---

## 8. Deployment & Ops
- **Local**: Streamlit on dev machine.
- **Docker**: `docker/Dockerfile` provided; expose `8501`.
- **Caching & Rate Limiting**:
  - Streamlit cache (`@st.cache_data`) TTL ≈ 60s.
  - Be a polite client: avoid sub-minute polling if the official feed cadence is 60s.
- **Observability**:
  - Basic logging helper in `utils/logging.py`.
  - (Optional) Write periodic snapshots (CSV/JSON) of coalition odds for backtesting.

---

## 9. Testing Strategy
- **Adapters**:
  - Fixture adapter: deterministic snapshots for profiles.
  - Live adapter: tolerant XML parsing (namespaces, missing fields); falls back to empty df + info message.
- **Simulator**:
  - Shape tests (N×P matrices), stability over seeds.
  - Sanity: identical inputs → stable means; edge cases with tiny parties near threshold.
- **Seats**:
  - D’Hondt allocator: known toy cases with hand-checked seat splits.

---

## 10. Roadmap
- **v0 (today)**:
  - Fixture mode working; baseline Monte Carlo; national D’Hondt approximation; coalition odds table.
- **v0.1**:
  - Live adapter verified against real XML (tag names/namespace); progress-informed remaining votes.
- **v0.2**:
  - Per-region modeling (variance by region completeness); better EST of total remaining (precincts/turnout).
- **v1.0**:
  - Coalition-aware thresholds (5/7/11); two-round legal allocation if applicable; basic charts (hist/line).
- **Later**:
  - Precinct-level model; hierarchical priors; uncertainty bands per coalition; interactive coalition builder.

---

## 11. Risks & Mitigations
- **Feed schema drift / namespace changes** → Keep XPath & NS configurable; add quick schema validator.
- **Counting-order bias** (urban late) → Use profiles; v1: region-aware variance/biasing.
- **Threshold cliffs** (parties at ~5%) → Monte Carlo captures; display **probability of crossing 5%** as a separate stat (nice-to-have).
- **Performance** at high N → Vectorize; limit party set to plausible entrants; cache base arrays.
- **Legal allocation nuances** → Start with approximation; document deviation; plan v1 exactness pass.

---

## 12. Open Questions
- Does the 2025 feed report **SPOLU** as a combined entity or by member parties?
- Exact **update cadence** and presence of **per-region topline** in the live XML?
- Out-of-country (**zahraničí**) timing and whether to include it early or wait until merged totals?

---

## 13. Glossary
- **D’Hondt**: highest averages method for proportional seat allocation.
- **Dirichlet–multinomial**: distribution for overdispersed counts, modeling uncertainty around party shares.
- **Majority**: 101 of 200 seats in the Chamber of Deputies.

---

## 14. File Map (key paths)
- `src/cz_elections_live/streamlit_app.py` — UI entrypoint
- `src/cz_elections_live/data_adapters/` — live + fixture sources
- `src/cz_elections_live/model/` — Monte Carlo + seats
- `config/coalitions.yaml` — coalition presets
- `config/parties.yaml` — party metadata
- `data/raw/` — sample 2021-like CSVs
- `docs/DESIGN.md` — this document
