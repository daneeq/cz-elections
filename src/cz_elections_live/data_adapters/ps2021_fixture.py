import pandas as pd

from cz_elections_live.sim_scenarios.progress_profiles import choose_regions_for_profile


def load_fixture_snapshot(progress: str = "urban_late_55") -> pd.DataFrame:
    nat = pd.read_csv("data/raw/ps2021_national_totals.csv")  # party_code,party_name,votes
    reg = pd.read_csv("data/raw/ps2021_region_totals.csv")  # region_code,party_code,votes

    counted = set(choose_regions_for_profile(progress))
    reg["counted"] = reg["region_code"].isin(counted)

    partial = (
        reg[reg["counted"]]
        .groupby("party_code", as_index=False)["votes"]
        .sum()
        .rename(columns={"votes": "votes_reported"})
    )
    df = partial.merge(nat[["party_code", "party_name"]], on="party_code", how="left")
    total = df["votes_reported"].sum()
    df["pct_reported"] = df["votes_reported"] / max(total, 1) * 100.0
    return df
