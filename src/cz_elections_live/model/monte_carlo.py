from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def dirichlet_multinomial_draw(n: int, probs: np.ndarray, alpha_scale: float = 300) -> np.ndarray:
    alpha = np.maximum(1e-9, alpha_scale * probs)
    theta = np.random.dirichlet(alpha)
    return np.random.multinomial(n, theta)


class SimulationResult:
    def __init__(self, parties: List[str], seats_matrix: np.ndarray):
        self.parties = parties
        self.seats = seats_matrix

    def coalition_stats(
        self, coalition_parties: List[str], majority: int = 101
    ) -> Tuple[float, float]:
        idx = [self.parties.index(p) for p in coalition_parties if p in self.parties]
        if not idx:
            return 0.0, 0.0
        total = self.seats[:, idx].sum(axis=1)
        return float((total >= majority).mean()), float(total.mean())


def simulate_outcomes(
    partial: pd.DataFrame,
    n_sims: int,
    seat_allocator: Callable[[List[str], np.ndarray, np.ndarray], Dict[str, int]],
    thresholds: Dict[str, float],
    est_total_factor: float = 1.8,
) -> SimulationResult:
    parties = partial["party_code"].tolist()
    votes = partial["votes_reported"].to_numpy(dtype=float)
    counted = float(votes.sum())
    est_total = int(counted * est_total_factor)  # replace with live progress as available
    remaining = max(0, est_total - int(counted))
    probs = votes / max(counted, 1.0)

    seats_matrix = np.zeros((n_sims, len(parties)), dtype=int)
    for k in range(n_sims):
        add = dirichlet_multinomial_draw(remaining, probs)
        final_votes = votes + add
        pct = final_votes / final_votes.sum()
        eligible_mask = pct >= thresholds["single"]  # simple single-party case
        seats = seat_allocator(parties, final_votes, eligible_mask)
        seats_matrix[k, :] = [seats.get(p, 0) for p in parties]
    return SimulationResult(parties, seats_matrix)
