from typing import Dict, List

import numpy as np


def allocate_seats_dhondt_regional(
    parties: List[str], votes, eligible_mask, seats_total: int = 200
) -> Dict[str, int]:
    # Approximate national D’Hondt as fast baseline
    v = np.array(votes, dtype=float)
    v[~eligible_mask] = 0.0
    quotients = []
    for i, vv in enumerate(v):
        if vv <= 0:
            continue
        # Precompute quotients up to seats_total
        quotients.extend((vv / (d + 1), i) for d in range(seats_total))
    top = sorted(quotients, key=lambda t: t[0], reverse=True)[:seats_total]
    counts = np.zeros_like(v, dtype=int)
    for _, i in top:
        counts[i] += 1
    return dict(zip(parties, counts))
