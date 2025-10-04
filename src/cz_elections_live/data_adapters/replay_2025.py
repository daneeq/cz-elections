"""
Replay 2021 election data as 2025 live feed for testing.

This adapter simulates the 2025 live XML feed structure using 2021 data,
allowing us to test the system with realistic partial results before polls open.
"""
import time
import random
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


# Map 2021 party codes to 2025 party codes
PARTY_CODE_MAP = {
    "ANO": "22",
    "SPOLU": "11",
    "SPD": "6",
    "STAN": "23",
    "PIR": "16",
    "CSSD": "7",
}


@dataclass
class ReplayConfig:
    """Configuration for 2025 replay mode."""
    duration_minutes: float = 3.0  # 3 minutes for fast testing (1% per 1.8 seconds)
    noise_factor: float = 0.03  # 3% random noise


class Replay2025Simulator:
    """
    Simulates 2025 live election feed using 2021 regional data.

    Replays 2021 results with:
    - 2025 party codes (numeric)
    - Realistic counting order (rural first, Prague last)
    - Added noise to simulate uncertainty
    """

    def __init__(self, config: Optional[ReplayConfig] = None):
        self.config = config or ReplayConfig()
        self.start_time = None
        self.target_data = self._load_target_data()
        self.region_data = self._load_region_data()

    def _load_target_data(self) -> pd.DataFrame:
        """Load 2021 final results and map to 2025 codes."""
        df = pd.read_csv("data/raw/ps2021_national_totals.csv")

        # Map to 2025 party codes
        df["party_code_2025"] = df["party_code"].map(PARTY_CODE_MAP)
        df = df.dropna(subset=["party_code_2025"])  # Drop unmapped parties

        return df

    def _load_region_data(self) -> pd.DataFrame:
        """Load regional data for realistic counting simulation."""
        df = pd.read_csv("data/raw/ps2021_region_totals.csv")

        # Map to 2025 party codes
        df["party_code_2025"] = df["party_code"].map(PARTY_CODE_MAP)
        df = df.dropna(subset=["party_code_2025"])

        return df

    def start_simulation(self) -> None:
        """Start the replay timer."""
        self.start_time = time.time()

    def reset_simulation(self) -> None:
        """Reset the replay."""
        self.start_time = None

    def get_current_partial(self) -> pd.DataFrame:
        """
        Get current partial results based on elapsed time.

        Returns:
            DataFrame with 2025 party codes and partial vote counts
        """
        if self.start_time is None:
            self.start_simulation()

        elapsed = time.time() - self.start_time
        progress = min(elapsed / (self.config.duration_minutes * 60), 1.0)

        return self._generate_partial_at_progress(progress)

    def _generate_partial_at_progress(self, progress: float) -> pd.DataFrame:
        """Generate partial results at given progress (0.0 to 1.0)."""
        if progress <= 0.0:
            return self._generate_initial_snapshot()
        elif progress >= 1.0:
            return self._generate_final_results()

        # Determine which regions have reported
        regions_reported = self._get_regions_at_progress(progress)

        # Get partial data from reported regions
        partial_data = (
            self.region_data[self.region_data["region_code"].isin(regions_reported)]
            .groupby("party_code_2025", as_index=False)["votes"]
            .sum()
            .rename(columns={"votes": "votes_reported", "party_code_2025": "party_code"})
        )

        # Add party names
        party_names = self.target_data.set_index("party_code_2025")["party_name"].to_dict()
        partial_data["party_name"] = partial_data["party_code"].map(party_names)

        # Add realistic noise
        partial_data = self._add_realistic_noise(partial_data, progress)

        # Calculate percentages
        total = partial_data["votes_reported"].sum()
        partial_data["pct_reported"] = (partial_data["votes_reported"] / max(total, 1)) * 100.0

        return partial_data[["party_code", "party_name", "votes_reported", "pct_reported"]]

    def _get_regions_at_progress(self, progress: float) -> list:
        """
        Determine which regions have reported at given progress.

        Realistic counting order:
        - Rural areas first (0-20%)
        - Medium cities (20-60%)
        - Prague last (60-100%)
        - Other regions gradually fill in
        """
        all_regions = self.region_data["region_code"].unique().tolist()

        rural_regions = ["CZ0311", "CZ0312", "CZ0321", "CZ0322"]
        medium_cities = ["CZ0201", "CZ0202", "CZ0401"]
        prague = ["CZ0100"]

        reported_regions = []

        # Start showing data immediately
        if progress >= 0.02:  # At 2% (~12 seconds with 10min duration)
            reported_regions.extend(rural_regions)

        if progress >= 0.25:  # At 25%
            reported_regions.extend(medium_cities)

        if progress >= 0.60:  # At 60%
            reported_regions.extend(prague)

        # Add randomness for other regions (start earlier)
        if progress > 0.15:
            for region in all_regions:
                if region not in reported_regions:
                    # More aggressive progression
                    if random.random() < (progress - 0.15) * 2.0:
                        reported_regions.append(region)

        return reported_regions

    def _add_realistic_noise(self, df: pd.DataFrame, progress: float) -> pd.DataFrame:
        """Add realistic noise to vote counts."""
        df = df.copy()
        noise_scale = self.config.noise_factor * (1.0 - progress * 0.5)

        df["votes_reported"] = df["votes_reported"].astype(float)
        for idx, row in df.iterrows():
            noise = random.gauss(0, row["votes_reported"] * noise_scale)
            df.loc[idx, "votes_reported"] = max(0, row["votes_reported"] + noise)

        return df

    def _generate_initial_snapshot(self) -> pd.DataFrame:
        """Generate initial snapshot (first few regions reporting)."""
        # Start with more regions so we have visible data immediately
        rural_regions = ["CZ0311", "CZ0312", "CZ0321"]

        initial_data = (
            self.region_data[self.region_data["region_code"].isin(rural_regions)]
            .groupby("party_code_2025", as_index=False)["votes"]
            .sum()
            .rename(columns={"votes": "votes_reported", "party_code_2025": "party_code"})
        )

        party_names = self.target_data.set_index("party_code_2025")["party_name"].to_dict()
        initial_data["party_name"] = initial_data["party_code"].map(party_names)

        total = initial_data["votes_reported"].sum()
        initial_data["pct_reported"] = (initial_data["votes_reported"] / max(total, 1)) * 100.0

        return initial_data[["party_code", "party_name", "votes_reported", "pct_reported"]]

    def _generate_final_results(self) -> pd.DataFrame:
        """Generate final results (close to 2021 actual)."""
        df = self.target_data.copy()
        df = df.rename(columns={"party_code_2025": "party_code", "votes": "votes_reported"})

        # Small final noise
        df["votes_reported"] = df["votes_reported"].astype(float)
        for idx, row in df.iterrows():
            noise = random.gauss(0, row["votes_reported"] * 0.005)
            df.loc[idx, "votes_reported"] = max(0, row["votes_reported"] + noise)

        total = df["votes_reported"].sum()
        df["pct_reported"] = (df["votes_reported"] / total) * 100.0

        return df[["party_code", "party_name", "votes_reported", "pct_reported"]]

    def get_progress_info(self) -> Dict[str, float]:
        """Get current progress information."""
        if self.start_time is None:
            return {"progress": 0.0, "elapsed_minutes": 0.0, "remaining_minutes": 0.0, "is_complete": False}

        elapsed = time.time() - self.start_time
        elapsed_minutes = elapsed / 60.0
        progress = min(elapsed / (self.config.duration_minutes * 60), 1.0)
        remaining_minutes = max(0, self.config.duration_minutes - elapsed_minutes)

        return {
            "progress": progress,
            "elapsed_minutes": elapsed_minutes,
            "remaining_minutes": remaining_minutes,
            "is_complete": progress >= 1.0
        }


def create_replay_simulator(duration_minutes: float = 3.0) -> Replay2025Simulator:
    """
    Create a 2025 replay simulator.

    Args:
        duration_minutes: How long the simulated count should take (default 3 hours)

    Returns:
        Configured Replay2025Simulator instance
    """
    config = ReplayConfig(duration_minutes=duration_minutes)
    return Replay2025Simulator(config)
