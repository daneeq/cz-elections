import time
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from cz_elections_live.sim_scenarios.progress_profiles import choose_regions_for_profile


@dataclass
class SimulationConfig:
    """Configuration for incremental simulation."""
    duration_minutes: float = 60.0
    update_interval_seconds: int = 10
    noise_factor: float = 0.02  # 2% random noise
    urban_late_bias: float = 0.05  # 5% bias for urban areas reporting late


class IncrementalDataSimulator:
    """
    Simulates live election night by progressively revealing 2021 data.
    
    This creates a realistic simulation of election night counting, where:
    - Rural areas report first
    - Urban areas (especially Prague) report later with different patterns
    - Results gradually converge to 2021 final results
    - Includes realistic noise and uncertainty
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.start_time = None
        self.target_data = self._load_target_data()
        self.region_data = self._load_region_data()
        self.current_snapshot = None
        
    def _load_target_data(self) -> pd.DataFrame:
        """Load the 2021 final results as target."""
        return pd.read_csv("data/raw/ps2021_national_totals.csv")
    
    def _load_region_data(self) -> pd.DataFrame:
        """Load regional data for realistic counting simulation."""
        return pd.read_csv("data/raw/ps2021_region_totals.csv")
    
    def start_simulation(self) -> None:
        """Start the simulation timer."""
        self.start_time = time.time()
        self.current_snapshot = self._generate_initial_snapshot()
        
    def get_current_partial(self) -> pd.DataFrame:
        """
        Get current partial results based on elapsed time.
        
        Returns:
            DataFrame with current partial results
        """
        if self.start_time is None:
            self.start_simulation()
            
        elapsed = time.time() - self.start_time
        progress = min(elapsed / (self.config.duration_minutes * 60), 1.0)
        
        return self._generate_partial_at_progress(progress)
    
    def _generate_initial_snapshot(self) -> pd.DataFrame:
        """Generate initial snapshot with minimal data (rural areas only)."""
        # Start with only rural regions counted
        rural_regions = ["CZ0311", "CZ0312", "CZ0321", "CZ0322"]
        
        initial_data = (
            self.region_data[self.region_data["region_code"].isin(rural_regions)]
            .groupby("party_code", as_index=False)["votes"]
            .sum()
            .rename(columns={"votes": "votes_reported"})
        )
        
        # Merge with party names
        df = initial_data.merge(
            self.target_data[["party_code", "party_name"]], 
            on="party_code", 
            how="left"
        )
        
        # Calculate percentages
        total = df["votes_reported"].sum()
        df["pct_reported"] = (df["votes_reported"] / max(total, 1)) * 100.0
        
        return df
    
    def _generate_partial_at_progress(self, progress: float) -> pd.DataFrame:
        """
        Generate partial results at given progress (0.0 to 1.0).
        
        Args:
            progress: Progress from 0.0 (start) to 1.0 (complete)
            
        Returns:
            DataFrame with partial results at this progress level
        """
        if progress <= 0.0:
            return self._generate_initial_snapshot()
        elif progress >= 1.0:
            return self._generate_final_results()
        
        # Determine which regions have reported based on progress
        regions_reported = self._get_regions_at_progress(progress)
        
        # Get partial data from reported regions
        partial_data = (
            self.region_data[self.region_data["region_code"].isin(regions_reported)]
            .groupby("party_code", as_index=False)["votes"]
            .sum()
            .rename(columns={"votes": "votes_reported"})
        )
        
        # Merge with party names
        df = partial_data.merge(
            self.target_data[["party_code", "party_name"]], 
            on="party_code", 
            how="left"
        )
        
        # Add realistic noise and bias
        df = self._add_realistic_noise(df, progress, regions_reported)
        
        # Calculate percentages
        total = df["votes_reported"].sum()
        df["pct_reported"] = (df["votes_reported"] / max(total, 1)) * 100.0
        
        return df
    
    def _get_regions_at_progress(self, progress: float) -> List[str]:
        """
        Determine which regions have reported at given progress.
        
        Uses realistic counting order:
        - Rural areas first (0-30% progress)
        - Medium cities (30-70% progress)  
        - Prague and large cities last (70-100% progress)
        """
        all_regions = self.region_data["region_code"].unique().tolist()
        
        # Define region types based on 2021 patterns
        rural_regions = ["CZ0311", "CZ0312", "CZ0321", "CZ0322"]
        medium_cities = ["CZ0201", "CZ0202", "CZ0401"]
        prague_large = ["CZ0100"]  # Prague
        
        reported_regions = []
        
        # Rural areas report first (0-30% of progress)
        if progress >= 0.1:
            reported_regions.extend(rural_regions)
        
        # Medium cities report next (30-70% of progress)
        if progress >= 0.4:
            reported_regions.extend(medium_cities)
            
        # Prague and large cities report last (70-100% of progress)
        if progress >= 0.8:
            reported_regions.extend(prague_large)
            
        # Add some randomness to make it more realistic
        if progress > 0.2:
            # Some regions report early or late
            for region in all_regions:
                if region not in reported_regions:
                    if random.random() < (progress - 0.2) * 0.3:  # Random early reporting
                        reported_regions.append(region)
        
        return reported_regions
    
    def _add_realistic_noise(self, df: pd.DataFrame, progress: float, regions_reported: List[str]) -> pd.DataFrame:
        """
        Add realistic noise and bias to simulate counting uncertainty.
        
        Args:
            df: Partial results DataFrame
            progress: Current progress (0.0 to 1.0)
            regions_reported: List of regions that have reported
            
        Returns:
            DataFrame with added noise and bias
        """
        df = df.copy()
        
        # Base noise decreases as more regions report
        noise_scale = self.config.noise_factor * (1.0 - progress * 0.7)
        
        # Add random noise to vote counts
        df["votes_reported"] = df["votes_reported"].astype(float)
        for idx, row in df.iterrows():
            noise = random.gauss(0, row["votes_reported"] * noise_scale)
            df.loc[idx, "votes_reported"] = max(0, row["votes_reported"] + noise)
        
        # Add urban bias if urban areas haven't reported yet
        prague_regions = ["CZ0100"]
        if not any(region in regions_reported for region in prague_regions):
            # Parties that do well in Prague (Pirates, STAN) get slight upward bias
            prague_parties = ["PIR", "STAN"]
            for idx, row in df.iterrows():
                if row["party_code"] in prague_parties:
                    bias = row["votes_reported"] * self.config.urban_late_bias * (1.0 - progress)
                    df.loc[idx, "votes_reported"] += bias
        
        return df
    
    def _generate_final_results(self) -> pd.DataFrame:
        """Generate final results (close to 2021 actual results)."""
        df = self.target_data.copy()
        df["votes_reported"] = df["votes"]
        
        # Add small final noise to make it realistic
        df["votes_reported"] = df["votes"].astype(float)
        for idx, row in df.iterrows():
            noise = random.gauss(0, row["votes"] * 0.005)  # 0.5% final noise
            df.loc[idx, "votes_reported"] = max(0, row["votes"] + noise)
        
        # Calculate final percentages
        total = df["votes_reported"].sum()
        df["pct_reported"] = (df["votes_reported"] / total) * 100.0
        
        return df[["party_code", "party_name", "votes_reported", "pct_reported"]]
    
    def get_progress_info(self) -> Dict[str, float]:
        """
        Get information about current simulation progress.
        
        Returns:
            Dictionary with progress information
        """
        if self.start_time is None:
            return {"progress": 0.0, "elapsed_minutes": 0.0, "remaining_minutes": 0.0}
        
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
    
    def reset_simulation(self) -> None:
        """Reset the simulation to start over."""
        self.start_time = None
        self.current_snapshot = None


def create_simulator(duration_minutes: float = 60.0) -> IncrementalDataSimulator:
    """
    Create an incremental data simulator with specified duration.
    
    Args:
        duration_minutes: How long the simulation should run (default 60 minutes)
        
    Returns:
        Configured IncrementalDataSimulator instance
    """
    config = SimulationConfig(duration_minutes=duration_minutes)
    return IncrementalDataSimulator(config)
