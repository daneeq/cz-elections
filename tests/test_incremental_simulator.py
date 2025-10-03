import time
import pytest
import pandas as pd
from unittest.mock import patch

from cz_elections_live.data_adapters.incremental_2021 import (
    IncrementalDataSimulator,
    SimulationConfig,
    create_simulator,
)


class TestIncrementalDataSimulator:
    """Test the incremental data simulator."""
    
    def test_simulator_initialization(self):
        """Test basic simulator initialization."""
        config = SimulationConfig(duration_minutes=10.0)
        simulator = IncrementalDataSimulator(config)
        
        assert simulator.config.duration_minutes == 10.0
        assert simulator.start_time is None
        assert simulator.current_snapshot is None
        
    def test_load_target_data(self):
        """Test loading of target 2021 data."""
        simulator = IncrementalDataSimulator()
        target_data = simulator._load_target_data()
        
        assert isinstance(target_data, pd.DataFrame)
        assert "party_code" in target_data.columns
        assert "party_name" in target_data.columns
        assert "votes" in target_data.columns
        assert len(target_data) > 0
        
    def test_load_region_data(self):
        """Test loading of regional data."""
        simulator = IncrementalDataSimulator()
        region_data = simulator._load_region_data()
        
        assert isinstance(region_data, pd.DataFrame)
        assert "region_code" in region_data.columns
        assert "party_code" in region_data.columns
        assert "votes" in region_data.columns
        assert len(region_data) > 0
        
    def test_start_simulation(self):
        """Test starting a simulation."""
        simulator = IncrementalDataSimulator()
        simulator.start_simulation()
        
        assert simulator.start_time is not None
        assert simulator.current_snapshot is not None
        assert isinstance(simulator.current_snapshot, pd.DataFrame)
        
    def test_get_current_partial_before_start(self):
        """Test getting partial data before starting simulation."""
        simulator = IncrementalDataSimulator()
        
        # Should auto-start simulation
        result = simulator.get_current_partial()
        
        assert isinstance(result, pd.DataFrame)
        assert "party_code" in result.columns
        assert "votes_reported" in result.columns
        assert "pct_reported" in result.columns
        assert simulator.start_time is not None
        
    def test_get_current_partial_at_progress(self):
        """Test getting partial data at specific progress levels."""
        simulator = IncrementalDataSimulator()
        
        # Test initial progress (0.0)
        result_0 = simulator._generate_partial_at_progress(0.0)
        assert isinstance(result_0, pd.DataFrame)
        assert len(result_0) > 0
        
        # Test mid progress (0.5)
        result_50 = simulator._generate_partial_at_progress(0.5)
        assert isinstance(result_50, pd.DataFrame)
        assert len(result_50) > 0
        
        # Test final progress (1.0)
        result_100 = simulator._generate_partial_at_progress(1.0)
        assert isinstance(result_100, pd.DataFrame)
        assert len(result_100) > 0
        
        # Final results should have more votes than initial
        total_initial = result_0["votes_reported"].sum()
        total_final = result_100["votes_reported"].sum()
        assert total_final > total_initial
        
    def test_get_regions_at_progress(self):
        """Test region reporting logic at different progress levels."""
        simulator = IncrementalDataSimulator()
        
        # At 0% progress, should have minimal regions
        regions_0 = simulator._get_regions_at_progress(0.0)
        assert len(regions_0) == 0
        
        # At 50% progress, should have some regions
        regions_50 = simulator._get_regions_at_progress(0.5)
        assert len(regions_50) > 0
        
        # At 100% progress, should have all regions
        all_regions = simulator.region_data["region_code"].unique().tolist()
        regions_100 = simulator._get_regions_at_progress(1.0)
        assert len(regions_100) >= len(all_regions) * 0.8  # Most regions reported
        
    def test_add_realistic_noise(self):
        """Test noise addition to results."""
        simulator = IncrementalDataSimulator()
        
        # Create test data
        test_data = pd.DataFrame({
            "party_code": ["A", "B", "C"],
            "votes_reported": [1000, 800, 600],
            "party_name": ["Party A", "Party B", "Party C"]
        })
        
        # Add noise
        noisy_data = simulator._add_realistic_noise(test_data, 0.5, ["CZ0311"])
        
        assert isinstance(noisy_data, pd.DataFrame)
        assert len(noisy_data) == len(test_data)
        assert "votes_reported" in noisy_data.columns
        
        # Vote counts should be non-negative
        assert all(noisy_data["votes_reported"] >= 0)
        
    def test_progress_info(self):
        """Test progress information retrieval."""
        simulator = IncrementalDataSimulator()
        
        # Before starting
        info_before = simulator.get_progress_info()
        assert info_before["progress"] == 0.0
        assert info_before["elapsed_minutes"] == 0.0
        
        # After starting
        simulator.start_simulation()
        time.sleep(0.1)  # Small delay
        
        info_after = simulator.get_progress_info()
        assert info_after["progress"] >= 0.0
        assert info_after["elapsed_minutes"] > 0.0
        assert "remaining_minutes" in info_after
        assert "is_complete" in info_after
        
    def test_reset_simulation(self):
        """Test simulation reset functionality."""
        simulator = IncrementalDataSimulator()
        simulator.start_simulation()
        
        assert simulator.start_time is not None
        assert simulator.current_snapshot is not None
        
        simulator.reset_simulation()
        
        assert simulator.start_time is None
        assert simulator.current_snapshot is None
        
    def test_simulation_config(self):
        """Test simulation configuration."""
        config = SimulationConfig(
            duration_minutes=30.0,
            update_interval_seconds=15,
            noise_factor=0.03,
            urban_late_bias=0.08
        )
        
        assert config.duration_minutes == 30.0
        assert config.update_interval_seconds == 15
        assert config.noise_factor == 0.03
        assert config.urban_late_bias == 0.08
        
    def test_create_simulator(self):
        """Test factory function for creating simulator."""
        simulator = create_simulator(duration_minutes=45.0)
        
        assert isinstance(simulator, IncrementalDataSimulator)
        assert simulator.config.duration_minutes == 45.0
        
    def test_final_results_structure(self):
        """Test that final results have correct structure."""
        simulator = IncrementalDataSimulator()
        final_results = simulator._generate_final_results()
        
        assert isinstance(final_results, pd.DataFrame)
        assert "party_code" in final_results.columns
        assert "party_name" in final_results.columns
        assert "votes_reported" in final_results.columns
        assert "pct_reported" in final_results.columns
        
        # Percentages should sum to approximately 100%
        assert abs(final_results["pct_reported"].sum() - 100.0) < 1.0
        
    def test_regional_counting_pattern(self):
        """Test realistic regional counting patterns."""
        simulator = IncrementalDataSimulator()
        
        # Rural regions should report first
        regions_early = simulator._get_regions_at_progress(0.2)
        rural_regions = ["CZ0311", "CZ0312", "CZ0321", "CZ0322"]
        
        # At least some rural regions should be reported early
        assert any(region in regions_early for region in rural_regions)
        
        # Prague should report late
        prague_regions = ["CZ0100"]
        regions_late = simulator._get_regions_at_progress(0.9)
        
        # Prague should be in late reporting regions
        assert any(region in regions_late for region in prague_regions)
        
    def test_simulation_timing_basic(self):
        """Test basic simulation timing functionality."""
        config = SimulationConfig(duration_minutes=1.0)  # 1 minute duration
        simulator = IncrementalDataSimulator(config)
        
        simulator.start_simulation()
        
        # Get initial progress info
        info = simulator.get_progress_info()
        
        # Should have valid progress info
        assert "progress" in info
        assert "elapsed_minutes" in info
        assert "remaining_minutes" in info
        assert "is_complete" in info
        assert isinstance(info["progress"], float)
        assert info["progress"] >= 0.0
