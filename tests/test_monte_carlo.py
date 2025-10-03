import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from cz_elections_live.model.monte_carlo import (
    dirichlet_multinomial_draw,
    SimulationResult,
    simulate_outcomes,
)
from cz_elections_live.model.seats import allocate_seats_dhondt_regional


class TestDirichletMultinomialDraw:
    """Test the Dirichlet-multinomial distribution sampling."""
    
    def test_basic_draw(self):
        """Test basic functionality with known parameters."""
        np.random.seed(42)
        n = 1000
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        
        result = dirichlet_multinomial_draw(n, probs)
        
        assert len(result) == len(probs)
        assert result.sum() == n
        assert all(x >= 0 for x in result)
        
    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        n = 500
        probs = np.array([0.6, 0.4])
        
        np.random.seed(123)
        result1 = dirichlet_multinomial_draw(n, probs)
        
        np.random.seed(123)
        result2 = dirichlet_multinomial_draw(n, probs)
        
        np.testing.assert_array_equal(result1, result2)
        
    def test_zero_probabilities(self):
        """Test handling of zero probabilities."""
        n = 100
        probs = np.array([0.8, 0.0, 0.2])
        
        result = dirichlet_multinomial_draw(n, probs)
        
        assert result.sum() == n
        assert result[1] == 0  # Zero probability should yield zero votes
        assert result[0] + result[2] == n
        
    def test_small_alpha_scale(self):
        """Test with very small alpha scale (high variance)."""
        np.random.seed(42)
        n = 1000
        probs = np.array([0.5, 0.5])
        
        # Small alpha scale should create high variance
        result = dirichlet_multinomial_draw(n, probs, alpha_scale=1.0)
        
        assert result.sum() == n
        # High variance might push one party to extreme values
        assert any(x > 0 for x in result)


class TestSimulationResult:
    """Test the SimulationResult class."""
    
    def test_coalition_stats_basic(self):
        """Test basic coalition statistics calculation."""
        parties = ["A", "B", "C", "D"]
        # Create deterministic seat matrix: A=60, B=40, C=30, D=20 in all sims
        seats_matrix = np.array([[60, 40, 30, 20]] * 1000)
        result = SimulationResult(parties, seats_matrix)
        
        # Test single party
        p, exp = result.coalition_stats(["A"], majority=101)
        assert p == 0.0  # A alone never reaches 101
        assert exp == 60.0  # Always gets 60 seats
        
        # Test coalition that reaches majority
        p, exp = result.coalition_stats(["A", "B"], majority=101)
        assert p == 0.0  # A+B = 100, never reaches 101
        assert exp == 100.0
        
        # Test coalition that exceeds majority
        p, exp = result.coalition_stats(["A", "B", "C"], majority=101)
        assert p == 1.0  # A+B+C = 130, always exceeds 101
        assert exp == 130.0
        
    def test_coalition_stats_partial_majority(self):
        """Test coalition statistics with partial majority."""
        parties = ["A", "B", "C"]
        # Mixed results: sometimes A+B reaches majority, sometimes not
        seats_matrix = np.array([
            [60, 50, 30],  # A+B=110 >= 101
            [50, 40, 40],  # A+B=90 < 101
            [55, 55, 20],  # A+B=110 >= 101
        ] * 333)  # 1000 total simulations
        result = SimulationResult(parties, seats_matrix)
        
        p, exp = result.coalition_stats(["A", "B"], majority=101)
        assert p == pytest.approx(2/3, rel=0.1)  # 2/3 of simulations reach majority
        assert exp == pytest.approx(100.0, rel=0.1)  # Average of 110, 90, 110
        
    def test_coalition_stats_missing_party(self):
        """Test coalition stats with party not in simulation."""
        parties = ["A", "B", "C"]
        seats_matrix = np.array([[60, 40, 30]] * 1000)
        result = SimulationResult(parties, seats_matrix)
        
        # Party "D" doesn't exist in simulation
        p, exp = result.coalition_stats(["A", "D"], majority=101)
        assert p == 0.0
        assert exp == 60.0  # Only A's seats count


class TestSimulateOutcomes:
    """Test the main simulation function."""
    
    def create_test_data(self):
        """Create standardized test data."""
        return pd.DataFrame({
            "party_code": ["A", "B", "C", "D"],
            "party_name": ["Party A", "Party B", "Party C", "Party D"],
            "votes_reported": [1000, 800, 600, 400],
            "pct_reported": [35.7, 28.6, 21.4, 14.3]
        })
    
    def test_simulation_basic(self):
        """Test basic simulation functionality."""
        df_partial = self.create_test_data()
        
        with patch('numpy.random.seed'):
            result = simulate_outcomes(
                partial=df_partial,
                n_sims=100,
                seat_allocator=allocate_seats_dhondt_regional,
                thresholds={"single": 0.05}
            )
        
        assert len(result.parties) == 4
        assert result.seats.shape == (100, 4)
        assert all(result.seats.sum(axis=1) == 200)  # All simulations allocate 200 seats
        
    def test_simulation_deterministic_with_seed(self):
        """Test that simulation is deterministic with fixed seed."""
        df_partial = self.create_test_data()
        
        # Mock numpy.random to ensure deterministic behavior
        with patch('numpy.random.dirichlet') as mock_dirichlet, \
             patch('numpy.random.multinomial') as mock_multinomial:
            
            # Set up deterministic responses
            mock_dirichlet.return_value = np.array([0.35, 0.29, 0.21, 0.15])
            mock_multinomial.return_value = np.array([350, 290, 210, 150])
            
            result1 = simulate_outcomes(
                partial=df_partial,
                n_sims=10,
                seat_allocator=allocate_seats_dhondt_regional,
                thresholds={"single": 0.05}
            )
            
            result2 = simulate_outcomes(
                partial=df_partial,
                n_sims=10,
                seat_allocator=allocate_seats_dhondt_regional,
                thresholds={"single": 0.05}
            )
            
            np.testing.assert_array_equal(result1.seats, result2.seats)
            
    def test_simulation_threshold_filtering(self):
        """Test that parties below threshold get zero seats."""
        # Create data where one party is below 5% threshold
        df_partial = pd.DataFrame({
            "party_code": ["A", "B", "C", "D"],
            "party_name": ["Party A", "Party B", "Party C", "Party D"],
            "votes_reported": [1000, 800, 600, 50],  # D is very small
            "pct_reported": [40.8, 32.7, 24.5, 2.0]  # D is below 5%
        })
        
        with patch('numpy.random.dirichlet') as mock_dirichlet, \
             patch('numpy.random.multinomial') as mock_multinomial:
            
            # Ensure D stays below threshold
            mock_dirichlet.return_value = np.array([0.41, 0.33, 0.25, 0.01])
            mock_multinomial.return_value = np.array([410, 330, 250, 10])
            
            result = simulate_outcomes(
                partial=df_partial,
                n_sims=1,
                seat_allocator=allocate_seats_dhondt_regional,
                thresholds={"single": 0.05}
            )
            
            # Party D should get 0 seats due to threshold
            assert result.seats[0, 3] == 0  # D is at index 3
            
    def test_simulation_edge_case_empty_data(self):
        """Test simulation with empty partial data."""
        df_empty = pd.DataFrame(columns=["party_code", "party_name", "votes_reported", "pct_reported"])
        
        # Empty data should return empty result
        result = simulate_outcomes(
            partial=df_empty,
            n_sims=10,
            seat_allocator=allocate_seats_dhondt_regional,
            thresholds={"single": 0.05}
        )
        
        assert len(result.parties) == 0
        assert result.seats.shape == (10, 0)
            
    def test_simulation_edge_case_single_party(self):
        """Test simulation with only one party."""
        df_single = pd.DataFrame({
            "party_code": ["A"],
            "party_name": ["Party A"],
            "votes_reported": [1000],
            "pct_reported": [100.0]
        })
        
        result = simulate_outcomes(
            partial=df_single,
            n_sims=10,
            seat_allocator=allocate_seats_dhondt_regional,
            thresholds={"single": 0.05}
        )
        
        assert len(result.parties) == 1
        assert all(result.seats[:, 0] == 200)  # Single party gets all seats
        
    def test_simulation_performance(self):
        """Test simulation performance with larger numbers."""
        df_partial = self.create_test_data()
        
        import time
        start_time = time.time()
        
        result = simulate_outcomes(
            partial=df_partial,
            n_sims=5000,
            seat_allocator=allocate_seats_dhondt_regional,
            thresholds={"single": 0.05}
        )
        
        elapsed = time.time() - start_time
        
        # Should complete 5000 simulations in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert result.seats.shape == (5000, 4)
