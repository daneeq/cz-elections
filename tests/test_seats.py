import numpy as np
import pytest

from cz_elections_live.model.seats import allocate_seats_dhondt_regional


class TestAllocateSeatsDhondtRegional:
    """Test the D'Hondt seat allocation algorithm."""
    
    def test_basic_allocation(self):
        """Test basic proportional allocation."""
        parties = ["A", "B", "C"]
        votes = [600, 400, 200]  # 6:4:2 ratio
        eligible_mask = np.array([True, True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=12)
        
        # Should allocate proportionally: A=6, B=4, C=2
        assert result["A"] == 6
        assert result["B"] == 4
        assert result["C"] == 2
        assert sum(result.values()) == 12
        
    def test_dhondt_quotients(self):
        """Test D'Hondt method with known quotients."""
        parties = ["A", "B", "C"]
        votes = [1000, 500, 250]  # Clear 2:1:0.5 ratio
        eligible_mask = np.array([True, True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=7)
        
        # With D'Hondt: A should get 4, B should get 2, C should get 1
        # Quotients: A(1000,500,333,250), B(500,250,167), C(250,125)
        assert result["A"] == 4
        assert result["B"] == 2
        assert result["C"] == 1
        assert sum(result.values()) == 7
        
    def test_threshold_filtering(self):
        """Test that ineligible parties get zero seats."""
        parties = ["A", "B", "C", "D"]
        votes = [1000, 800, 600, 50]  # D is very small
        eligible_mask = np.array([True, True, True, False])  # D not eligible
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask)
        
        assert result["D"] == 0  # Ineligible party gets no seats
        assert result["A"] > 0   # Eligible parties get seats
        assert result["B"] > 0
        assert result["C"] > 0
        assert sum(result.values()) == 200  # All 200 seats allocated
        
    def test_exact_proportionality(self):
        """Test with votes that should give exact proportional results."""
        parties = ["A", "B", "C"]
        votes = [300, 200, 100]  # 3:2:1 ratio
        eligible_mask = np.array([True, True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=6)
        
        # Should be exactly proportional: A=3, B=2, C=1
        assert result["A"] == 3
        assert result["B"] == 2
        assert result["C"] == 1
        
    def test_tie_breaking(self):
        """Test behavior when parties have equal votes."""
        parties = ["A", "B"]
        votes = [1000, 1000]  # Equal votes
        eligible_mask = np.array([True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=3)
        
        # Should allocate seats fairly (1-2 or 2-1 split)
        assert result["A"] + result["B"] == 3
        assert abs(result["A"] - result["B"]) <= 1  # Difference should be at most 1
        
    def test_zero_votes(self):
        """Test handling of parties with zero votes."""
        parties = ["A", "B", "C"]
        votes = [1000, 0, 500]  # B has zero votes
        eligible_mask = np.array([True, True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=6)
        
        assert result["B"] == 0  # Zero votes should get zero seats
        assert result["A"] > 0   # Non-zero votes get seats
        assert result["C"] > 0
        assert sum(result.values()) == 6
        
    def test_single_party(self):
        """Test allocation with only one eligible party."""
        parties = ["A", "B", "C"]
        votes = [1000, 500, 300]
        eligible_mask = np.array([True, False, False])  # Only A eligible
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask)
        
        assert result["A"] == 200  # Gets all seats
        assert result["B"] == 0    # Ineligible
        assert result["C"] == 0    # Ineligible
        
    def test_all_ineligible(self):
        """Test edge case where no parties are eligible."""
        parties = ["A", "B", "C"]
        votes = [1000, 500, 300]
        eligible_mask = np.array([False, False, False])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask)
        
        # All parties should get zero seats
        assert all(seats == 0 for seats in result.values())
        
    def test_large_seat_count(self):
        """Test with larger number of seats (200 - real Czech case)."""
        parties = ["A", "B", "C", "D"]
        votes = [4000000, 3000000, 2000000, 1000000]  # 4:3:2:1 ratio
        eligible_mask = np.array([True, True, True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask)
        
        # Should allocate approximately 4:3:2:1 ratio of 200 seats
        # Expected: A=80, B=60, C=40, D=20 (approximately)
        assert result["A"] > 70 and result["A"] < 90
        assert result["B"] > 50 and result["B"] < 70
        assert result["C"] > 30 and result["C"] < 50
        assert result["D"] > 10 and result["D"] < 30
        assert sum(result.values()) == 200
        
    def test_very_small_differences(self):
        """Test with very small vote differences."""
        parties = ["A", "B", "C"]
        votes = [1001, 1000, 999]  # Very close votes
        eligible_mask = np.array([True, True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=6)
        
        # Should still allocate all seats
        assert sum(result.values()) == 6
        assert all(seats >= 0 for seats in result.values())
        
    def test_fractional_votes(self):
        """Test with fractional vote counts."""
        parties = ["A", "B"]
        votes = [1000.5, 999.5]  # Fractional votes
        eligible_mask = np.array([True, True])
        
        result = allocate_seats_dhondt_regional(parties, votes, eligible_mask, seats_total=3)
        
        # Should handle fractional votes correctly
        assert sum(result.values()) == 3
        assert all(seats >= 0 for seats in result.values())
