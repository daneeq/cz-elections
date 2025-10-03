import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

from cz_elections_live.data_adapters.ps2021_fixture import load_fixture_snapshot
from cz_elections_live.data_adapters.volby2025 import fetch_current_totals


class TestPs2021Fixture:
    """Test the 2021 fixture data adapter."""
    
    def test_load_fixture_snapshot_basic(self):
        """Test basic fixture loading functionality."""
        result = load_fixture_snapshot("urban_late_55")
        
        # Should return DataFrame with expected columns
        assert isinstance(result, pd.DataFrame)
        assert "party_code" in result.columns
        assert "party_name" in result.columns
        assert "votes_reported" in result.columns
        assert "pct_reported" in result.columns
        
        # Should have some parties
        assert len(result) > 0
        
        # Percentages should sum to approximately 100%
        assert abs(result["pct_reported"].sum() - 100.0) < 1.0
        
    def test_load_fixture_different_profiles(self):
        """Test different counting profiles."""
        profiles = ["urban_late_55", "rural_first_30", "balanced_70"]
        
        for profile in profiles:
            result = load_fixture_snapshot(profile)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "votes_reported" in result.columns
            
            # All vote counts should be non-negative
            assert all(result["votes_reported"] >= 0)
            
    def test_load_fixture_data_consistency(self):
        """Test that fixture data is internally consistent."""
        result = load_fixture_snapshot("balanced_70")
        
        # Total votes should be positive
        total_votes = result["votes_reported"].sum()
        assert total_votes > 0
        
        # Percentages should be calculated correctly
        for _, row in result.iterrows():
            expected_pct = (row["votes_reported"] / total_votes) * 100.0
            assert abs(row["pct_reported"] - expected_pct) < 0.1
            
    def test_load_fixture_party_codes(self):
        """Test that expected party codes are present."""
        result = load_fixture_snapshot("urban_late_55")
        
        # Should contain major parties from 2021
        party_codes = result["party_code"].tolist()
        assert "ANO" in party_codes
        assert "SPOLU" in party_codes
        
    def test_load_fixture_empty_profile(self):
        """Test behavior with unknown profile."""
        # Should default to rural profile
        result = load_fixture_snapshot("unknown_profile")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestVolby2025:
    """Test the live 2025 data adapter."""
    
    def test_fetch_current_totals_mock_xml(self):
        """Test XML parsing with mock data."""
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <root xmlns="http://www.volby.cz/ps/">
            <STRANA KSTRANA="ANO" NAZEV="ANO" HLASY="1450000" PROCENT="27,5"/>
            <STRANA KSTRANA="SPOLU" NAZEV="SPOLU" HLASY="1500000" PROCENT="28,4"/>
            <STRANA KSTRANA="PIR" NAZEV="Pirates" HLASY="420000" PROCENT="8,0"/>
        </root>"""
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_xml.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = fetch_current_totals()
            
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "ANO" in result["party_code"].values
        assert "SPOLU" in result["party_code"].values
        assert "PIR" in result["party_code"].values
        
        # Check vote counts
        ano_row = result[result["party_code"] == "ANO"].iloc[0]
        assert ano_row["votes_reported"] == 1450000
        assert ano_row["pct_reported"] == 27.5
        
    def test_fetch_current_totals_empty_xml(self):
        """Test handling of empty XML response."""
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <root xmlns="http://www.volby.cz/ps/">
        </root>"""
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_xml.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = fetch_current_totals()
            
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "party_code" in result.columns
        assert "timestamp" in result.columns
        
    def test_fetch_current_totals_malformed_xml(self):
        """Test handling of malformed XML."""
        malformed_xml = "<root><STRANA KSTRANA="  # Incomplete XML
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.content = malformed_xml.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with pytest.raises(ET.ParseError):
                fetch_current_totals()
                
    def test_fetch_current_totals_missing_attributes(self):
        """Test handling of missing XML attributes."""
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <root xmlns="http://www.volby.cz/ps/">
            <STRANA KSTRANA="ANO" NAZEV="ANO"/>
            <STRANA KSTRANA="SPOLU" HLASY="1500000" PROCENT="28,4"/>
        </root>"""
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_xml.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = fetch_current_totals()
            
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        
        # First party should have default values for missing attributes
        ano_row = result[result["party_code"] == "ANO"].iloc[0]
        assert ano_row["votes_reported"] == 0  # Default value
        assert ano_row["pct_reported"] == 0.0  # Default value
        
    def test_fetch_current_totals_network_error(self):
        """Test handling of network errors."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            with pytest.raises(Exception, match="Network error"):
                fetch_current_totals()
                
    def test_fetch_current_totals_timestamp(self):
        """Test that timestamp is added correctly."""
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <root xmlns="http://www.volby.cz/ps/">
            <STRANA KSTRANA="ANO" NAZEV="ANO" HLASY="1450000" PROCENT="27,5"/>
        </root>"""
        
        with patch('requests.get') as mock_get, \
             patch('pandas.Timestamp.utcnow') as mock_timestamp:
            
            mock_timestamp.return_value = pd.Timestamp("2025-01-15 20:30:00")
            mock_response = MagicMock()
            mock_response.content = mock_xml.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = fetch_current_totals()
            
        assert "timestamp" in result.columns
        assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-15 20:30:00")
