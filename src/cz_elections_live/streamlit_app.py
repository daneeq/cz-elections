import pandas as pd
import streamlit as st
import yaml
import time

from cz_elections_live.data_adapters.ps2021_fixture import load_fixture_snapshot
from cz_elections_live.data_adapters.volby2025 import fetch_current_totals
from cz_elections_live.data_adapters.incremental_2021 import create_simulator
from cz_elections_live.model.monte_carlo import simulate_outcomes
from cz_elections_live.model.seats import allocate_seats_dhondt_regional
from cz_elections_live.settings import CACHE_TTL, DATA_MODE, MAJORITY, THRESHOLDS


def main():
    st.set_page_config(page_title="CZ 2025 Live Forecast", layout="wide")
    st.title("Czech Parliamentary Elections — Live Coalition Odds")

    with st.sidebar:
        sims = st.slider("Simulations", 1000, 20000, 5000, 1000)
        
        # Show performance recommendations
        if sims <= 3000:
            st.success("🚀 Fast updates (~1-2s)")
        elif sims <= 8000:
            st.info("⚡ Medium speed (~3-5s)")
        else:
            st.warning("🐌 Slower updates (~5-10s)")
        
        if DATA_MODE == "ps2021_incremental":
            duration = st.slider("Simulation Duration (minutes)", 1, 10, 2, 1)
            if st.button("Start New Simulation"):
                st.session_state.simulator = create_simulator(duration_minutes=duration)
                st.session_state.simulator.reset_simulation()
                st.session_state.simulator.start_simulation()
                st.cache_data.clear()
                st.rerun()
        else:
            profile = st.selectbox(
                "Fixture profile (2021)", ["urban_late_55", "rural_first_30", "balanced_70"]
            )
            
        if st.button("Refresh"):
            st.cache_data.clear()

    def get_partial():
        if DATA_MODE == "ps2025_live":
            return fetch_current_totals()
        elif DATA_MODE == "ps2021_incremental":
            if "simulator" not in st.session_state:
                st.session_state.simulator = create_simulator(duration_minutes=2)
                st.session_state.simulator.start_simulation()
            return st.session_state.simulator.get_current_partial()
        else:
            return load_fixture_snapshot(progress=profile)

    df_partial = get_partial()
    
    # Show progress info for incremental mode
    if DATA_MODE == "ps2021_incremental" and "simulator" in st.session_state:
        progress_info = st.session_state.simulator.get_progress_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Progress", f"{progress_info['progress']:.1%}")
        with col2:
            st.metric("Elapsed", f"{progress_info['elapsed_minutes']:.1f} min")
        with col3:
            st.metric("Remaining", f"{progress_info['remaining_minutes']:.1f} min")
            
        # Progress bar
        st.progress(progress_info['progress'])
        
        if progress_info['is_complete']:
            st.success("🎉 Simulation Complete! Results have converged to 2021 final results.")
    
    st.subheader("Partial totals")
    if df_partial.empty:
        st.info(
            "No data yet. In fixture mode, ensure data/raw CSVs exist. "
            "In live mode, wait for first XML."
        )
    st.dataframe(
        df_partial.sort_values("votes_reported", ascending=False), use_container_width=True
    )

    with open("config/coalitions.yaml") as f:
        COALS = yaml.safe_load(f)["coalitions"]

    # Show simulation info for incremental mode
    if DATA_MODE == "ps2021_incremental" and "simulator" in st.session_state:
        progress_info = st.session_state.simulator.get_progress_info()
        st.info(f"🎲 Running {sims:,} Monte Carlo simulations with current partial data (Progress: {progress_info['progress']:.1%})")
    
    # Time the Monte Carlo simulation
    import time
    start_time = time.time()
    
    # Create a progress bar for Monte Carlo simulation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🎲 Initializing Monte Carlo simulation...")
    progress_bar.progress(0.1)
    
    with st.spinner("Running Monte Carlo…"):
        status_text.text(f"🎲 Running {sims:,} Monte Carlo simulations...")
        progress_bar.progress(0.5)
        
        sim_result = simulate_outcomes(
            partial=df_partial,
            n_sims=sims,
            seat_allocator=allocate_seats_dhondt_regional,
            thresholds=THRESHOLDS,
        )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Monte Carlo simulation complete!")
    
    calc_time = time.time() - start_time
    
    # Show calculation time and clear progress indicators after a brief moment
    if calc_time > 0.5:  # Only show if it took more than 0.5 seconds
        st.caption(f"⏱️ Monte Carlo calculation took {calc_time:.1f} seconds")
    
    # Clear the progress indicators after showing completion
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
    status_text.empty()

    rows = []
    for c in COALS:
        p, exp = sim_result.coalition_stats(c["parties"], majority=MAJORITY)
        rows.append({"Coalition": c["name"], "P(≥101)": f"{p:.1%}", "Exp seats": f"{exp:.1f}"})
    st.subheader("Coalition majority odds")
    st.dataframe(
        pd.DataFrame(rows).sort_values("P(≥101)", ascending=False), use_container_width=True
    )

    st.caption("Mode: " + DATA_MODE)
    
    # Auto-refresh for incremental mode
    if DATA_MODE == "ps2021_incremental" and "simulator" in st.session_state:
        progress_info = st.session_state.simulator.get_progress_info()
        if not progress_info['is_complete']:
            # Calculate refresh interval based on simulation count and calculation time
            base_interval = 3.0  # Base 3 seconds
            sim_factor = max(1.0, sims / 10000)  # Longer wait for more simulations
            calc_factor = max(1.0, calc_time * 0.5)  # Factor in calculation time
            
            refresh_interval = base_interval * sim_factor * calc_factor
            refresh_interval = min(refresh_interval, 10.0)  # Cap at 10 seconds
            
            st.caption(f"🔄 Auto-refreshing every {refresh_interval:.1f} seconds (based on {sims:,} sims, {calc_time:.1f}s calc time)...")
            
            # Store timing info for next iteration
            if "last_calc_time" not in st.session_state:
                st.session_state.last_calc_time = calc_time
            
            time.sleep(refresh_interval)
            st.rerun()


def main_cli():
    """Entry point for Poetry script 'cz-live'"""
    import os
    import subprocess
    import sys

    # Get the absolute path to this file
    app_path = os.path.abspath(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    main()
