import pandas as pd
import streamlit as st
import yaml
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
                # Clear historical data when starting new simulation
                st.session_state.historical_data = []
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
    elif df_partial["votes_reported"].sum() == 0:
        st.warning("⏳ Waiting for votes to be counted... Polls may not have opened yet.")
    st.dataframe(
        df_partial.sort_values("votes_reported", ascending=False), use_container_width=True
    )

    with open("config/coalitions.yaml") as f:
        COALS = yaml.safe_load(f)["coalitions"]

    with open("config/parties.yaml") as f:
        PARTIES_CONFIG = yaml.safe_load(f)

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

    # Calculate individual party seat projections and threshold probabilities
    party_seats = {}
    threshold_probs = {}
    for i, party_code in enumerate(sim_result.parties):
        party_seats[party_code] = sim_result.seats[:, i].mean()
        # Calculate probability of crossing 5% threshold (getting any seats)
        threshold_probs[party_code] = float((sim_result.seats[:, i] > 0).mean())
    
    # Add projected seats and threshold probabilities to partial data
    df_partial_with_seats = df_partial.copy()
    df_partial_with_seats["projected_seats"] = df_partial_with_seats["party_code"].map(party_seats).fillna(0).round(1)
    df_partial_with_seats["threshold_prob"] = df_partial_with_seats["party_code"].map(threshold_probs).fillna(0) * 100

    # Display parties table with projected seats
    st.subheader("Partial totals & Projected seats")
    display_df = df_partial_with_seats[["party_code", "party_name", "votes_reported", "pct_reported", "projected_seats", "threshold_prob"]].copy()
    display_df.columns = ["Party", "Name", "Votes", "Pct", "Proj. Seats", "P(>5%)"]
    display_df["P(>5%)"] = display_df["P(>5%)"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(
        display_df.sort_values("Votes", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Show parties near threshold
    near_threshold = df_partial_with_seats[
        (df_partial_with_seats["pct_reported"] >= 3) &
        (df_partial_with_seats["pct_reported"] <= 7)
    ].sort_values("pct_reported", ascending=False)

    if not near_threshold.empty:
        st.info("⚠️ **Parties near 5% threshold:** " +
                ", ".join([f"{row['party_name']} ({row['pct_reported']:.1f}%, {row['threshold_prob']:.0f}% chance)"
                          for _, row in near_threshold.iterrows()]))

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Projected Seat Distribution**")
        # Get colors for parties
        colors_map = {party: PARTIES_CONFIG.get(party, {}).get("color", "#808080")
                     for party in df_partial_with_seats["party_code"]}
        colors_list = [colors_map.get(party, "#808080") for party in display_df["Party"]]

        fig_seats = go.Figure(data=[go.Bar(
            x=display_df["Proj. Seats"],
            y=display_df["Name"],
            orientation='h',
            marker=dict(color=colors_list),
            text=display_df["Proj. Seats"],
            textposition='outside'
        )])
        fig_seats.update_layout(
            height=400,
            xaxis_title="Projected Seats",
            yaxis_title="",
            showlegend=False,
            xaxis=dict(range=[0, 100])
        )
        fig_seats.add_vline(x=101, line_dash="dash", line_color="red",
                           annotation_text="Majority (101)", annotation_position="top")
        st.plotly_chart(fig_seats, use_container_width=True)

    with col2:
        st.markdown("**Vote Share Distribution**")
        fig_votes = go.Figure(data=[go.Bar(
            x=display_df["Pct"],
            y=display_df["Name"],
            orientation='h',
            marker=dict(color=colors_list),
            text=[f"{p:.1f}%" for p in display_df["Pct"]],
            textposition='outside'
        )])
        fig_votes.update_layout(
            height=400,
            xaxis_title="Vote Share (%)",
            yaxis_title="",
            showlegend=False,
            xaxis=dict(range=[0, 40])
        )
        fig_votes.add_vline(x=5, line_dash="dash", line_color="orange",
                           annotation_text="5% threshold", annotation_position="top")
        st.plotly_chart(fig_votes, use_container_width=True)
    
    # Calculate coalition odds
    rows = []
    for c in COALS:
        p, exp = sim_result.coalition_stats(c["parties"], majority=MAJORITY)
        rows.append({"Coalition": c["name"], "P(≥101)": f"{p:.1%}", "Exp seats": f"{exp:.1f}"})
    st.subheader("Coalition majority odds")
    coalition_df = pd.DataFrame(rows).sort_values("P(≥101)", ascending=False)
    st.dataframe(coalition_df, use_container_width=True, hide_index=True)
    
    # Store historical data for evolution charts (only in incremental mode)
    if DATA_MODE == "ps2021_incremental" and "simulator" in st.session_state:
        progress_info = st.session_state.simulator.get_progress_info()
        
        # Initialize historical data storage
        if "historical_data" not in st.session_state:
            st.session_state.historical_data = []
        
        # Add current data point
        current_time = progress_info['elapsed_minutes']
        data_point = {
            "time": current_time,
            "progress": progress_info['progress']
        }
        
        # Add coalition data
        for i, row in coalition_df.iterrows():
            coalition_name = row["Coalition"]
            # Extract probability and expected seats from formatted strings
            prob_str = row["P(≥101)"].replace("%", "")
            exp_seats_str = row["Exp seats"]
            
            data_point[f"{coalition_name}_prob"] = float(prob_str) / 100.0
            data_point[f"{coalition_name}_exp_seats"] = float(exp_seats_str)
        
        st.session_state.historical_data.append(data_point)
        
        # Create evolution charts for top 3 coalitions
        if len(st.session_state.historical_data) > 1:
            st.subheader("📈 Coalition Evolution")
            
            # Convert to DataFrame for plotting
            hist_df = pd.DataFrame(st.session_state.historical_data)
            
            # Get top 3 coalitions by current probability
            top_coalitions = coalition_df.head(3)["Coalition"].tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Majority Probability Evolution**")
                fig_prob = go.Figure()
                
                for coalition in top_coalitions:
                    if f"{coalition}_prob" in hist_df.columns:
                        fig_prob.add_trace(go.Scatter(
                            x=hist_df["time"],
                            y=hist_df[f"{coalition}_prob"] * 100,
                            mode='lines+markers',
                            name=coalition,
                            line=dict(width=3),
                            marker=dict(size=6)
                        ))
                
                fig_prob.update_layout(
                    title="Probability of Reaching Majority (≥101 seats)",
                    xaxis_title="Time (minutes)",
                    yaxis_title="Probability (%)",
                    yaxis=dict(range=[0, 100]),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
            
            with col2:
                st.markdown("**Expected Seats Evolution**")
                fig_seats = go.Figure()
                
                for coalition in top_coalitions:
                    if f"{coalition}_exp_seats" in hist_df.columns:
                        fig_seats.add_trace(go.Scatter(
                            x=hist_df["time"],
                            y=hist_df[f"{coalition}_exp_seats"],
                            mode='lines+markers',
                            name=coalition,
                            line=dict(width=3),
                            marker=dict(size=6)
                        ))
                
                # Add majority line
                fig_seats.add_hline(y=101, line_dash="dash", line_color="red", 
                                  annotation_text="Majority (101 seats)", annotation_position="top right")
                
                fig_seats.update_layout(
                    title="Expected Total Seats",
                    xaxis_title="Time (minutes)",
                    yaxis_title="Expected Seats",
                    yaxis=dict(range=[0, 200]),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_seats, use_container_width=True)
            
            # Show data points count
            st.caption(f"📊 Tracking {len(st.session_state.historical_data)} data points")
            
            # Add clear history button
            if st.button("🗑️ Clear History", help="Clear the evolution charts history"):
                st.session_state.historical_data = []
                st.rerun()

    st.caption("Mode: " + DATA_MODE)

    # Auto-refresh for incremental and live modes
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

    elif DATA_MODE == "ps2025_live":
        # Auto-refresh for live election data (60 seconds default, aligned with feed updates)
        refresh_interval = max(60.0, calc_time * 2)  # At least 60s, more if calculation is slow
        st.caption(f"🔄 Auto-refreshing every {refresh_interval:.0f} seconds for live results...")
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
