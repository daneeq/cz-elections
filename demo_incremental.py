#!/usr/bin/env python3
"""
Demo script for the incremental data simulator.

This script demonstrates how the incremental simulator works by showing
how partial results evolve over time during a simulated election night.
"""

import time
import pandas as pd
from cz_elections_live.data_adapters.incremental_2021 import create_simulator


def demo_incremental_simulation(duration_minutes=1):
    """
    Run a quick demo of the incremental simulation.
    
    Args:
        duration_minutes: How long the demo should run (default 2 minutes)
    """
    print(f"🚀 Starting incremental election simulation ({duration_minutes} minute)")
    print("=" * 60)
    
    # Create simulator
    simulator = create_simulator(duration_minutes=duration_minutes)
    simulator.start_simulation()
    
    # Track progress
    last_progress = 0
    update_interval = 10  # Update every 10 seconds
    
    try:
        while True:
            # Get current partial results
            partial_data = simulator.get_current_partial()
            progress_info = simulator.get_progress_info()
            
            # Only print updates when progress has changed significantly
            if progress_info['progress'] - last_progress >= 0.1 or progress_info['is_complete']:
                print(f"\n⏰ Time: {progress_info['elapsed_minutes']:.1f} min | "
                      f"Progress: {progress_info['progress']:.1%}")
                print("-" * 40)
                
                # Show top parties
                top_parties = partial_data.nlargest(4, 'votes_reported')
                for _, party in top_parties.iterrows():
                    print(f"{party['party_code']:>6}: {party['votes_reported']:>8,.0f} votes "
                          f"({party['pct_reported']:>5.1f}%)")
                
                last_progress = progress_info['progress']
                
                if progress_info['is_complete']:
                    print("\n🎉 Simulation Complete!")
                    break
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
    
    # Show final results
    final_data = simulator.get_current_partial()
    print(f"\n📊 Final Results (after {progress_info['elapsed_minutes']:.1f} minutes):")
    print("=" * 50)
    
    for _, party in final_data.iterrows():
        print(f"{party['party_code']:>6}: {party['votes_reported']:>8,.0f} votes "
              f"({party['pct_reported']:>5.1f}%)")


def demo_coalition_odds():
    """Demo showing how coalition odds change during simulation."""
    print("\n🔮 Coalition Odds Demo")
    print("=" * 30)
    
    simulator = create_simulator(duration_minutes=1)
    simulator.start_simulation()
    
    # Import here to avoid circular imports
    from cz_elections_live.model.monte_carlo import simulate_outcomes
    from cz_elections_live.model.seats import allocate_seats_dhondt_regional
    from cz_elections_live.settings import THRESHOLDS
    
    last_progress = 0
    
    try:
        while True:
            partial_data = simulator.get_current_partial()
            progress_info = simulator.get_progress_info()
            
            if progress_info['progress'] - last_progress >= 0.2 or progress_info['is_complete']:
                print(f"\n📈 Progress: {progress_info['progress']:.1%}")
                
                # Run Monte Carlo simulation
                sim_result = simulate_outcomes(
                    partial=partial_data,
                    n_sims=1000,  # Quick simulation for demo
                    seat_allocator=allocate_seats_dhondt_regional,
                    thresholds=THRESHOLDS,
                )
                
                # Show coalition odds
                coalitions = [
                    (["SPOLU", "PIR", "STAN"], "Gov bloc"),
                    (["ANO", "SPD"], "Opposition core"),
                    (["SPOLU", "ANO"], "Grand coalition")
                ]
                
                for parties, name in coalitions:
                    p_majority, exp_seats = sim_result.coalition_stats(parties, majority=101)
                    print(f"{name:>15}: {p_majority:>6.1%} chance of majority "
                          f"({exp_seats:>5.1f} expected seats)")
                
                last_progress = progress_info['progress']
                
                if progress_info['is_complete']:
                    break
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Coalition demo stopped")


if __name__ == "__main__":
    print("🇨🇿 Czech Elections Live - Incremental Simulator Demo")
    print("=" * 55)
    
    try:
        # Run basic simulation demo
        demo_incremental_simulation(duration_minutes=1)
        
        # Ask if user wants to see coalition odds demo
        print("\n" + "=" * 55)
        response = input("Would you like to see coalition odds demo? (y/n): ").lower()
        
        if response == 'y':
            demo_coalition_odds()
            
    except KeyboardInterrupt:
        print("\n👋 Demo ended")
    
    print("\n✨ Demo complete! To run the full dashboard:")
    print("   poetry run streamlit run src/cz_elections_live/streamlit_app.py")
