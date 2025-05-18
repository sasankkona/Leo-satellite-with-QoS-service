import numpy as np
import matplotlib.pyplot as plt
from new import LEOSatelliteNetwork
import random
import pandas as pd
import seaborn as sns

def run_comparison_with_increasing_traffic(rows=4, cols=5, steps=100, traffic_rates=None, use_fixed_size=False):
    """Run a comparison between BP and DHBP with increasing traffic rates"""
    if traffic_rates is None:
        traffic_rates = np.arange(0.5, 5.1, 0.5)  # From 0.5 to 5.0 Mbps with 0.5 increments

    results = {
        'traffic_rate': [],
        'algorithm': [],
        'delivered_packets': [],
        'dropped_packets': [],
        'avg_delay': [],
        'throughput': [],
        'delivery_ratio': [],
        'avg_forwarding': []
    }

    target_node = random.randint(1, rows * cols)  # Random target node
    print(f"Using target node: {target_node}")

    for rate in traffic_rates:
        print(f"Running simulation with traffic rate: {rate} Mbps")

        # Create two identical networks
        bp_network = LEOSatelliteNetwork(rows=rows, cols=cols)
        dhbp_network = LEOSatelliteNetwork(rows=rows, cols=cols)

        # Set the target node for both networks
        bp_network.set_target_node(target_node)
        dhbp_network.set_target_node(target_node)

        # Initialize orbital parameters for realistic hop calculation
        bp_network.initialize_orbital_parameters()
        dhbp_network.initialize_orbital_parameters()

        # Set base traffic rate for all satellites
        for sat in bp_network.satellites:
            # Set shape and scale parameters to control traffic pattern
            sat.shape = random.uniform(1.5, 3.0)  # Pareto shape parameter
            sat.scale = rate  # Use the traffic rate as the scale parameter

        for sat in dhbp_network.satellites:
            # Use the same shape and scale for fair comparison
            sat.shape = bp_network.satellites[sat.id-1].shape
            sat.scale = bp_network.satellites[sat.id-1].scale

        # Run simulations
        bp_network.run_simulation(steps=steps, use_dhbp=False, use_fixed_size=use_fixed_size)
        dhbp_network.run_simulation(steps=steps, use_dhbp=True, use_fixed_size=use_fixed_size)

        # Calculate average forwarding for BP
        bp_total_forwarded = sum(sat.packets_forwarded for sat in bp_network.satellites)
        bp_avg_forwarding = bp_total_forwarded / max(1, bp_network.simulation_stats['delivered_packets'])

        # Calculate average forwarding for DHBP
        dhbp_total_forwarded = sum(sat.packets_forwarded for sat in dhbp_network.satellites)
        dhbp_avg_forwarding = dhbp_total_forwarded / max(1, dhbp_network.simulation_stats['delivered_packets'])

        # Collect BP results
        results['traffic_rate'].append(rate)
        results['algorithm'].append('BP')
        results['delivered_packets'].append(bp_network.simulation_stats['delivered_packets'])
        results['dropped_packets'].append(bp_network.simulation_stats['dropped_packets'])
        results['avg_delay'].append(bp_network.simulation_stats['avg_delay'])
        results['throughput'].append(bp_network.simulation_stats['delivered_packets'] / steps * 10)  # Packets per second (10 timeslots/second)
        total_bp = bp_network.simulation_stats['delivered_packets'] + bp_network.simulation_stats['dropped_packets']
        results['delivery_ratio'].append(bp_network.simulation_stats['delivered_packets'] / max(1, total_bp))
        results['avg_forwarding'].append(bp_avg_forwarding)

        # Collect DHBP results
        results['traffic_rate'].append(rate)
        results['algorithm'].append('DHBP')
        results['delivered_packets'].append(dhbp_network.simulation_stats['delivered_packets'])
        results['dropped_packets'].append(dhbp_network.simulation_stats['dropped_packets'])
        results['avg_delay'].append(dhbp_network.simulation_stats['avg_delay'])
        results['throughput'].append(dhbp_network.simulation_stats['delivered_packets'] / steps * 10)  # Packets per second
        total_dhbp = dhbp_network.simulation_stats['delivered_packets'] + dhbp_network.simulation_stats['dropped_packets']
        results['delivery_ratio'].append(dhbp_network.simulation_stats['delivered_packets'] / max(1, total_dhbp))
        results['avg_forwarding'].append(dhbp_avg_forwarding)

    return results


def plot_comparison_results(results, packet_type="Variable-size", save_fig=True):
    """Plot comparison graphs similar to the research paper"""
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Create a figure with 4 subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Set custom colors and markers
    bp_style = {'color': 'red', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8}
    dhbp_style = {'color': 'blue', 'marker': 's', 'linestyle': '-', 'linewidth': 2, 'markersize': 8}

    # Plot 1: Average Delay
    bp_data = df[df['algorithm'] == 'BP']
    dhbp_data = df[df['algorithm'] == 'DHBP']

    axes[0, 0].plot(bp_data['traffic_rate'], bp_data['avg_delay'], **bp_style, label='BP')
    axes[0, 0].plot(dhbp_data['traffic_rate'], dhbp_data['avg_delay'], **dhbp_style, label='DHBP')
    axes[0, 0].set_xlabel('CBR (Mbps)')
    axes[0, 0].set_ylabel('Average delay (s)')
    axes[0, 0].set_title('(a) Average delay')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Average number of forwarding per packets
    axes[0, 1].plot(bp_data['traffic_rate'], bp_data['avg_forwarding'], **bp_style, label='BP')
    axes[0, 1].plot(dhbp_data['traffic_rate'], dhbp_data['avg_forwarding'], **dhbp_style, label='DHBP')
    axes[0, 1].set_xlabel('CBR (Mbps)')
    axes[0, 1].set_ylabel('Average num of forwarding per packets')
    axes[0, 1].set_title('(b) Average number of forwarding per packets')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Throughput
    axes[1, 0].plot(bp_data['traffic_rate'], bp_data['throughput'], **bp_style, label='BP')
    axes[1, 0].plot(dhbp_data['traffic_rate'], dhbp_data['throughput'], **dhbp_style, label='DHBP')
    axes[1, 0].set_xlabel('CBR (Mbps)')
    axes[1, 0].set_ylabel('Throughput (Mbps)')
    axes[1, 0].set_title('(c) Throughput')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Data delivery ratio
    axes[1, 1].plot(bp_data['traffic_rate'], bp_data['delivery_ratio'], **bp_style, label='BP')
    axes[1, 1].plot(dhbp_data['traffic_rate'], dhbp_data['delivery_ratio'], **dhbp_style, label='DHBP')
    axes[1, 1].set_xlabel('CBR (Mbps)')
    axes[1, 1].set_ylabel('Data delivery ratio')
    axes[1, 1].set_title('(d) Data delivery ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    # Set overall title
    fig.suptitle(f'The performance under CBR traffic with {packet_type} packets', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_fig:
        plt.savefig(f"bp_dhbp_comparison_{packet_type.lower().replace('-', '_')}.png", dpi=300, bbox_inches='tight')

    return fig


def run_pareto_comparison(rows=4, cols=5, steps=100, traffic_rates=None, use_fixed_size=False):
    """Run comparison with Pareto traffic distribution"""
    # Similar to the constant bit rate function but using Pareto distribution
    if traffic_rates is None:
        traffic_rates = np.arange(0.5, 5.1, 0.5)  # From 0.5 to 5.0 Mbps with 0.5 increments

    results = {
        'traffic_rate': [],
        'algorithm': [],
        'delivered_packets': [],
        'dropped_packets': [],
        'avg_delay': [],
        'throughput': [],
        'delivery_ratio': [],
        'avg_forwarding': []
    }

    target_node = random.randint(1, rows * cols)
    print(f"Using target node: {target_node}")

    for rate in traffic_rates:
        print(f"Running Pareto simulation with base traffic rate: {rate} Mbps")

        # Create networks with identical initial settings
        bp_network = LEOSatelliteNetwork(rows=rows, cols=cols)
        dhbp_network = LEOSatelliteNetwork(rows=rows, cols=cols)

        bp_network.set_target_node(target_node)
        dhbp_network.set_target_node(target_node)

        bp_network.initialize_orbital_parameters()
        dhbp_network.initialize_orbital_parameters()

        # For Pareto distribution, ensure the shape parameters are set
        for sat in bp_network.satellites:
            # Shape controls the heaviness of the tail - lower values mean more burstiness
            sat.shape = random.uniform(1.1, 1.5)  # More bursty traffic with heavier tails
            sat.scale = rate  # Base rate

        # Ensure DHBP network has the same parameters
        for sat in dhbp_network.satellites:
            sat.shape = bp_network.satellites[sat.id-1].shape
            sat.scale = bp_network.satellites[sat.id-1].scale

        # Run simulations
        bp_network.run_simulation(steps=steps, use_dhbp=False, use_fixed_size=use_fixed_size)
        dhbp_network.run_simulation(steps=steps, use_dhbp=True, use_fixed_size=use_fixed_size)

        # Calculate average forwarding for BP
        bp_total_forwarded = sum(sat.packets_forwarded for sat in bp_network.satellites)
        bp_avg_forwarding = bp_total_forwarded / max(1, bp_network.simulation_stats['delivered_packets'])

        # Calculate average forwarding for DHBP
        dhbp_total_forwarded = sum(sat.packets_forwarded for sat in dhbp_network.satellites)
        dhbp_avg_forwarding = dhbp_total_forwarded / max(1, dhbp_network.simulation_stats['delivered_packets'])

        # Collect BP results
        results['traffic_rate'].append(rate)
        results['algorithm'].append('BP')
        results['delivered_packets'].append(bp_network.simulation_stats['delivered_packets'])
        results['dropped_packets'].append(bp_network.simulation_stats['dropped_packets'])
        results['avg_delay'].append(bp_network.simulation_stats['avg_delay'])
        results['throughput'].append(bp_network.simulation_stats['delivered_packets'] / steps * 10)
        total_bp = bp_network.simulation_stats['delivered_packets'] + bp_network.simulation_stats['dropped_packets']
        results['delivery_ratio'].append(bp_network.simulation_stats['delivered_packets'] / max(1, total_bp))
        results['avg_forwarding'].append(bp_avg_forwarding)

        # Collect DHBP results
        results['traffic_rate'].append(rate)
        results['algorithm'].append('DHBP')
        results['delivered_packets'].append(dhbp_network.simulation_stats['delivered_packets'])
        results['dropped_packets'].append(dhbp_network.simulation_stats['dropped_packets'])
        results['avg_delay'].append(dhbp_network.simulation_stats['avg_delay'])
        results['throughput'].append(dhbp_network.simulation_stats['delivered_packets'] / steps * 10)
        total_dhbp = dhbp_network.simulation_stats['delivered_packets'] + dhbp_network.simulation_stats['dropped_packets']
        results['delivery_ratio'].append(dhbp_network.simulation_stats['delivered_packets'] / max(1, total_dhbp))
        results['avg_forwarding'].append(dhbp_avg_forwarding)

    return results


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Parameters for the comparison
    rows = 4
    cols = 5
    steps = 100  # Number of simulation steps
    traffic_rates = np.arange(1.0, 5.1, 0.5)  # From 1.0 to 5.0 Mbps with 0.5 increments

    # Run with variable packet sizes
    print("Running comparison with variable-size packets and constant bit rate...")
    cbr_results = run_comparison_with_increasing_traffic(
        rows=rows, cols=cols, steps=steps, traffic_rates=traffic_rates, use_fixed_size=False
    )

    # Plot variable packet size results
    plot_comparison_results(cbr_results, "Variable-size")

    # Run with fixed packet sizes
    print("Running comparison with fixed-size packets and constant bit rate...")
    cbr_fixed_results = run_comparison_with_increasing_traffic(
        rows=rows, cols=cols, steps=steps, traffic_rates=traffic_rates, use_fixed_size=True
    )

    # Plot fixed packet size results
    plot_comparison_results(cbr_fixed_results, "Fixed-size")

    # Run with Pareto traffic distribution and variable packet sizes
    print("Running comparison with variable-size packets and Pareto traffic...")
    pareto_results = run_pareto_comparison(
        rows=rows, cols=cols, steps=steps, traffic_rates=traffic_rates, use_fixed_size=False
    )

    # Plot Pareto with variable packet size results
    plot_comparison_results(pareto_results, "Variable-size Pareto", save_fig=True)

    # Run with Pareto traffic distribution and fixed packet sizes
    print("Running comparison with fixed-size packets and Pareto traffic...")
    pareto_fixed_results = run_pareto_comparison(
        rows=rows, cols=cols, steps=steps, traffic_rates=traffic_rates, use_fixed_size=True
    )

    # Plot Pareto with fixed packet size results
    plot_comparison_results(pareto_fixed_results, "Fixed-size Pareto", save_fig=True)

    print("All simulations and plots complete.")
    plt.show()


if __name__ == "__main__":
    main()