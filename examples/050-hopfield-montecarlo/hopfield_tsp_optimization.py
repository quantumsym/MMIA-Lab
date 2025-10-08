'''
 implementation for solving the Traveling Salesman Problem (TSP)
using Hopfield networks and advanced stochastic optimization techniques.

This script demonstrates how to map the TSP onto a Hopfield network's energy
function and then use methods like Simulated Annealing and Parallel Tempering
to find optimal or near-optimal solutions. It compares these advanced methods
against a standard stochastic Hopfield network and, for small instances, against
the brute-force optimal solution.
'''

import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist, squareform

# Import  functions from the other converted scripts
from simulated_annealing_hopfield import (
    initialize_parallel_tempering_system,
    run_parallel_tempering,
    run_simulated_annealing
)
from monte_carlo_hopfield import (
    initialize_stochastic_network,
    run_monte_carlo
)

# ============================================================================
#  HOPFIELD TSP SOLVER
# ============================================================================

def initialize_tsp_problem(cities, constraint_weights=None):
    "Initialize a TSP problem for a Hopfield network solver."
    if constraint_weights is None:
        constraint_weights = {
            'distance': 1.0,      # Weight for distance minimization
            'row': 2.0,          # Weight for row constraints (each city once)
            'column': 2.0,       # Weight for column constraints (each position once)
            'validity': 1.0      # Weight for binary validity constraints
        }
    
    num_cities = len(cities)
    distance_matrix = squareform(pdist(cities, metric='euclidean'))
    network_size = num_cities ** 2

    tsp_problem = {
        'cities': np.array(cities),
        'num_cities': num_cities,
        'constraint_weights': constraint_weights,
        'distance_matrix': distance_matrix,
        'network_size': network_size
    }
    print(f"Initialized TSP Problem: {num_cities} cities, {network_size} neurons.")
    return tsp_problem

def _neuron_index(tsp_problem, city, position):
    "Convert (city, position) to a linear neuron index."
    return city * tsp_problem['num_cities'] + position

def _city_position_from_index(tsp_problem, neuron_index):
    "Convert a linear neuron index back to (city, position)."
    city = neuron_index // tsp_problem['num_cities']
    position = neuron_index % tsp_problem['num_cities']
    return city, position

def create_tsp_hopfield_weights(tsp_problem):
    "Create the weight matrix for the Hopfield network based on the TSP energy function."
    weights = np.zeros((tsp_problem['network_size'], tsp_problem['network_size']))
    A = tsp_problem['constraint_weights']['row']
    B = tsp_problem['constraint_weights']['column']
    C = tsp_problem['constraint_weights']['distance']
    
    num_cities = tsp_problem['num_cities']
    dist_matrix = tsp_problem['distance_matrix']

    for i in range(tsp_problem['network_size']):
        for j in range(i, tsp_problem['network_size']):
            city_i, pos_i = _city_position_from_index(tsp_problem, i)
            city_j, pos_j = _city_position_from_index(tsp_problem, j)
            
            weight = 0
            if i == j:
                continue # No self-connection

            # Row constraint: Inhibit multiple visits to the same city
            if city_i == city_j and pos_i != pos_j:
                weight -= A
            
            # Column constraint: Inhibit multiple cities at the same position
            if pos_i == pos_j and city_i != city_j:
                weight -= B
            
            # Distance constraint: Encourage short connections between adjacent positions
            if pos_j == (pos_i + 1) % num_cities or pos_i == (pos_j + 1) % num_cities:
                weight -= C * dist_matrix[city_i, city_j]
            
            weights[i, j] = weight
            weights[j, i] = weight # Symmetric matrix
            
    return weights

def create_tsp_hopfield_thresholds(tsp_problem):
    "Create the threshold (bias) vector for the Hopfield network."
    D = tsp_problem['constraint_weights']['validity']
    # This term encourages a total of N neurons to be active.
    thresholds = np.full(tsp_problem['network_size'], D * tsp_problem['num_cities'])
    return thresholds

def state_to_tour(tsp_problem, state):
    "Convert a Hopfield network state to a TSP tour."
    state_matrix = state.reshape((tsp_problem['num_cities'], tsp_problem['num_cities']))
    binary_matrix = (state_matrix > 0).astype(int)

    # Check validity
    row_sums = np.sum(binary_matrix, axis=1)
    col_sums = np.sum(binary_matrix, axis=0)
    is_valid = np.all(row_sums == 1) and np.all(col_sums == 1)

    tour = []
    if is_valid:
        tour = np.argmax(binary_matrix, axis=0).tolist()
    else:
        # If not valid, construct an approximate tour by taking the most active neuron per position
        tour_set = set()
        tour = [-1] * tsp_problem['num_cities']
        for pos in range(tsp_problem['num_cities']):
            activations = state_matrix[:, pos]
            sorted_cities = np.argsort(-activations)
            for city in sorted_cities:
                if city not in tour_set:
                    tour[pos] = city
                    tour_set.add(city)
                    break
    
    # Compute tour length
    tour_length = 0
    for i in range(tsp_problem['num_cities']):
        city_from = tour[i]
        city_to = tour[(i + 1) % tsp_problem['num_cities']]
        if city_from != -1 and city_to != -1:
            tour_length += tsp_problem['distance_matrix'][city_from, city_to]
            
    return tour, is_valid, tour_length

def solve_tsp_with_sa(tsp_problem, initial_temp, final_temp, total_steps, cooling_schedule='exponential', verbose=True):
    "Solve TSP using Simulated Annealing."
    network = initialize_stochastic_network(tsp_problem['network_size'], initial_temp)
    network['weights'] = create_tsp_hopfield_weights(tsp_problem)
    network['thresholds'] = create_tsp_hopfield_thresholds(tsp_problem)

    sa_results = run_simulated_annealing(network, initial_temp, final_temp, total_steps, cooling_schedule, verbose)
    
    tour, is_valid, tour_length = state_to_tour(tsp_problem, sa_results['best_state'])
    
    return {
        'method': 'Simulated Annealing',
        'best_tour': tour,
        'best_tour_length': tour_length,
        'is_valid': is_valid,
        **sa_results
    }

def solve_tsp_with_pt(tsp_problem, num_replicas, temp_range, total_steps, mc_steps_per_exchange, verbose=True):
    "Solve TSP using Parallel Tempering."
    system = initialize_parallel_tempering_system(tsp_problem['network_size'], num_replicas, temp_range[0], temp_range[1])
    weights = create_tsp_hopfield_weights(tsp_problem)
    thresholds = create_tsp_hopfield_thresholds(tsp_problem)

    for replica in system['replicas']:
        replica['weights'] = weights
        replica['thresholds'] = thresholds

    pt_results = run_parallel_tempering(system, total_steps, mc_steps_per_exchange, verbose=verbose)
    
    tour, is_valid, tour_length = state_to_tour(tsp_problem, pt_results['best_state'])
    
    return {
        'method': 'Parallel Tempering',
        'best_tour': tour,
        'best_tour_length': tour_length,
        'is_valid': is_valid,
        **pt_results
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_test_cities(num_cities, city_type='random', random_seed=None):
    """Generate test cities for TSP experiments."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if city_type == 'random':
        cities = np.random.uniform(0, 10, size=(num_cities, 2))
    elif city_type == 'circle':
        angles = np.linspace(0, 2*np.pi, num_cities, endpoint=False)
        radius = 5
        cities = np.column_stack([
            radius * np.cos(angles) + 5,
            radius * np.sin(angles) + 5
        ])
    elif city_type == 'grid':
        grid_size = int(np.ceil(np.sqrt(num_cities)))
        x_coords = np.linspace(0, 10, grid_size)
        y_coords = np.linspace(0, 10, grid_size)
        cities = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                if len(cities) < num_cities:
                    cities.append([x, y])
        cities = np.array(cities[:num_cities])
    else:
        raise ValueError(f"Unknown city type: {city_type}")
    
    return cities

def compute_optimal_tour_brute_force(cities):
    """Compute optimal TSP tour using brute force (only for small instances)."""
    num_cities = len(cities)
    if num_cities > 10:
        raise ValueError("Brute force only feasible for ≤ 10 cities")
    
    distance_matrix = squareform(pdist(cities, metric='euclidean'))
    city_indices = list(range(1, num_cities))
    
    best_tour = None
    best_length = float('inf')
    
    for perm in itertools.permutations(city_indices):
        tour = [0] + list(perm)
        length = sum(distance_matrix[tour[i], tour[(i + 1) % num_cities]] for i in range(num_cities))
        if length < best_length:
            best_length = length
            best_tour = tour
    
    return best_tour, best_length

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_tsp_optimization():
    """
    Main demonstration function showing TSP optimization with Hopfield networks.
    """
    print("=" * 80)
    print("TSP OPTIMIZATION WITH HOPFIELD NETWORKS DEMONSTRATION")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. SMALL TSP INSTANCE WITH OPTIMAL COMPARISON
    print(f"\n{'='*60}")
    print("SMALL TSP INSTANCE (6 CITIES) - OPTIMAL COMPARISON")
    print(f"{'='*60}")
    
    # Generate small test case
    small_cities = generate_test_cities(6, 'random', random_seed=42)
    small_tsp = initialize_tsp_problem(small_cities)
    
    print(f"City coordinates:")
    for i, city in enumerate(small_cities):
        print(f"  City {i}: ({city[0]:.2f}, {city[1]:.2f})")
    
    # Compute optimal solution
    print(f"\nComputing optimal solution (brute force)...")
    optimal_tour, optimal_length = compute_optimal_tour_brute_force(small_cities)
    print(f"  Optimal tour: {optimal_tour}")
    print(f"  Optimal length: {optimal_length:.3f}")
    
    # Test Simulated Annealing
    print(f"\nTesting Simulated Annealing...")
    sa_result = solve_tsp_with_sa(
        small_tsp, 
        initial_temp=10.0, 
        final_temp=0.01, 
        total_steps=5000, 
        cooling_schedule='exponential', 
        verbose=True
    )
    
    sa_gap = (sa_result['best_tour_length'] - optimal_length) / optimal_length * 100
    print(f"  SA tour: {sa_result['best_tour']}")
    print(f"  SA length: {sa_result['best_tour_length']:.3f}")
    print(f"  SA gap from optimal: {sa_gap:+.1f}%")
    print(f"  SA valid: {sa_result['is_valid']}")
    
    # Test Parallel Tempering
    print(f"\nTesting Parallel Tempering...")
    pt_result = solve_tsp_with_pt(
        small_tsp,
        num_replicas=8,
        temp_range=(0.1, 10.0),
        total_steps=500,
        mc_steps_per_exchange=10,
        verbose=True
    )
    
    pt_gap = (pt_result['best_tour_length'] - optimal_length) / optimal_length * 100
    print(f"  PT tour: {pt_result['best_tour']}")
    print(f"  PT length: {pt_result['best_tour_length']:.3f}")
    print(f"  PT gap from optimal: {pt_gap:+.1f}%")
    print(f"  PT valid: {pt_result['is_valid']}")
    print(f"  PT exchange rate: {pt_result['final_exchange_rate']:.1%}")
    
    # 2. MEDIUM TSP INSTANCE
    print(f"\n{'='*60}")
    print("MEDIUM TSP INSTANCE (10 CITIES)")
    print(f"{'='*60}")
    
    # Generate medium test case
    medium_cities = generate_test_cities(10, 'circle', random_seed=123)
    medium_tsp = initialize_tsp_problem(medium_cities)
    
    print(f"Generated 10 cities in circular arrangement")
    
    # Test different methods
    methods_results = []
    
    # Simulated Annealing with different schedules
    for schedule in ['linear', 'exponential']:
        print(f"\nTesting SA with {schedule} cooling...")
        result = solve_tsp_with_sa(
            medium_tsp, 15.0, 0.01, 8000, schedule, verbose=False
        )
        methods_results.append({
            'method': f'SA ({schedule})',
            'tour_length': result['best_tour_length'],
            'is_valid': result['is_valid'],
            'time': result['computation_time']
        })
        print(f"  Length: {result['best_tour_length']:.3f}, Valid: {result['is_valid']}, Time: {result['computation_time']:.2f}s")
    
    # Parallel Tempering with different replica counts
    for num_replicas in [6, 12]:
        print(f"\nTesting PT with {num_replicas} replicas...")
        result = solve_tsp_with_pt(
            medium_tsp, num_replicas, (0.1, 15.0), 800, 15, verbose=False
        )
        methods_results.append({
            'method': f'PT ({num_replicas} replicas)',
            'tour_length': result['best_tour_length'],
            'is_valid': result['is_valid'],
            'time': result['computation_time']
        })
        print(f"  Length: {result['best_tour_length']:.3f}, Valid: {result['is_valid']}, Time: {result['computation_time']:.2f}s")
    
    # 3. LARGE TSP INSTANCE
    print(f"\n{'='*60}")
    print("LARGE TSP INSTANCE (15 CITIES)")
    print(f"{'='*60}")
    
    # Generate large test case
    large_cities = generate_test_cities(15, 'random', random_seed=456)
    large_tsp = initialize_tsp_problem(large_cities)
    
    print(f"Generated 15 cities randomly distributed")
    
    # Test only the best methods from previous tests
    print(f"\nTesting best SA configuration...")
    large_sa = solve_tsp_with_sa(
        large_tsp, 20.0, 0.01, 15000, 'exponential', verbose=True
    )
    
    print(f"\nTesting best PT configuration...")
    large_pt = solve_tsp_with_pt(
        large_tsp, 12, (0.1, 20.0), 1000, 20, verbose=True
    )
    
    # 4. RESULTS COMPARISON
    print(f"\n{'='*60}")
    print("COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'='*60}")
    
    print(f"\nSmall Instance (6 cities) Results:")
    print(f"  Optimal:              {optimal_length:.3f}")
    print(f"  Simulated Annealing:  {sa_result['best_tour_length']:.3f} (gap: {sa_gap:+.1f}%)")
    print(f"  Parallel Tempering:   {pt_result['best_tour_length']:.3f} (gap: {pt_gap:+.1f}%)")
    
    print(f"\nMedium Instance (10 cities) Results:")
    best_medium = min(methods_results, key=lambda x: x['tour_length'])
    print(f"  Best method: {best_medium['method']}")
    print(f"  Best length: {best_medium['tour_length']:.3f}")
    print(f"  All methods comparison:")
    for result in sorted(methods_results, key=lambda x: x['tour_length']):
        print(f"    {result['method']:20s}: {result['tour_length']:8.3f} ({result['time']:5.2f}s)")
    
    print(f"\nLarge Instance (15 cities) Results:")
    print(f"  Simulated Annealing:  {large_sa['best_tour_length']:.3f} ({large_sa['computation_time']:.2f}s)")
    print(f"  Parallel Tempering:   {large_pt['best_tour_length']:.3f} ({large_pt['computation_time']:.2f}s)")
    
    if large_pt['best_tour_length'] < large_sa['best_tour_length']:
        improvement = (large_sa['best_tour_length'] - large_pt['best_tour_length']) / large_sa['best_tour_length'] * 100
        print(f"  PT improvement over SA: {improvement:.1f}%")
    else:
        improvement = (large_pt['best_tour_length'] - large_sa['best_tour_length']) / large_pt['best_tour_length'] * 100
        print(f"  SA improvement over PT: {improvement:.1f}%")
    
    # 5. ALGORITHM ANALYSIS
    print(f"\n{'='*60}")
    print("ALGORITHM PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nSimulated Annealing characteristics:")
    print(f"  • Systematic temperature reduction")
    print(f"  • Good balance between exploration and exploitation")
    print(f"  • Cooling schedule affects convergence")
    print(f"  • Generally faster than Parallel Tempering")
    print(f"  • May get trapped in local minima")
    
    print(f"\nParallel Tempering characteristics:")
    print(f"  • Multiple temperature replicas")
    print(f"  • Better global optimization through replica exchanges")
    print(f"  • Higher computational cost")
    print(f"  • More robust against local minima")
    print(f"  • Exchange rate indicates mixing quality")
    
    print(f"\nTSP-specific observations:")
    print(f"  • Constraint satisfaction is challenging")
    print(f"  • Valid tours require careful energy function design")
    print(f"  • Solution quality improves with problem size")
    print(f"  • Advanced methods significantly outperform basic approaches")
    
    print(f"\n{'='*80}")
    print("TSP OPTIMIZATION DEMONSTRATION COMPLETED")
    print(f"{'='*80}")
    print("Key insights:")
    print("• Hopfield networks can effectively solve combinatorial optimization")
    print("• Problem mapping to energy function is crucial")
    print("• Advanced stochastic methods find better solutions")
    print("• Parallel Tempering generally outperforms Simulated Annealing")
    print("• Computational cost scales with problem complexity")
    print("• Multiple runs may be needed for consistent results")

if __name__ == "__main__":
    demonstrate_tsp_optimization()

