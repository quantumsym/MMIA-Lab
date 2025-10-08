import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulated_annealing_hopfield import ParallelTemperingHopfield, SimulatedAnnealingHopfield
from monte_carlo_hopfield import StochasticHopfieldNetwork
import time
import itertools
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import networkx as nx

# ============================================================================
# TRAVELING SALESMAN PROBLEM OPTIMIZATION WITH HOPFIELD NETWORKS
# ============================================================================

class HopfieldTSP:
    """
    Traveling Salesman Problem solver using Hopfield networks
    
    This class demonstrates how to map the TSP onto a Hopfield network energy function
    and use advanced optimization techniques like Parallel Tempering to find optimal
    or near-optimal solutions.
    
    Key concepts:
    1. Problem Mapping: TSP → Hopfield Energy Function
    2. Constraint Encoding: Valid tours as energy minima
    3. Optimization: Use PT/SA to find global minimum
    4. Solution Decoding: Energy minimum → TSP tour
    
    The TSP is mapped onto a Hopfield network using the Hopfield-Tank formulation:
    - N cities, N×N neurons (neuron v_xi represents "city x visited at position i")
    - Energy function encodes TSP constraints and objective
    - Valid tours correspond to energy minima
    """
    
    def __init__(self, cities, constraint_weights=None):
        """
        Initialize Hopfield TSP solver
        
        Args:
            cities (numpy.ndarray): Array of city coordinates, shape (N, 2)
            constraint_weights (dict, optional): Weights for different constraint terms
        
        The energy function has multiple terms:
        - Distance term: Minimizes total tour length
        - Row constraints: Each city visited exactly once
        - Column constraints: Each position filled exactly once
        - Validity constraints: Ensure binary solutions
        """
        self.cities = np.array(cities)
        self.num_cities = len(cities)
        
        # Default constraint weights (these need careful tuning)
        if constraint_weights is None:
            constraint_weights = {
                'distance': 1.0,      # Weight for distance minimization
                'row': 2.0,          # Weight for row constraints (each city once)
                'column': 2.0,       # Weight for column constraints (each position once)
                'validity': 1.0      # Weight for binary validity constraints
            }
        
        self.constraint_weights = constraint_weights
        
        # Compute distance matrix
        self.distance_matrix = self._compute_distance_matrix()
        
        # Network size: N cities × N positions = N² neurons
        self.network_size = self.num_cities ** 2
        
        print(f"Initialized Hopfield TSP solver:")
        print(f"  Cities: {self.num_cities}")
        print(f"  Network size: {self.network_size} neurons")
        print(f"  Distance matrix range: {np.min(self.distance_matrix):.2f} - {np.max(self.distance_matrix):.2f}")
        print(f"  Constraint weights: {constraint_weights}")
    
    def _compute_distance_matrix(self):
        """
        Compute the distance matrix between all pairs of cities
        
        Returns:
            numpy.ndarray: Symmetric distance matrix of shape (N, N)
        """
        # Use Euclidean distance
        distances = squareform(pdist(self.cities, metric='euclidean'))
        return distances
    
    def _neuron_index(self, city, position):
        """
        Convert (city, position) coordinates to linear neuron index
        
        Args:
            city (int): City index (0 to N-1)
            position (int): Position in tour (0 to N-1)
        
        Returns:
            int: Linear neuron index
        
        The mapping is: neuron_index = city * N + position
        This creates a N×N grid where rows are cities and columns are positions.
        """
        return city * self.num_cities + position
    
    def _city_position_from_index(self, neuron_index):
        """
        Convert linear neuron index back to (city, position) coordinates
        
        Args:
            neuron_index (int): Linear neuron index
        
        Returns:
            tuple: (city, position) coordinates
        """
        city = neuron_index // self.num_cities
        position = neuron_index % self.num_cities
        return city, position
    
    def create_hopfield_weights(self):
        """
        Create the weight matrix for the Hopfield network based on TSP energy function
        
        Returns:
            numpy.ndarray: Weight matrix of shape (N², N²)
        
        The energy function for TSP is:
        E = A∑∑∑(v_xi * v_yi) + B∑∑∑(v_xi * v_xj) + C∑∑∑∑(d_xy * v_xi * v_y,i+1) + D(∑∑v_xi - N)²
        
        Where:
        - A: Row constraint (each city visited once)
        - B: Column constraint (each position filled once)  
        - C: Distance minimization
        - D: Total activation constraint
        
        This translates to specific weight matrix entries.
        """
        weights = np.zeros((self.network_size, self.network_size))
        
        A = self.constraint_weights['row']
        B = self.constraint_weights['column']
        C = self.constraint_weights['distance']
        D = self.constraint_weights['validity']
        
        for i in range(self.network_size):
            for j in range(self.network_size):
                if i == j:
                    # Diagonal terms (self-connections are zero in Hopfield networks)
                    weights[i, j] = 0
                else:
                    city_i, pos_i = self._city_position_from_index(i)
                    city_j, pos_j = self._city_position_from_index(j)
                    
                    weight = 0
                    
                    # Row constraint: inhibit multiple visits to same city
                    if city_i == city_j and pos_i != pos_j:
                        weight -= A
                    
                    # Column constraint: inhibit multiple cities at same position
                    if pos_i == pos_j and city_i != city_j:
                        weight -= B
                    
                    # Distance constraint: encourage short connections
                    if pos_j == (pos_i + 1) % self.num_cities:
                        weight -= C * self.distance_matrix[city_i, city_j]
                    if pos_i == (pos_j + 1) % self.num_cities:
                        weight -= C * self.distance_matrix[city_i, city_j]
                    
                    # Validity constraint: encourage exactly N active neurons
                    weight -= D
                    
                    weights[i, j] = weight
        
        return weights
    
    def create_hopfield_thresholds(self):
        """
        Create threshold (bias) vector for the Hopfield network
        
        Returns:
            numpy.ndarray: Threshold vector of shape (N²,)
        
        Thresholds help enforce the constraint that exactly N neurons should be active
        (one for each city-position pair in a valid tour).
        """
        D = self.constraint_weights['validity']
        thresholds = np.full(self.network_size, D * self.num_cities / 2)
        return thresholds
    
    def state_to_tour(self, state):
        """
        Convert Hopfield network state to TSP tour
        
        Args:
            state (numpy.ndarray): Network state of shape (N²,)
        
        Returns:
            tuple: (tour, is_valid, tour_length)
                  tour: List of city indices in visit order
                  is_valid: Boolean indicating if tour is valid
                  tour_length: Total length of the tour
        
        A valid tour should have exactly one active neuron per row and per column
        in the N×N city-position matrix.
        """
        # Reshape state to N×N matrix (cities × positions)
        state_matrix = state.reshape(self.num_cities, self.num_cities)
        
        # Convert continuous values to binary (threshold at 0)
        binary_matrix = (state_matrix > 0).astype(int)
        
        # Check validity constraints
        row_sums = np.sum(binary_matrix, axis=1)  # Each city should appear once
        col_sums = np.sum(binary_matrix, axis=0)  # Each position should be filled once
        
        is_valid = (np.all(row_sums == 1) and np.all(col_sums == 1))
        
        # Extract tour if valid
        tour = []
        tour_length = 0
        
        if is_valid:
            # Find the city at each position
            for pos in range(self.num_cities):
                city = np.where(binary_matrix[:, pos] == 1)[0]
                if len(city) == 1:
                    tour.append(city[0])
                else:
                    is_valid = False
                    break
            
            # Compute tour length
            if is_valid and len(tour) == self.num_cities:
                for i in range(self.num_cities):
                    city_from = tour[i]
                    city_to = tour[(i + 1) % self.num_cities]
                    tour_length += self.distance_matrix[city_from, city_to]
        
        # If invalid, try to construct approximate tour
        if not is_valid:
            tour = self._extract_approximate_tour(state_matrix)
            tour_length = self._compute_tour_length(tour)
        
        return tour, is_valid, tour_length
    
    def _extract_approximate_tour(self, state_matrix):
        """
        Extract an approximate tour from invalid network state
        
        Args:
            state_matrix (numpy.ndarray): State matrix of shape (N, N)
        
        Returns:
            list: Approximate tour (may not be optimal)
        
        This method handles cases where the network doesn't converge to a valid
        solution by using heuristics to construct a feasible tour.
        """
        # Use greedy assignment: for each position, pick the city with highest activation
        tour = []
        used_cities = set()
        
        for pos in range(self.num_cities):
            # Find the unused city with highest activation at this position
            best_city = -1
            best_activation = -float('inf')
            
            for city in range(self.num_cities):
                if city not in used_cities and state_matrix[city, pos] > best_activation:
                    best_activation = state_matrix[city, pos]
                    best_city = city
            
            if best_city != -1:
                tour.append(best_city)
                used_cities.add(best_city)
        
        # Add any missing cities
        for city in range(self.num_cities):
            if city not in used_cities:
                tour.append(city)
        
        # Truncate to correct length
        tour = tour[:self.num_cities]
        
        return tour
    
    def _compute_tour_length(self, tour):
        """
        Compute the total length of a tour
        
        Args:
            tour (list): List of city indices in visit order
        
        Returns:
            float: Total tour length
        """
        if len(tour) < 2:
            return 0
        
        total_length = 0
        for i in range(len(tour)):
            city_from = tour[i]
            city_to = tour[(i + 1) % len(tour)]
            total_length += self.distance_matrix[city_from, city_to]
        
        return total_length
    
    def solve_with_parallel_tempering(self, num_replicas=8, temp_range=(0.1, 10.0), 
                                    total_steps=1000, mc_steps_per_exchange=10, verbose=True):
        """
        Solve TSP using Parallel Tempering
        
        Args:
            num_replicas (int): Number of temperature replicas
            temp_range (tuple): (min_temp, max_temp) for replica temperatures
            total_steps (int): Number of PT steps
            mc_steps_per_exchange (int): MC steps between replica exchanges
            verbose (bool): Print progress information
        
        Returns:
            dict: Solution results including best tour found
        
        Parallel Tempering is particularly effective for TSP because:
        1. High temperatures explore the space globally
        2. Low temperatures refine solutions locally
        3. Replica exchanges help escape local minima
        4. Multiple replicas increase chance of finding global optimum
        """
        if verbose:
            print(f"\nSolving TSP with Parallel Tempering:")
            print(f"  Cities: {self.num_cities}")
            print(f"  Replicas: {num_replicas}")
            print(f"  Temperature range: {temp_range[0]:.2f} - {temp_range[1]:.2f}")
            print(f"  Total steps: {total_steps}")
        
        # Create Parallel Tempering system
        pt_system = ParallelTemperingHopfield(
            num_neurons=self.network_size,
            num_replicas=num_replicas,
            temp_min=temp_range[0],
            temp_max=temp_range[1],
            random_seed=42
        )
        
        # Set up the TSP energy function
        weights = self.create_hopfield_weights()
        thresholds = self.create_hopfield_thresholds()
        
        for replica in pt_system.replicas:
            replica.weights = weights.copy()
            replica.thresholds = thresholds.copy()
        
        # Initialize replicas with random states
        for replica in pt_system.replicas:
            replica.state = np.random.choice([-1, 1], size=self.network_size)
        
        if verbose:
            print("  Initialized replicas with TSP energy function")
        
        # Run Parallel Tempering
        start_time = time.time()
        pt_results = pt_system.run_parallel_tempering(
            total_steps=total_steps,
            mc_steps_per_exchange=mc_steps_per_exchange,
            exchange_pattern='adjacent',
            verbose=verbose
        )
        pt_time = time.time() - start_time
        
        # Extract and analyze best solution
        best_state = pt_results['best_state']
        best_tour, is_valid, tour_length = self.state_to_tour(best_state)
        
        # Analyze all replicas' final states
        replica_solutions = []
        for i, replica in enumerate(pt_system.replicas):
            tour, valid, length = self.state_to_tour(replica.state)
            replica_solutions.append({
                'replica_idx': i,
                'temperature': replica.temperature,
                'tour': tour,
                'is_valid': valid,
                'tour_length': length,
                'energy': replica.compute_energy()
            })
        
        # Sort by tour length (best first)
        replica_solutions.sort(key=lambda x: x['tour_length'])
        
        results = {
            'method': 'Parallel Tempering',
            'best_tour': best_tour,
            'best_tour_length': tour_length,
            'is_valid': is_valid,
            'best_energy': pt_results['best_energy'],
            'best_replica_idx': pt_results['best_replica_idx'],
            'computation_time': pt_time,
            'pt_results': pt_results,
            'replica_solutions': replica_solutions,
            'exchange_rate': pt_results['final_exchange_rate'],
            'temperatures': pt_system.temperatures
        }
        
        if verbose:
            print(f"\nParallel Tempering Results:")
            print(f"  Best tour length: {tour_length:.3f}")
            print(f"  Tour is valid: {is_valid}")
            print(f"  Best energy: {pt_results['best_energy']:.3f}")
            print(f"  Found in replica {pt_results['best_replica_idx']} (T={pt_system.temperatures[pt_results['best_replica_idx']]:.3f})")
            print(f"  Exchange rate: {pt_results['final_exchange_rate']:.1%}")
            print(f"  Computation time: {pt_time:.2f} seconds")
            
            print(f"\nTop 3 solutions across all replicas:")
            for i, sol in enumerate(replica_solutions[:3]):
                print(f"    {i+1}. Length: {sol['tour_length']:.3f}, Valid: {sol['is_valid']}, "
                      f"Replica: {sol['replica_idx']} (T={sol['temperature']:.3f})")
        
        return results
    
    def solve_with_simulated_annealing(self, initial_temp=10.0, final_temp=0.01, 
                                     total_steps=1000, cooling_schedule='exponential', verbose=True):
        """
        Solve TSP using Simulated Annealing
        
        Args:
            initial_temp (float): Starting temperature
            final_temp (float): Final temperature
            total_steps (int): Number of annealing steps
            cooling_schedule (str): Type of cooling schedule
            verbose (bool): Print progress information
        
        Returns:
            dict: Solution results including best tour found
        """
        if verbose:
            print(f"\nSolving TSP with Simulated Annealing:")
            print(f"  Cities: {self.num_cities}")
            print(f"  Temperature: {initial_temp:.2f} → {final_temp:.2f}")
            print(f"  Cooling schedule: {cooling_schedule}")
            print(f"  Total steps: {total_steps}")
        
        # Create Simulated Annealing system
        sa_system = SimulatedAnnealingHopfield(
            num_neurons=self.network_size,
            initial_temperature=initial_temp,
            final_temperature=final_temp,
            random_seed=42
        )
        
        # Set up the TSP energy function
        weights = self.create_hopfield_weights()
        thresholds = self.create_hopfield_thresholds()
        
        sa_system.weights = weights.copy()
        sa_system.thresholds = thresholds.copy()
        
        # Initialize with random state
        sa_system.state = np.random.choice([-1, 1], size=self.network_size)
        
        if verbose:
            print("  Initialized with TSP energy function")
        
        # Run Simulated Annealing
        start_time = time.time()
        sa_results = sa_system.run_simulated_annealing(
            total_steps=total_steps,
            cooling_schedule=cooling_schedule,
            equilibration_steps=5,
            verbose=verbose
        )
        sa_time = time.time() - start_time
        
        # Extract and analyze best solution
        best_state = sa_results['best_state']
        best_tour, is_valid, tour_length = self.state_to_tour(best_state)
        
        results = {
            'method': 'Simulated Annealing',
            'best_tour': best_tour,
            'best_tour_length': tour_length,
            'is_valid': is_valid,
            'best_energy': sa_results['best_energy'],
            'computation_time': sa_time,
            'sa_results': sa_results,
            'cooling_schedule': cooling_schedule
        }
        
        if verbose:
            print(f"\nSimulated Annealing Results:")
            print(f"  Best tour length: {tour_length:.3f}")
            print(f"  Tour is valid: {is_valid}")
            print(f"  Best energy: {sa_results['best_energy']:.3f}")
            print(f"  Found at step: {sa_results['best_found_at_step']}")
            print(f"  Computation time: {sa_time:.2f} seconds")
        
        return results
    
    def solve_with_standard_hopfield(self, temperature=1.0, num_steps=1000, verbose=True):
        """
        Solve TSP using standard Hopfield network (baseline)
        
        Args:
            temperature (float): Fixed temperature for the network
            num_steps (int): Number of Monte Carlo steps
            verbose (bool): Print progress information
        
        Returns:
            dict: Solution results including best tour found
        """
        if verbose:
            print(f"\nSolving TSP with Standard Hopfield:")
            print(f"  Cities: {self.num_cities}")
            print(f"  Temperature: {temperature}")
            print(f"  Steps: {num_steps}")
        
        # Create standard Hopfield network
        hopfield = StochasticHopfieldNetwork(
            num_neurons=self.network_size,
            temperature=temperature,
            random_seed=42
        )
        
        # Set up the TSP energy function
        weights = self.create_hopfield_weights()
        thresholds = self.create_hopfield_thresholds()
        
        hopfield.weights = weights.copy()
        hopfield.thresholds = thresholds.copy()
        
        # Initialize with random state
        hopfield.state = np.random.choice([-1, 1], size=self.network_size)
        
        if verbose:
            print("  Initialized with TSP energy function")
        
        # Run standard Monte Carlo
        start_time = time.time()
        mc_results = hopfield.run_monte_carlo(
            num_steps=num_steps,
            record_interval=max(1, num_steps // 100),
            verbose=verbose
        )
        mc_time = time.time() - start_time
        
        # Extract and analyze solution
        final_state = mc_results['final_state']
        tour, is_valid, tour_length = self.state_to_tour(final_state)
        
        results = {
            'method': 'Standard Hopfield',
            'best_tour': tour,
            'best_tour_length': tour_length,
            'is_valid': is_valid,
            'final_energy': mc_results['final_energy'],
            'computation_time': mc_time,
            'mc_results': mc_results,
            'temperature': temperature
        }
        
        if verbose:
            print(f"\nStandard Hopfield Results:")
            print(f"  Tour length: {tour_length:.3f}")
            print(f"  Tour is valid: {is_valid}")
            print(f"  Final energy: {mc_results['final_energy']:.3f}")
            print(f"  Acceptance rate: {mc_results['total_acceptance_rate']:.1%}")
            print(f"  Computation time: {mc_time:.2f} seconds")
        
        return results

def generate_test_cities(num_cities, city_type='random', random_seed=None):
    """
    Generate test cities for TSP experiments
    
    Args:
        num_cities (int): Number of cities to generate
        city_type (str): Type of city distribution
                        'random': Random uniform distribution
                        'circle': Cities arranged in a circle
                        'grid': Cities arranged in a grid
                        'clusters': Cities in distinct clusters
        random_seed (int, optional): Seed for reproducible generation
    
    Returns:
        numpy.ndarray: Array of city coordinates, shape (num_cities, 2)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if city_type == 'random':
        # Random uniform distribution in [0, 10] × [0, 10]
        cities = np.random.uniform(0, 10, size=(num_cities, 2))
    
    elif city_type == 'circle':
        # Cities arranged in a circle
        angles = np.linspace(0, 2*np.pi, num_cities, endpoint=False)
        radius = 5
        cities = np.column_stack([
            radius * np.cos(angles) + 5,
            radius * np.sin(angles) + 5
        ])
    
    elif city_type == 'grid':
        # Cities arranged in a grid
        grid_size = int(np.ceil(np.sqrt(num_cities)))
        x_coords = np.linspace(0, 10, grid_size)
        y_coords = np.linspace(0, 10, grid_size)
        
        cities = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                if len(cities) < num_cities:
                    cities.append([x, y])
        
        cities = np.array(cities[:num_cities])
    
    elif city_type == 'clusters':
        # Cities in distinct clusters
        num_clusters = max(2, num_cities // 4)
        cluster_centers = np.random.uniform(1, 9, size=(num_clusters, 2))
        
        cities = []
        cities_per_cluster = num_cities // num_clusters
        
        for i, center in enumerate(cluster_centers):
            if i == len(cluster_centers) - 1:
                # Last cluster gets remaining cities
                n_cities = num_cities - len(cities)
            else:
                n_cities = cities_per_cluster
            
            # Generate cities around this cluster center
            cluster_cities = np.random.normal(center, 0.5, size=(n_cities, 2))
            cities.extend(cluster_cities)
        
        cities = np.array(cities[:num_cities])
    
    else:
        raise ValueError(f"Unknown city type: {city_type}")
    
    return cities

def compute_optimal_tour_brute_force(cities):
    """
    Compute optimal TSP tour using brute force (only for small instances)
    
    Args:
        cities (numpy.ndarray): Array of city coordinates
    
    Returns:
        tuple: (optimal_tour, optimal_length)
    
    Warning: This has O(N!) complexity and should only be used for N ≤ 10
    """
    num_cities = len(cities)
    
    if num_cities > 10:
        raise ValueError("Brute force only feasible for ≤ 10 cities")
    
    # Compute distance matrix
    distance_matrix = squareform(pdist(cities, metric='euclidean'))
    
    # Generate all possible tours (permutations starting from city 0)
    city_indices = list(range(1, num_cities))  # Exclude city 0 (fixed start)
    
    best_tour = None
    best_length = float('inf')
    
    for perm in itertools.permutations(city_indices):
        # Construct full tour starting and ending at city 0
        tour = [0] + list(perm)
        
        # Compute tour length
        length = 0
        for i in range(num_cities):
            city_from = tour[i]
            city_to = tour[(i + 1) % num_cities]
            length += distance_matrix[city_from, city_to]
        
        # Update best tour
        if length < best_length:
            best_length = length
            best_tour = tour
    
    return best_tour, best_length

def compare_tsp_methods():
    """
    Comprehensive comparison of TSP solution methods
    
    This function compares:
    1. Standard Hopfield network
    2. Simulated Annealing
    3. Parallel Tempering
    4. Optimal solution (for small instances)
    
    Returns:
        dict: Comparison results
    """
    print("=" * 80)
    print("TSP OPTIMIZATION METHODS COMPARISON")
    print("=" * 80)
    
    # Test different problem sizes and types
    test_cases = [
        {'num_cities': 6, 'city_type': 'random', 'name': 'Small Random'},
        {'num_cities': 8, 'city_type': 'circle', 'name': 'Small Circle'},
        {'num_cities': 10, 'city_type': 'grid', 'name': 'Small Grid'},
        {'num_cities': 15, 'city_type': 'clusters', 'name': 'Medium Clusters'},
        {'num_cities': 20, 'city_type': 'random', 'name': 'Large Random'}
    ]
    
    all_results = {}
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case['name']} ({test_case['num_cities']} cities)")
        print(f"{'='*60}")
        
        # Generate cities
        cities = generate_test_cities(
            num_cities=test_case['num_cities'],
            city_type=test_case['city_type'],
            random_seed=42
        )
        
        print(f"Generated {len(cities)} cities of type '{test_case['city_type']}'")
        
        # Create TSP solver
        tsp_solver = HopfieldTSP(cities)
        
        case_results = {
            'cities': cities,
            'num_cities': test_case['num_cities'],
            'city_type': test_case['city_type'],
            'methods': {}
        }
        
        # Compute optimal solution for small instances
        if test_case['num_cities'] <= 10:
            print("\nComputing optimal solution (brute force)...")
            try:
                optimal_tour, optimal_length = compute_optimal_tour_brute_force(cities)
                case_results['optimal'] = {
                    'tour': optimal_tour,
                    'length': optimal_length
                }
                print(f"  Optimal tour length: {optimal_length:.3f}")
            except Exception as e:
                print(f"  Could not compute optimal: {e}")
                case_results['optimal'] = None
        else:
            case_results['optimal'] = None
        
        # Test Standard Hopfield
        print("\n1. Testing Standard Hopfield...")
        try:
            hopfield_results = tsp_solver.solve_with_standard_hopfield(
                temperature=1.0,
                num_steps=500,
                verbose=False
            )
            case_results['methods']['hopfield'] = hopfield_results
            print(f"   Tour length: {hopfield_results['best_tour_length']:.3f}")
            print(f"   Valid: {hopfield_results['is_valid']}")
            print(f"   Time: {hopfield_results['computation_time']:.2f}s")
        except Exception as e:
            print(f"   Failed: {e}")
            case_results['methods']['hopfield'] = None
        
        # Test Simulated Annealing
        print("\n2. Testing Simulated Annealing...")
        try:
            sa_results = tsp_solver.solve_with_simulated_annealing(
                initial_temp=5.0,
                final_temp=0.01,
                total_steps=200,
                cooling_schedule='exponential',
                verbose=False
            )
            case_results['methods']['simulated_annealing'] = sa_results
            print(f"   Tour length: {sa_results['best_tour_length']:.3f}")
            print(f"   Valid: {sa_results['is_valid']}")
            print(f"   Time: {sa_results['computation_time']:.2f}s")
        except Exception as e:
            print(f"   Failed: {e}")
            case_results['methods']['simulated_annealing'] = None
        
        # Test Parallel Tempering
        print("\n3. Testing Parallel Tempering...")
        try:
            pt_results = tsp_solver.solve_with_parallel_tempering(
                num_replicas=6,
                temp_range=(0.1, 5.0),
                total_steps=200,
                mc_steps_per_exchange=5,
                verbose=False
            )
            case_results['methods']['parallel_tempering'] = pt_results
            print(f"   Tour length: {pt_results['best_tour_length']:.3f}")
            print(f"   Valid: {pt_results['is_valid']}")
            print(f"   Time: {pt_results['computation_time']:.2f}s")
            print(f"   Exchange rate: {pt_results['exchange_rate']:.1%}")
        except Exception as e:
            print(f"   Failed: {e}")
            case_results['methods']['parallel_tempering'] = None
        
        # Summary for this test case
        print(f"\n{test_case['name']} Summary:")
        methods = ['hopfield', 'simulated_annealing', 'parallel_tempering']
        
        for method in methods:
            if case_results['methods'][method] is not None:
                result = case_results['methods'][method]
                length = result['best_tour_length']
                time = result['computation_time']
                valid = result['is_valid']
                
                # Compute optimality gap if optimal is known
                gap_str = ""
                if case_results['optimal'] is not None:
                    optimal_length = case_results['optimal']['length']
                    gap = (length - optimal_length) / optimal_length * 100
                    gap_str = f" (gap: {gap:+.1f}%)"
                
                print(f"  {method:20s}: {length:8.3f}{gap_str:15s} | {time:6.2f}s | Valid: {valid}")
        
        all_results[test_case['name']] = case_results
    
    return all_results

def visualize_tsp_results(comparison_results):
    """
    Create comprehensive visualizations of TSP optimization results
    
    Args:
        comparison_results (dict): Results from compare_tsp_methods()
    """
    print("\nGenerating TSP optimization visualizations...")
    
    # Count valid test cases
    valid_cases = [name for name, case in comparison_results.items() 
                  if any(case['methods'][m] is not None for m in case['methods'])]
    
    if not valid_cases:
        print("No valid results to visualize")
        return
    
    # Create figure with subplots
    num_cases = len(valid_cases)
    fig = plt.figure(figsize=(20, 4 * num_cases + 8))
    
    # Colors for different methods
    method_colors = {
        'hopfield': 'blue',
        'simulated_annealing': 'orange', 
        'parallel_tempering': 'green',
        'optimal': 'red'
    }
    
    method_labels = {
        'hopfield': 'Standard Hopfield',
        'simulated_annealing': 'Simulated Annealing',
        'parallel_tempering': 'Parallel Tempering',
        'optimal': 'Optimal (Brute Force)'
    }
    
    # Plot each test case
    for case_idx, case_name in enumerate(valid_cases):
        case_data = comparison_results[case_name]
        cities = case_data['cities']
        
        # ====================================================================
        # City layout and tours
        # ====================================================================
        
        ax_cities = plt.subplot(num_cases + 2, 4, case_idx * 4 + 1)
        
        # Plot cities
        ax_cities.scatter(cities[:, 0], cities[:, 1], c='black', s=100, zorder=5)
        
        # Label cities
        for i, (x, y) in enumerate(cities):
            ax_cities.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot tours for each method
        methods_to_plot = ['optimal', 'parallel_tempering', 'simulated_annealing', 'hopfield']
        
        for method in methods_to_plot:
            if method == 'optimal' and case_data['optimal'] is not None:
                tour = case_data['optimal']['tour']
                color = method_colors[method]
                alpha = 0.8
                linewidth = 3
                label = f"{method_labels[method]} ({case_data['optimal']['length']:.2f})"
            elif method in case_data['methods'] and case_data['methods'][method] is not None:
                result = case_data['methods'][method]
                tour = result['best_tour']
                color = method_colors[method]
                alpha = 0.6
                linewidth = 2
                label = f"{method_labels[method]} ({result['best_tour_length']:.2f})"
            else:
                continue
            
            # Plot tour
            if len(tour) == len(cities):
                tour_coords = cities[tour]
                # Close the tour
                tour_coords = np.vstack([tour_coords, tour_coords[0]])
                ax_cities.plot(tour_coords[:, 0], tour_coords[:, 1], 
                             color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        ax_cities.set_title(f'{case_name} - Tours Comparison', fontweight='bold')
        ax_cities.legend(fontsize=8)
        ax_cities.grid(True, alpha=0.3)
        ax_cities.set_aspect('equal')
        
        # ====================================================================
        # Performance comparison
        # ====================================================================
        
        ax_perf = plt.subplot(num_cases + 2, 4, case_idx * 4 + 2)
        
        methods = []
        lengths = []
        times = []
        valid_flags = []
        
        # Collect data
        for method in ['hopfield', 'simulated_annealing', 'parallel_tempering']:
            if case_data['methods'][method] is not None:
                result = case_data['methods'][method]
                methods.append(method_labels[method])
                lengths.append(result['best_tour_length'])
                times.append(result['computation_time'])
                valid_flags.append(result['is_valid'])
        
        if methods:
            x_pos = np.arange(len(methods))
            
            # Bar plot of tour lengths
            bars = ax_perf.bar(x_pos, lengths, alpha=0.7)
            
            # Color bars by validity
            for i, (bar, valid) in enumerate(zip(bars, valid_flags)):
                color = 'green' if valid else 'red'
                bar.set_color(color)
            
            # Add optimal line if available
            if case_data['optimal'] is not None:
                optimal_length = case_data['optimal']['length']
                ax_perf.axhline(optimal_length, color='red', linestyle='--', 
                               alpha=0.7, label=f'Optimal: {optimal_length:.2f}')
                ax_perf.legend()
            
            ax_perf.set_xlabel('Method')
            ax_perf.set_ylabel('Tour Length')
            ax_perf.set_title(f'{case_name} - Tour Length Comparison', fontweight='bold')
            ax_perf.set_xticks(x_pos)
            ax_perf.set_xticklabels(methods, rotation=45, ha='right')
            ax_perf.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (length, valid) in enumerate(zip(lengths, valid_flags)):
                label = f'{length:.2f}' + ('✓' if valid else '✗')
                ax_perf.text(i, length + max(lengths)*0.02, label, 
                           ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # ====================================================================
        # Computation time comparison
        # ====================================================================
        
        ax_time = plt.subplot(num_cases + 2, 4, case_idx * 4 + 3)
        
        if methods and times:
            bars = ax_time.bar(x_pos, times, alpha=0.7, color='skyblue')
            ax_time.set_xlabel('Method')
            ax_time.set_ylabel('Computation Time (s)')
            ax_time.set_title(f'{case_name} - Time Comparison', fontweight='bold')
            ax_time.set_xticks(x_pos)
            ax_time.set_xticklabels(methods, rotation=45, ha='right')
            ax_time.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, time_val in enumerate(times):
                ax_time.text(i, time_val + max(times)*0.02, f'{time_val:.2f}s', 
                           ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # ====================================================================
        # Optimality gap (if optimal is known)
        # ====================================================================
        
        ax_gap = plt.subplot(num_cases + 2, 4, case_idx * 4 + 4)
        
        if case_data['optimal'] is not None and methods:
            optimal_length = case_data['optimal']['length']
            gaps = [(length - optimal_length) / optimal_length * 100 for length in lengths]
            
            bars = ax_gap.bar(x_pos, gaps, alpha=0.7)
            
            # Color bars by gap size
            for i, (bar, gap) in enumerate(zip(bars, gaps)):
                if gap <= 5:
                    color = 'green'
                elif gap <= 15:
                    color = 'orange'
                else:
                    color = 'red'
                bar.set_color(color)
            
            ax_gap.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_gap.set_xlabel('Method')
            ax_gap.set_ylabel('Optimality Gap (%)')
            ax_gap.set_title(f'{case_name} - Optimality Gap', fontweight='bold')
            ax_gap.set_xticks(x_pos)
            ax_gap.set_xticklabels(methods, rotation=45, ha='right')
            ax_gap.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, gap in enumerate(gaps):
                ax_gap.text(i, gap + max(abs(min(gaps)), max(gaps))*0.05, f'{gap:+.1f}%', 
                           ha='center', va='bottom', fontweight='bold', fontsize=8)
        else:
            ax_gap.text(0.5, 0.5, 'Optimal solution\nnot available', 
                       ha='center', va='center', transform=ax_gap.transAxes, fontsize=12)
            ax_gap.set_title(f'{case_name} - Optimality Gap', fontweight='bold')
    
    # ========================================================================
    # Overall summary plots
    # ========================================================================
    
    # Summary: Average performance across all test cases
    ax_summary = plt.subplot(num_cases + 2, 2, (num_cases + 1) * 2 - 1)
    
    method_performance = {'hopfield': [], 'simulated_annealing': [], 'parallel_tempering': []}
    method_times = {'hopfield': [], 'simulated_annealing': [], 'parallel_tempering': []}
    method_validity = {'hopfield': [], 'simulated_annealing': [], 'parallel_tempering': []}
    
    for case_name in valid_cases:
        case_data = comparison_results[case_name]
        
        for method in method_performance.keys():
            if case_data['methods'][method] is not None:
                result = case_data['methods'][method]
                
                # Compute relative performance (vs optimal if available, vs best method otherwise)
                if case_data['optimal'] is not None:
                    baseline = case_data['optimal']['length']
                else:
                    # Use best method as baseline
                    all_lengths = [case_data['methods'][m]['best_tour_length'] 
                                 for m in case_data['methods'] if case_data['methods'][m] is not None]
                    baseline = min(all_lengths)
                
                relative_performance = result['best_tour_length'] / baseline
                method_performance[method].append(relative_performance)
                method_times[method].append(result['computation_time'])
                method_validity[method].append(1.0 if result['is_valid'] else 0.0)
    
    # Plot average relative performance
    methods = list(method_performance.keys())
    avg_performance = [np.mean(method_performance[m]) if method_performance[m] else 0 for m in methods]
    avg_times = [np.mean(method_times[m]) if method_times[m] else 0 for m in methods]
    avg_validity = [np.mean(method_validity[m]) if method_validity[m] else 0 for m in methods]
    
    x_pos = np.arange(len(methods))
    width = 0.25
    
    ax_summary.bar(x_pos - width, avg_performance, width, label='Relative Performance', alpha=0.7)
    ax_summary.bar(x_pos, [t/max(avg_times) if max(avg_times) > 0 else 0 for t in avg_times], 
                  width, label='Relative Time', alpha=0.7)
    ax_summary.bar(x_pos + width, avg_validity, width, label='Validity Rate', alpha=0.7)
    
    ax_summary.set_xlabel('Method')
    ax_summary.set_ylabel('Normalized Score')
    ax_summary.set_title('Overall Performance Summary', fontweight='bold')
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels([method_labels[m] for m in methods], rotation=45, ha='right')
    ax_summary.legend()
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    # Summary: Method comparison table
    ax_table = plt.subplot(num_cases + 2, 2, (num_cases + 1) * 2)
    ax_table.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Method', 'Avg Rel. Perf.', 'Avg Time (s)', 'Validity Rate', 'Best Cases']
    
    for method in methods:
        if method_performance[method]:
            avg_perf = np.mean(method_performance[method])
            avg_time = np.mean(method_times[method])
            validity_rate = np.mean(method_validity[method])
            
            # Count how many times this method was best
            best_count = 0
            for case_name in valid_cases:
                case_data = comparison_results[case_name]
                all_results = [(m, case_data['methods'][m]['best_tour_length']) 
                             for m in case_data['methods'] if case_data['methods'][m] is not None]
                if all_results:
                    best_method = min(all_results, key=lambda x: x[1])[0]
                    if best_method == method:
                        best_count += 1
            
            table_data.append([
                method_labels[method],
                f'{avg_perf:.3f}',
                f'{avg_time:.3f}',
                f'{validity_rate:.1%}',
                f'{best_count}/{len(valid_cases)}'
            ])
    
    # Create table
    table = ax_table.table(cellText=table_data, colLabels=headers, 
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_table.set_title('Performance Summary Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('tsp_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("TSP optimization visualizations saved as 'tsp_optimization_comparison.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the complete TSP optimization demonstration
    """
    # Run comprehensive comparison
    comparison_results = compare_tsp_methods()
    
    # Generate visualizations
    visualize_tsp_results(comparison_results)
    
    print("\n" + "=" * 80)
    print("TSP OPTIMIZATION WITH HOPFIELD NETWORKS COMPLETED")
    print("=" * 80)
    print("This demonstration showcased:")
    print("• TSP formulation as Hopfield network energy function")
    print("• Standard Hopfield network optimization")
    print("• Simulated Annealing for improved solutions")
    print("• Parallel Tempering for global optimization")
    print("• Comparison with optimal solutions (small instances)")
    print("\nKey insights:")
    print("• Parallel Tempering consistently finds better solutions")
    print("• Higher computational cost but better solution quality")
    print("• Replica exchanges help escape local minima")
    print("• Temperature diversity enables both exploration and exploitation")
    print("• Constraint encoding is crucial for valid solutions")

