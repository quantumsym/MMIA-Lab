
"""
ERROR ESTIMATION IN PYTHON: Computing Norms in R^n Space
========================================================

This script demonstrates how to calculate different error metrics between
target values and predicted values using various norms in R^n space.

Key concepts:
- Outlier: An observation that lies an abnormal distance from other values 
  in a dataset. Outliers can skew results and affect model performance.
- Regression: A statistical method used to model the relationship between 
  a dependent variable and one or more independent variables, typically 
  used for prediction.

NumPy functions used:
- np.mean(): Computes the arithmetic mean (average) of array elements
- np.sqrt(): Returns the square root of each element in the array
- np.abs(): Computes the absolute value of each element
- len(): Returns the number of items in an object (built-in Python function)
- np.max(): Returns the maximum value along an axis
- np.linalg.norm(): Computes vector or matrix norms (various types)
"""

import numpy as np

# Set random seed for reproducible results
np.random.seed(42)

def demonstrate_error_norms():
    """
    Demonstrates calculation of different error norms with detailed examples.

    This function creates sample data and shows how to compute:
    1. L² norm (Euclidean) - RMSE (Root Mean Square Error)
    2. L¹ norm (Manhattan) - MAE (Mean Absolute Error) 
    3. L∞ norm (Chebyshev) - Maximum absolute error
    """

    print("="*80)
    print("ERROR ESTIMATION WITH DIFFERENT NORMS IN R^n SPACE")
    print("="*80)

    # Generate sample data for demonstration
    n_samples = 8
    target_values = np.array([2.5, 4.1, 1.8, 6.2, 3.7, 5.9, 2.1, 4.8])
    predicted_values = np.array([2.3, 4.5, 2.1, 5.8, 3.2, 6.3, 2.5, 4.6])

    print(f"Sample size (n): {n_samples}")
    print(f"Target values:    {target_values}")
    print(f"Predicted values: {predicted_values}")

    # Calculate error vector (difference between predicted and target)
    error_vector = predicted_values - target_values
    print(f"Error vector:     {error_vector}")
    print()

    # ========================================================================
    # 1. EUCLIDEAN NORM L² (Root Mean Square Error - RMSE)
    # ========================================================================
    print("1. EUCLIDEAN NORM L² - Root Mean Square Error (RMSE)")
    print("-" * 60)
    print("   Characteristics:")
    print("   - Penalizes larger errors more heavily due to squaring")
    print("   - Sensitive to outliers")
    print("   - Most commonly used in regression problems")
    print()

    # Method 1: Direct calculation (most common approach)
    squared_errors = error_vector**2  # Square each error
    mean_squared_error = np.mean(squared_errors)  # Average of squared errors
    rmse_direct = np.sqrt(mean_squared_error)  # Square root of MSE

    print(f"   Step-by-step calculation:")
    print(f"   - Squared errors: {squared_errors}")
    print(f"   - Mean squared error (MSE): {mean_squared_error:.6f}")
    print(f"   - RMSE (√MSE): {rmse_direct:.6f}")

    # Method 2: Using numpy.linalg.norm
    l2_norm = np.linalg.norm(error_vector, ord=2)  # L² norm of error vector
    rmse_linalg = l2_norm / np.sqrt(len(error_vector))  # Normalize by √n

    print(f"   Using np.linalg.norm:")
    print(f"   - L² norm: {l2_norm:.6f}")
    print(f"   - RMSE: {rmse_linalg:.6f}")
    print()

    # ========================================================================
    # 2. MANHATTAN NORM L¹ (Mean Absolute Error - MAE)
    # ========================================================================
    print("2. MANHATTAN NORM L¹ - Mean Absolute Error (MAE)")
    print("-" * 60)
    print("   Characteristics:")
    print("   - Treats all errors equally (linear penalty)")
    print("   - More robust to outliers than RMSE")
    print("   - Easier to interpret (average absolute deviation)")
    print()

    # Method 1: Direct calculation (most common approach)
    absolute_errors = np.abs(error_vector)  # Absolute value of each error
    mae_direct = np.mean(absolute_errors)  # Mean of absolute errors

    print(f"   Step-by-step calculation:")
    print(f"   - Absolute errors: {absolute_errors}")
    print(f"   - MAE (mean of absolute errors): {mae_direct:.6f}")

    # Method 2: Using numpy.linalg.norm
    l1_norm = np.linalg.norm(error_vector, ord=1)  # L¹ norm of error vector
    mae_linalg = l1_norm / len(error_vector)  # Normalize by n

    print(f"   Using np.linalg.norm:")
    print(f"   - L¹ norm: {l1_norm:.6f}")
    print(f"   - MAE: {mae_linalg:.6f}")
    print()

    # ========================================================================
    # 3. CHEBYSHEV NORM L∞ (Maximum Absolute Error)
    # ========================================================================
    print("3. CHEBYSHEV NORM L∞ - Maximum Absolute Error")
    print("-" * 60)
    print("   Characteristics:")
    print("   - Considers only the worst-case error")
    print("   - Useful for safety-critical applications")
    print("   - Pessimistic measure (worst-case scenario)")
    print()

    # Method 1: Direct calculation
    max_absolute_error = np.max(absolute_errors)  # Maximum absolute error

    print(f"   Step-by-step calculation:")
    print(f"   - Absolute errors: {absolute_errors}")
    print(f"   - Maximum absolute error: {max_absolute_error:.6f}")

    # Method 2: Using numpy.linalg.norm
    linf_norm = np.linalg.norm(error_vector, ord=np.inf)  # L∞ norm

    print(f"   Using np.linalg.norm:")
    print(f"   - L∞ norm: {linf_norm:.6f}")
    print()

    return {
        'RMSE': rmse_direct,
        'MAE': mae_direct,
        'Max_Error': max_absolute_error,
        'MSE': mean_squared_error
    }


def compute_all_error_metrics(y_true, y_pred, verbose=True):
    """
    Compute all three error metrics for given target and predicted values.

    Parameters:
    -----------
    y_true : numpy.ndarray
        True/target values (ground truth)
    y_pred : numpy.ndarray  
        Predicted values from a model
    verbose : bool, optional
        If True, print detailed explanations (default: True)

    Returns:
    --------
    dict
        Dictionary containing all computed error metrics

    Notes:
    ------
    - RMSE is sensitive to outliers due to squaring operation
    - MAE is more robust to outliers (linear penalty)
    - Max Error shows worst-case performance
    """

    # Input validation
    if len(y_true) != len(y_pred):
        raise ValueError("Target and predicted arrays must have same length")

    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate error vector (residuals)
    error_vector = y_pred - y_true

    # Compute all metrics
    # L² norm: Root Mean Square Error (RMSE)
    mse = np.mean(error_vector**2)  # Mean of squared errors
    rmse = np.sqrt(mse)  # Square root of MSE

    # L¹ norm: Mean Absolute Error (MAE)  
    mae = np.mean(np.abs(error_vector))  # Mean of absolute errors

    # L∞ norm: Maximum Absolute Error
    max_error = np.max(np.abs(error_vector))  # Maximum absolute error

    # Create results dictionary
    results = {
        'RMSE': rmse,           # Root Mean Square Error
        'MAE': mae,             # Mean Absolute Error  
        'Max_Error': max_error, # Maximum Absolute Error
        'MSE': mse,             # Mean Square Error (bonus metric)
        'n_samples': len(y_true) # Number of samples
    }

    if verbose:
        print(f"Error Metrics Summary (n={len(y_true)} samples):")
        print(f"  RMSE (L² norm):     {rmse:.6f}")
        print(f"  MAE (L¹ norm):      {mae:.6f}")
        print(f"  Max Error (L∞):     {max_error:.6f}")
        print(f"  MSE:                {mse:.6f}")

    return results


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Run the detailed demonstration
    demo_results = demonstrate_error_norms()

    print("="*80)
    print("PRACTICAL USAGE EXAMPLES")
    print("="*80)

    # Example 1: Simple case
    print("Example 1: Simple prediction scenario")
    target_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted_1 = np.array([1.1, 1.9, 3.2, 3.8, 5.3])

    print(f"Target:    {target_1}")
    print(f"Predicted: {predicted_1}")
    results_1 = compute_all_error_metrics(target_1, predicted_1, verbose=True)
    print()

    # Example 2: Case with outlier
    print("Example 2: Scenario with outlier (notice RMSE vs MAE difference)")
    target_2 = np.array([10, 20, 30, 40, 50])
    predicted_2 = np.array([12, 19, 28, 35, 80])  # Last value is outlier

    print(f"Target:    {target_2}")
    print(f"Predicted: {predicted_2} <- outlier at position 4")
    results_2 = compute_all_error_metrics(target_2, predicted_2, verbose=True)

    print("\nNotice how RMSE is much higher than MAE due to the outlier!")
    print("This demonstrates RMSE's sensitivity to outliers.")

    print("\n" + "="*80)
    print("CODE TEMPLATES FOR QUICK USE")
    print("="*80)

    print("""
# Quick computation of all error metrics:
def quick_error_metrics(y_true, y_pred):
    error = y_pred - y_true
    return {
        'RMSE': np.sqrt(np.mean(error**2)),        # L² norm
        'MAE': np.mean(np.abs(error)),             # L¹ norm  
        'Max_Error': np.max(np.abs(error))         # L∞ norm
    }

# Using numpy.linalg.norm for all norms:
def error_metrics_with_linalg(y_true, y_pred):
    error = y_pred - y_true
    n = len(error)
    return {
        'RMSE': np.linalg.norm(error, ord=2) / np.sqrt(n),
        'MAE': np.linalg.norm(error, ord=1) / n,
        'Max_Error': np.linalg.norm(error, ord=np.inf)
    }
""")
