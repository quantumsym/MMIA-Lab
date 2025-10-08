
import numpy as np
from sklearn.datasets import fetch_openml
import warnings
# Make sure hopfield_basic.py is in the same directory
from hopfield_basic import initialize_network, store_multiple_patterns, recall_pattern

warnings.filterwarnings('ignore')

def downsample_image(image, target_size=(8, 8)):
    """
    Downsample an image to target size using averaging.
    """
    original_h, original_w = image.shape
    target_h, target_w = target_size
    block_h, block_w = original_h // target_h, original_w // target_w
    downsampled = np.zeros(target_size)
    for i in range(target_h):
        for j in range(target_w):
            block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            downsampled[i, j] = np.mean(block)
    return downsampled

def load_and_preprocess_mnist(target_digits=[0, 1, 2]):
    """
    Load and preprocess MNIST dataset for Hopfield network training.
    """
    print("Loading MNIST dataset...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
    except Exception as e:
        print(f"Error loading MNIST: {e}. Aborting.")
        return None

    processed_patterns = {}
    for digit in target_digits:
        # Get first 5 samples for each digit
        digit_images = X[y == digit][:5]
        patterns = []
        for img in digit_images:
            img_2d = img.reshape(28, 28)
            downsampled = downsample_image(img_2d, target_size=(8, 8))
            binary = (downsampled > 128).astype(int)
            bipolar = 2 * binary - 1
            patterns.append(bipolar.flatten())
        processed_patterns[digit] = np.array(patterns)
    return processed_patterns

def train_hopfield_on_digits(processed_patterns, max_patterns_per_digit=2):
    """
    Train a Hopfield network on digit patterns.
    """
    print("Training Hopfield network on digit patterns...")
    pattern_size = list(processed_patterns.values())[0].shape[1]
    network = initialize_network(num_neurons=pattern_size)

    patterns_to_store = []
    stored_patterns_info = {'patterns_by_digit': {}}
    for digit, patterns in processed_patterns.items():
        selected_patterns = patterns[:min(len(patterns), max_patterns_per_digit)]
        patterns_to_store.extend(selected_patterns)
        stored_patterns_info['patterns_by_digit'][digit] = selected_patterns

    store_multiple_patterns(network, patterns_to_store)
    return network, stored_patterns_info

def main_mnist():
    """
    Main function for MNIST digit recognition demo .
    """
    np.random.seed(42)
    # Choose digits that are visually distinct
    target_digits = [0, 1, 8]

    processed_patterns = load_and_preprocess_mnist(target_digits=target_digits)
    if processed_patterns is None:
        return

    network, stored_patterns_info = train_hopfield_on_digits(processed_patterns, max_patterns_per_digit=2)

    # Test recall for one digit with noise
    test_digit = target_digits[0]
    original_pattern = processed_patterns[test_digit][0]

    # Add noise
    noisy_pattern = original_pattern.copy()
    num_flips = int(0.2 * len(noisy_pattern))
    flip_indices = np.random.choice(len(noisy_pattern), size=num_flips, replace=False)
    noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]

    print(f"\nTesting recall for digit {test_digit} with 20% noise...")
    recalled_pattern, _, _ = recall_pattern(network, noisy_pattern)

    # Check which digit was recalled
    recognized_digit = None
    best_match_score = -1
    for digit, patterns in stored_patterns_info['patterns_by_digit'].items():
        for p in patterns:
            similarity = np.mean(recalled_pattern == p)
            if similarity > best_match_score:
                best_match_score = similarity
                if similarity > 0.9: # Recognition threshold
                    recognized_digit = digit

    print(f"Recognized digit: {recognized_digit or 'UNKNOWN'}")
    print(f"Success: {'✓' if recognized_digit == test_digit else '✗'}")

if __name__ == '__main__':
    main_mnist()

