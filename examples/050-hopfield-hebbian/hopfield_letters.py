
import numpy as np
import matplotlib.pyplot as plt
# Make sure hopfield_basic.py is in the same directory
from hopfield_basic import initialize_network, store_multiple_patterns, recall_pattern

def create_letter_patterns():
    """
    Create pixelated letter patterns for Hopfield network training.
    """
    letter_A = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]])
    letter_E = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    letter_I = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
    letters = {'A': letter_A, 'E': letter_E, 'I': letter_I}
    # Convert to bipolar and flatten
    for name, pattern in letters.items():
        letters[name] = (2 * pattern - 1).flatten()
    return letters

def add_noise_to_pattern(pattern, noise_level=0.2):
    """
    Add random noise to a pattern by flipping pixels.
    """
    noisy_pattern = pattern.copy()
    num_pixels_to_corrupt = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), size=num_pixels_to_corrupt, replace=False)
    noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]
    return noisy_pattern

def test_letter_recognition(network, letters, test_scenarios):
    """
    Test the letter recognition capabilities of the network.
    """
    print("=" * 60)
    print("LETTER RECOGNITION TESTING ")
    print("=" * 60)

    successful_recognitions = 0
    for i, scenario in enumerate(test_scenarios):
        print(f"\nTest {i + 1}/{len(test_scenarios)}: {scenario['description']}")

        original_pattern = letters[scenario['letter']]
        noisy_pattern = add_noise_to_pattern(original_pattern, noise_level=scenario['noise_level'])
        recalled_pattern, _, _ = recall_pattern(network, noisy_pattern)

        # Determine which letter was recognized
        recognized_letter = None
        best_match_score = -1
        for name, stored_pattern in letters.items():
            similarity = np.mean(recalled_pattern == stored_pattern)
            if similarity > best_match_score:
                best_match_score = similarity
                if similarity > 0.9: # Recognition threshold
                    recognized_letter = name

        success = (recognized_letter == scenario['letter'])
        if success:
            successful_recognitions += 1

        print(f"Recognized as: {recognized_letter or 'UNKNOWN'}")
        print(f"Success: {'✓' if success else '✗'}")

    print(f"\nTotal success: {successful_recognitions}/{len(test_scenarios)} ({100*successful_recognitions/len(test_scenarios):.1f}%)")

def main_letters():
    """
    Main function for letter recognition demo.
    """
    np.random.seed(42)
    letters = create_letter_patterns()

    # Train network
    network = initialize_network(num_neurons=25)
    store_multiple_patterns(network, list(letters.values()))

    # Define and run test scenarios
    test_scenarios = [
        {'letter': 'A', 'noise_level': 0.2, 'description': "Letter A with 20% noise"},
        {'letter': 'E', 'noise_level': 0.2, 'description': "Letter E with 20% noise"},
        {'letter': 'I', 'noise_level': 0.3, 'description': "Letter I with 30% noise"},
    ]

    test_letter_recognition(network, letters, test_scenarios)

if __name__ == '__main__':
    main_letters()

