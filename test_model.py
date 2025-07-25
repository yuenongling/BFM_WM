'''
Script to test wall model on different cases interactively

Models should be saved in ./models/

Example:
    python test_model.py wall_model.pth 0 # Not save the plots
    python test_model.py wall_model.pth 1 # Save the plots
'''
import sys
import os
from src.wall_model import WallModel
from wall_model_cases import TURB_CASES, TURB_CASES_TREE, print_dataset_tree
import Levenshtein # This is to find similar dataset names

# --- Read in command line arguments ---
model_path = sys.argv[1] if len(sys.argv) > 1 else None
save_mode = int(sys.argv[2]) if len(sys.argv) > 2 else False
masked_threshold  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0002
#

def get_closest_matches(invalid_key, valid_keys, num_suggestions=10):
    """
    Finds the closest matching keys from a list of valid keys
    based on Levenshtein distance.
    """
    distances = []
    for key in valid_keys:
        distance = Levenshtein.distance(invalid_key, key)
        distances.append((distance, key))

    # Sort by distance and return the top N
    distances.sort()
    return [key for distance, key in distances[:num_suggestions]]

def load_and_test_model(model_path, test_dataset=None, wall_model=None):

    if wall_model is None:
        wall_model = WallModel.load_compact(model_path, device="cpu")

    # Test each dataset
    print(f"\nTesting on {test_dataset}...")

    save_path = "./results/" if save_mode else None

    results = wall_model.test_external_dataset(
        dataset_key=test_dataset,
        tauw=True,
        mask_threshold=masked_threshold,
        save_path=save_path,
        LogTransform=wall_model.config.get('training', {}).get('LogTransform', False),
    )

    return results, wall_model


# Outer code handling dataset input
valid_datasets = TURB_CASES

model_path = os.path.join('./models', model_path) if model_path else None

test_dataset = None
wall_model   = None

while True:
    test_dataset = input(f"Enter the dataset to test (P for printing all): ")
    
    if test_dataset == 'P':
# Call the function to print the dataset tree
        print_dataset_tree(TURB_CASES_TREE)
        continue
    elif test_dataset == 'Q':
        print("Exiting the program.")
        break
    elif test_dataset not in valid_datasets:
        print(f"Invalid dataset '{test_dataset}'.")
        closest_matches = get_closest_matches(test_dataset, valid_datasets)
        print("Did you mean one of these?")
        for match in closest_matches:
            print(f"- {match}")
        continue # Let the user try againe
    else:
        # Valid dataset entered, test the model
        if wall_model is None:
            results, wall_model = load_and_test_model(model_path, test_dataset)
        else:
            results, _ = load_and_test_model(model_path, test_dataset, wall_model=wall_model)

        # if results is not None:
        #     wall_model.test_external_dataset(
        #         dataset_key=test_dataset,
        #         purpose = 1
        #     )

        print(results)
