import sys
import os
from src.wall_model import WallModel
from wall_model_cases import TURB_CASES, TURB_CASES_TREE, print_dataset_tree

def load_and_test_model(model_path, test_dataset=None, wall_model=None):

    if wall_model is None:
        wall_model = WallModel.load_compact(model_path, device="cpu")

    # Test each dataset
    print(f"\nTesting on {test_dataset}...")
    results = wall_model.test_external_dataset(
        dataset_key=test_dataset,
        tauw=True,
        mask_threshold=6e-4,
        save_path=None,
        LogTransform=wall_model.config.get('training', {}).get('LogTransform', False),
    )

    return results, wall_model


# Outer code handling dataset input
valid_datasets = TURB_CASES

model_path = sys.argv[1] if len(sys.argv) > 1 else None
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
        print(f"Invalid dataset '{test_dataset}'. Please choose from {valid_datasets}.")
        continue
    else:
        # Valid dataset entered, test the model
        if wall_model is None:
            results, wall_model = load_and_test_model(model_path, test_dataset)
        else:
            results, _ = load_and_test_model(model_path, test_dataset, wall_model=wall_model)

        if results is not None:
            wall_model.test_external_dataset(
                dataset_key=test_dataset,
                purpose = 1
            )

        print(results)
