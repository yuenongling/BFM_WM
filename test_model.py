from proposal.wall_model import WallModel
from wall_model_cases import TURB_CASES, TURB_CASES_TREE, print_dataset_tree

def load_and_test_model(model_path, test_dataset=None):
    wall_model = WallModel.load_compact(model_path, device="cpu")

    # Test each dataset
    print(f"\nTesting on {test_dataset}...")
    results = wall_model.test_external_dataset(
        dataset_key=test_dataset,
        tauw=True,
        mask_threshold=2e-4,
        save_path=None,
    )

    return results, wall_model


# Outer code handling dataset input
valid_datasets = TURB_CASES
model_path = './models/NN_wm_CH1_G0_S0_tn210_vn53_fds0_lds0_inputs2/NN_wm_CH1_G0_S0_tn210_vn53_fds0_lds0_inputs2_final_ep1000_tl0.00131264_vl0.00142734.pth'

test_dataset = None
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
        results, wall_model = load_and_test_model(model_path, test_dataset)

