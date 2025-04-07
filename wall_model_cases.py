# NOTE: High-fidelity data: case names and corresponding input files
TURB_CASES = [
              # High Reynolds number ZPG cases
              'CH', 'SYN', 'PIPE',
              # TBL cases
              'apg', 'bub', 'TBL',
              'naca_0012', 'naca_0025', 'naca_4412', 'aairfoil'
              'apg_kth', 'gaussian_2M', 'curve']

# NOTE: File extension for the input files
EXT = 'h5'

# NOTE: TBL has different ramp angles; will be updated by TBL
INPUT_TURB_FILES = dict(zip(TURB_CASES, [f'./data/{case}_data.{EXT}' for case in TURB_CASES]))
DATASET_PLOT_TITLE = dict(zip(TURB_CASES, [r'Channel flow $Re_{\tau}:550-4200$', 'Synthetic', 'Pipe', 'APG', 'Bub', 'TBL', 'NACA0012 Re=400K', 'NACA4412', 'NACA0025', 'NACA4412', 'Aairfoil', 'APG_KTH', 'Gaussian 2M', 'Curve']))

# NOTE: Our TBL data
TBL_ANGLES = [-4,-3,-2,-1,5,10,15,20]
for angle in TBL_ANGLES:
    INPUT_TURB_FILES[f'TBL_{angle}'] = f'./data/TBL_{angle}_data.{EXT}'
    TURB_CASES.append(f'TBL_{angle}')
    DATASET_PLOT_TITLE[f'{angle}'] = f'Turbulent Boundary Layer, Ramp Angle: {angle}°'
# NOTE: APG data 
SUBCASES = ['b1n', 'b2n', 'm13n', 'm16n', 'm18n']
for subcase in SUBCASES:
    INPUT_TURB_FILES[f'apg_{subcase}'] = f'./data/apg_{subcase}_data.{EXT}'
    TURB_CASES.append(f'apg_{subcase}')
    DATASET_PLOT_TITLE[f'apg_{subcase}'] = f'Mild Adverse Pressure Gradient, Subcase: {subcase}'
# NOTE: aairfoil with different Reynolds numbers
RE = [2, 10]
for re in RE:
    INPUT_TURB_FILES[f'aairfoil_{re}M'] = f'./data/aairfoil_{re}M_data.{EXT}'
    TURB_CASES.append(f'aairfoil_{re}M')
    DATASET_PLOT_TITLE[f'aairfoil_{re}M'] = f'A-Airfoil, Reynolds number: {re}M'
# NOTE: Gaussian data
RE = [1, 2]
POS = ['MAPG', 'FPG', 'APG', 'SEP', 'concave', 'convex', 'FPG_concave', 'FPG_convex']
for re in RE:
    INPUT_TURB_FILES[f'gaussian_{re}M'] = f'./data/gaussian_{re}M_data.{EXT}'
    TURB_CASES.append(f'gaussian_{re}M')
    DATASET_PLOT_TITLE[f'gaussian_{re}M'] = f'Gaussian Bump: {re}M'
    for pos in POS:
        INPUT_TURB_FILES[f'gaussian_{re}M_{pos}'] = f'./data/gaussian_{re}M_data_{pos}.{EXT}'
        TURB_CASES.append(f'gaussian_{re}M_{pos}')
        DATASET_PLOT_TITLE[f'gaussian_{re}M_{pos}'] = f'Gaussian Bump: {re}M {pos}'
# NOTE: aairfoil with different Reynolds numbers
BUB_SUB = ['A', 'B', 'C', 'K']
for sub in BUB_SUB:
    INPUT_TURB_FILES[f'bub_{sub}'] = f'./data/bub_{sub}_data.{EXT}'
    TURB_CASES.append(f'bub_{sub}')
    DATASET_PLOT_TITLE[f'bub_{sub}'] = f'Pressure-induced Separation bubble, Subcase: {sub}'
# NOTE: naca4412 with different Reynolds numbers
NACA_SUB = ['10', '4', '2', '1']
for sub in NACA_SUB:
    INPUT_TURB_FILES[f'naca_4412_{sub}'] = f'./data/naca_4412_top{sub}n_data.{EXT}'
    TURB_CASES.append(f'naca_4412_{sub}')
    DATASET_PLOT_TITLE[f'naca_4412_{sub}'] = f'NACA4412 Re={sub}00000'
INPUT_TURB_FILES[f'naca_0012'] = f'./data/naca_0012_top4n12_data.{EXT}'
INPUT_TURB_FILES[f'naca_0025'] = f'./data/naca_0025_data.{EXT}'
DATASET_PLOT_TITLE[f'naca_0025'] = f'NACA0025'

# NOTE: Falkner-Skan laminar boundary layer
LBL_CASES = ['ZPG','FPG','APG']
for case in LBL_CASES:
    INPUT_TURB_FILES[f'FS_{case}'] = f'./data/falkner_skan_data_{case}.{EXT}'
    TURB_CASES.append(f'FS_{case}')
    DATASET_PLOT_TITLE[f'FS_{case}'] = f'Laminar Boundary Layer: {case}'

# NOTE: Cases with curvature from Applebaum et al. (2025)
CURVE_CASES = ['pg']
for case in CURVE_CASES:
    INPUT_TURB_FILES[f'curve_{case}'] = f'./data/curve_{case}_data.{EXT}'
    TURB_CASES.append(f'curve_{case}')
    DATASET_PLOT_TITLE[f'curve_{case}'] = f'Boundary layer with curvature: {case}'

# NOTE: Reynolds numbers to cover for CH
RE_NUMS = [550, 950, 2000, 4200]


def print_dataset_tree(datasets, indent=0):
    for key, value in datasets.items():
        print('    ' * indent + f"├── {key}")
        if isinstance(value, dict) and value:  # Only recurse if the value is a non-empty dictionary
            print_dataset_tree(value, indent + 1)


            # Dataset dictionary representing a tree-like structure
TURB_CASES_TREE = {
    "CH": {},
    "SYN": {},
    "PIPE": {},
    "apg": {
        "apg_b1n": {},
        "apg_b2n": {},
        "apg_m13n": {},
        "apg_m16n": {},
        "apg_m18n": {}
    },
    "bub": {
        "bub_A": {},
        "bub_B": {},
        "bub_C": {},
        "bub_K": {}
    },
    "TBL": {
        "TBL_-4": {},
        "TBL_-3": {},
        "TBL_-2": {},
        "TBL_-1": {},
        "TBL_5": {},
        "TBL_10": {},
        "TBL_15": {},
        "TBL_20": {}
    },
    "naca_0012": {},
    "naca_0025": {},
    "naca_4412": {
        "naca_4412_10": {},
        "naca_4412_4": {},
        "naca_4412_2": {},
        "naca_4412_1": {}
    },
    "aairfoilapg_kth": {},
    "gaussian_2M": {},
    "curve": {},
    "aairfoil_2M": {},
    "aairfoil_10M": {},
    "gaussian_1M": {
        "gaussian_1M_MAPG": {},
        "gaussian_1M_FPG": {},
        "gaussian_1M_APG": {},
        "gaussian_1M_SEP": {},
        "gaussian_1M_concave": {},
        "gaussian_1M_convex": {},
        "gaussian_1M_FPG_concave": {},
        "gaussian_1M_FPG_convex": {}
    },
    "gaussian_2M": {
        "gaussian_2M_MAPG": {},
        "gaussian_2M_FPG": {},
        "gaussian_2M_APG": {},
        "gaussian_2M_SEP": {},
        "gaussian_2M_concave": {},
        "gaussian_2M_convex": {},
        "gaussian_2M_FPG_concave": {},
        "gaussian_2M_FPG_convex": {}
    },
    "FS_ZPG": {},
    "FS_FPG": {},
    "FS_APG": {},
    "curve_pg": {}
}

