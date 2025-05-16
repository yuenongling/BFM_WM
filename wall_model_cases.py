'''
This is the file where we define the cases we want to use for training or testing the model.

Standard way to introduce a new case:
# NOTE: New case
# Replace 
# 1. *Case_name* with the name of the case
# 2. *./data/datafile.h5* with the path to the data file
# 3. *Title for the case* with the title of the case
INPUT_TURB_FILES[f'Case_name'] = f'./data/datafile.h5']
TURB_CASES.append(f'Case_name')
DATASET_PLOT_TITLE[f'Case_name'] = f'Title for the case'
'''

# NOTE: High-fidelity data: case names and corresponding input files
TURB_CASES = [
              # High Reynolds number ZPG cases
              'CH', 'SYN', 'PIPE',
              # TBL cases
              'naca_0012', 'naca_0025',
              'backstep', 'axissym_BL', 'ph_B',  'bend', 'convdiv',
              'hump', 'smoothramp', 'round_step']

# NOTE: File extension for the input files
EXT = 'h5'

# NOTE: TBL has different ramp angles; will be updated by TBL
INPUT_TURB_FILES = dict(zip(TURB_CASES, [f'./data/{case}_data.{EXT}' for case in TURB_CASES]))
DATASET_PLOT_TITLE = dict(zip(TURB_CASES, [r'Channel flow $Re_{\tau}:550-4200$', 'Synthetic (log law) data (up to $Re_{\tau}=100,000$)', 'Pipe (up to $Re_{\tau}=12,000$)', 'NACA0012 Re=400K', 'NACA0025', 
                                        '2D Backward Step (Driver et al. 1985)', 'Axissymetric Boundary Layer (Driver et al. 1991)', 'Periodic Hill (Balakumar 2015)', 'Bended Boundary Layer (Smits et al. 1979)', 'Convergent and Divergent Channel (Laval et al. 2009)',
                                        'NASA Hump (Uzun et al. 2017)', 'Smooth Ramp (Uzun et al. 2024)', 'Round Step DLR']))

# NOTE: Psuedo-Ekman by Spalart 1988
casename = 'strained_TBL'
Z_LIST = [-150, -100, 0]
for z in Z_LIST:
    INPUT_TURB_FILES[f'{casename}_z{z}'] = f'./data/{casename}_z{z}_data.{EXT}'
    TURB_CASES.append(f'{casename}_z{z}')
    DATASET_PLOT_TITLE[f'{casename}_z{z}'] = f'{casename} z={z}'

# NOTE: Psuedo-Ekman by Spalart 1988
INPUT_TURB_FILES[f'Ekman'] = f'./data/Ekman_data.{EXT}'
TURB_CASES.append(f'Ekman')
DATASET_PLOT_TITLE[f'Ekman'] = f'Pseudo-Ekman by Spalart (1988)'

# NOTE: Concave TBL by Johnson et al. (1988)
INPUT_TURB_FILES[f'Concave_J'] = f'./data/Concave_Johnson_data.{EXT}'
TURB_CASES.append(f'Concave_J')
DATASET_PLOT_TITLE[f'Concave_J'] = f'Concave TBL by Johnson et al. (1988)'

# NOTE: APG TBL by Gasser et al.
INPUT_TURB_FILES[f'APG_Gasser'] = f'./data/APG_Gasser_data.{EXT}'
TURB_CASES.append(f'APG_Gasser')
DATASET_PLOT_TITLE[f'APG_Gasser'] = f'APG Gasser et al.'

# NOTE: Backstep step with 6 degree expansion angle
INPUT_TURB_FILES[f'backstep_inc'] = f'./data/backstep_inclined_data.{EXT}'
TURB_CASES.append(f'backstep_inc')
DATASET_PLOT_TITLE[f'backstep_inc'] = f'2D Backward Step with 6 degree inclination (Driver et al. 1985)'

# NOTE: Swept wing (35 degree yaw) data
INPUT_TURB_FILES[f'swept_wing'] = f'./data/swept_wing_data.h5'
TURB_CASES.append(f'swept_wing')
DATASET_PLOT_TITLE[f'swept_wing'] = f'Swept wing (Van den Berg et al. 1975)'

# NOTE: Duct
INPUT_TURB_FILES[f'duct'] = f'./data/DUCT_data.h5'
TURB_CASES.append(f'duct')
DATASET_PLOT_TITLE[f'duct'] = f'Duct (Pirozzo et al. 2018)'

# NOTE: diffuser
INPUT_TURB_FILES[f'diffuser'] = f'./data/diffuser_data.h5'
TURB_CASES.append(f'diffuser')
DATASET_PLOT_TITLE[f'diffuser'] = f'Cherry Diffuser (Ohlsson et al. 2010)'

# NOTE: Channel with rotation
INPUT_TURB_FILES[f'CH_rot'] = f'./data/CH_rot_data.h5'
TURB_CASES.append(f'CH_rot')
DATASET_PLOT_TITLE[f'CH_rot'] = f'Channel with Rotation'

# NOTE: Our TBL data
TBL_ANGLES = [-4,-3,-2,-1,5,10,15,20]
for angle in TBL_ANGLES:
    INPUT_TURB_FILES[f'TBL_{angle}'] = f'./data/TBL_{angle}_data.{EXT}'
    TURB_CASES.append(f'TBL_{angle}')
    DATASET_PLOT_TITLE[f'{angle}'] = f'Turbulent Boundary Layer, Ramp Angle: {angle}°'
    DATASET_PLOT_TITLE[f'TBL_{angle}'] = f'Turbulent Boundary Layer, Ramp Angle: {angle}°'
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

# NOTE: Gaussian data (2M)
RE = 2
POS = ['MAPG', 'FPG', 'APG', 'SEP']
INPUT_TURB_FILES[f'gaussian_{RE}M'] = f'./data/gaussian_{RE}M_data.{EXT}'
TURB_CASES.append(f'gaussian_{RE}M')
DATASET_PLOT_TITLE[f'gaussian_{RE}M'] = f'Gaussian Bump: {RE}M'
for pos in POS:
    INPUT_TURB_FILES[f'gaussian_{RE}M_{pos}'] = f'./data/gaussian_{RE}M_data_{pos}.{EXT}'
    TURB_CASES.append(f'gaussian_{RE}M_{pos}')
    DATASET_PLOT_TITLE[f'gaussian_{RE}M_{pos}'] = f'Gaussian Bump: {RE}M {pos}'
# NOTE: Gaussian data (1M)
RE = 1
POS = ['MAPG', 'FPG_concave', 'FPG_convex', 'APG_stable', 'APG', 'SEP']
INPUT_TURB_FILES[f'gaussian_{RE}M'] = f'./data/gaussian_{RE}M_data.{EXT}'
TURB_CASES.append(f'gaussian_{RE}M')
DATASET_PLOT_TITLE[f'gaussian_{RE}M'] = f'Gaussian Bump: {RE}M'
for pos in POS:
    INPUT_TURB_FILES[f'gaussian_{RE}M_{pos}'] = f'./data/gaussian_{RE}M_data_{pos}.{EXT}'
    TURB_CASES.append(f'gaussian_{RE}M_{pos}')
    DATASET_PLOT_TITLE[f'gaussian_{RE}M_{pos}'] = f'Gaussian Bump: {RE}M {pos}'

# NOTE: Different bubble cases
BUB_SUB = ['A', 'B', 'C', 'K']
for sub in BUB_SUB:
    INPUT_TURB_FILES[f'bub_{sub}'] = f'./data/bub_{sub}_data.{EXT}'
    TURB_CASES.append(f'bub_{sub}')
    DATASET_PLOT_TITLE[f'bub_{sub}'] = f'Pressure-induced Separation bubble, Subcase: {sub}'
# NOTE: Bubble with swept
sub = 'C35'
INPUT_TURB_FILES[f'bub_{sub}'] = f'./data/bub_{sub}_data.{EXT}'
TURB_CASES.append(f'bub_{sub}')
DATASET_PLOT_TITLE[f'bub_{sub}'] = f'Pressure-induced Separation bubble with swept, Subcase: {sub}'

# NOTE: naca4412 with different Reynolds numbers
NACA_SUB = ['10', '4', '2', '1']
for sub in NACA_SUB:
    INPUT_TURB_FILES[f'naca_4412_{sub}'] = f'./data/naca_4412_top{sub}n_data.{EXT}'
    TURB_CASES.append(f'naca_4412_{sub}')
    DATASET_PLOT_TITLE[f'naca_4412_{sub}'] = f'NACA4412 Re={sub}00000'
# NOTE: naca0012 (turbulent, KTH) 
INPUT_TURB_FILES[f'naca_0012'] = f'./data/naca_0012_top4n12_data.{EXT}'

# NOTE: NACA0025 from Konrad (private communication)
INPUT_TURB_FILES[f'naca_0025'] = f'./data/naca0025_Konrad_data.{EXT}'
DATASET_PLOT_TITLE[f'naca_0025'] = f'NACA0025'
REGION = ['Laminar', 'Transition', 'Turbulent']
for rg in REGION:
    INPUT_TURB_FILES[f'naca_0025_{rg}'] = f'./data/naca0025_{rg}_Konrad_data.{EXT}'
    TURB_CASES.append(f'naca_0025_{rg}')
    DATASET_PLOT_TITLE[f'naca_0025_{rg}'] = f'NACA_0025: {rg} region'

# NOTE: Falkner-Skan laminar boundary layer
LBL_CASES = ['ZPG','FPG','APG', 'ALL']
for case in LBL_CASES:
    INPUT_TURB_FILES[f'FS_{case}'] = f'./data/FS_{case}_data.{EXT}'
    TURB_CASES.append(f'FS_{case}')
    DATASET_PLOT_TITLE[f'FS_{case}'] = f'Laminar Boundary Layer: {case}'

# NOTE: Stokes Second 
INPUT_TURB_FILES[f'Stokes2'] = f'./data/Stokes2_data.{EXT}'
TURB_CASES.append(f'Stokes2')
DATASET_PLOT_TITLE[f'Stokes2'] = f'Stokes Second Problem'

# NOTE: Oscillating pipe
INPUT_TURB_FILES[f'OscPipe'] = f'./data/OscPipe_data.{EXT}'
TURB_CASES.append(f'OscPipe')
DATASET_PLOT_TITLE[f'OscPipe'] = f'Oscillating Pipe'

# NOTE: Transition ZPG boundary layer from JHU
INPUT_TURB_FILES[f'Transition'] = f'./data/transition_JHU_data.{EXT}'
TURB_CASES.append(f'Transition')
DATASET_PLOT_TITLE[f'Transition'] = f'Transition ZPG Boundary Layer'
# NOTE: Transition ZPG boundary layer from JHU
REGION = ['laminar', 'transition', 'turbulent']
for rg in REGION:
    INPUT_TURB_FILES[f'Transition_{rg}'] = f'./data/transition_JHU_{rg}_data.{EXT}'
    TURB_CASES.append(f'Transition_{rg}')
    DATASET_PLOT_TITLE[f'Transition_{rg}'] = f'Transition ZPG Boundary Layer: {rg} region'

# NOTE: NACA0012 laminar (Rec=5000)
INPUT_TURB_FILES[  f'naca0012_laminar'] = f'./data/naca0012_laminar_data.{EXT}'
TURB_CASES.append( f'naca0012_laminar')
DATASET_PLOT_TITLE[f'naca0012_laminar'] = f'Laminar NACA0012 (Rec=5000)'

# NOTE: Cases with curvature from Applebaum et al. (2025)
INPUT_TURB_FILES[f'curved_TBL'] = f'./data/curved_TBL_data.{EXT}'
TURB_CASES.append(f'curved_TBL')
DATASET_PLOT_TITLE[f'curved_TBL'] = f'Boundary layer with curvature'
REGION = ['APG', 'FPG']
for r in REGION:
    INPUT_TURB_FILES[f'curved_TBL_{r}'] = f'./data/curved_TBL_{r}_data.{EXT}'
    TURB_CASES.append(f'curved_TBL_{r}')
    DATASET_PLOT_TITLE[f'curved_TBL_{r}'] = f'Boundary layer with curvature: {r}'

# NOTE: Periodic Hill from Xiao et al. (2020)
PH_CASES = ['X']
for case in PH_CASES:
    INPUT_TURB_FILES[f'ph_{case}'] = f'./data/ph_{case}_data.{EXT}'
    TURB_CASES.append(f'ph_{case}')
    DATASET_PLOT_TITLE[f'ph_{case}'] = f'Periodic Hill: {case}'

# NOTE: ph_G from Gloerfelt et al. (2019)
RE = [2800, 10595, 19000, 37000]
for re in RE:
    INPUT_TURB_FILES[f'ph_G_{re}'] = f'./data/ph_G_{re}_data.{EXT}'
    TURB_CASES.append(f'ph_G_{re}')
    DATASET_PLOT_TITLE[f'ph_G_{re}'] = f'Periodic Hill by Gloerfelt: {case}'


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
    "curve_pg": {},
    "bend": {},
    "convdiv": {},
    "backstep": {},
    "ph_B": {},
}
###############################
# NOTE: Go over all cases and add a "stencil" version 
#       Currently, we only support a few cases
#       1. TBL
#       2. apg by KTH
#       3. Channel and Pipe
TURB_CASES_WITH_STENCIL = ['TBL_5', 'CH', 'PIPE', 'apg_b1n', 'apg_b2n', 'apg_m13n', 'apg_m16n', 'apg_m18n', 'TBL_-4', 'TBL_-3', 'TBL_-2', 'TBL_-1', 'TBL_5', 'TBL_10', 'TBL_15', 'TBL_20']
for case in TURB_CASES_WITH_STENCIL:
    TURB_CASES += [f'{case}_stencil']
    INPUT_TURB_FILES[f'{case}_stencil'] = f'./data/stencil/{case}_data_stencils.{EXT}'

#################
# NOTE: Here we also define whether for each cases, we have extra cases focusing on certain cases
#
STATION = {
        "bend": [-0.193, 0.02499936, 0.17799969, 0.3300006 , 0.48299949, 0.9400002 , 1.245], # The station right after 30 degree bend
        "convdiv": [0, 2.0203, 4.0488, 5.07059, 5.27287, 5.50133, 5.60819, 6.63815, 7.12181, 7.46414, 8.05638],
        "backstep": [-4.,  1.,  4.,  6., 10.],
        "ph_B": [1.1475, 4.9719, 5.827861, 6.052468, 6.373975, 6.775461, 8.002593, 8.354093],
        "gaussian_2M_MAPG": [-0.59959855, -0.40889207],
        "gaussian_2M_FPG": [-0.2, -0.15, -0.1, ],
        "gaussian_2M_APG": [2.84180234e-05, 4.99274804e-02],
        "gaussian_2M_SEP": [0.2, 0.3597485],
        "gaussian_1M_MAPG": [-0.59999815,-0.40040712],
        "gaussian_1M_FPG_concave": [-0.22034062, -0.16039],
        "gaussian_1M_FPG_convex": [-0.10079176,  -0.04998937, -0.00222268],
        "gaussian_1M_APG_stable": [0.00998682,0.02995354,0.04391049],
        "gaussian_1M_APG": [0.08037356, 0.10057804,0.17514278],
        "hump": [-0.8, -0.4, 0, 0.10116047, 0.2014175, 0.30294264, 0.50484454, 0.64700406, 0.66666904, 0.77782227, 0.78822388, 0.79862236, 0.90906449, 1.00000186, 1.2, 1.4],
        "bub_A": [-5.02708292, -2.9958334, -1.1677103,0.0510397,0.25417042,0.3015604],
        "bub_B": [-5.1187501, -2.07188034,-1.1984396,0.32499981,1.99061966,3.00625038],
        "aairfoil_10M": [0.05 , 0.08 , 0.1  , 0.15 , 0.2  , 0.5  , 0.825, 0.93, 0.99],
        "aairfoil_2M": [0.030006787, 0.08063563, 0.10109831 , 0.15148925, 0.30277907, 0.5055756 ,0.82811433, 0.93907689, 0.94093326],
        "smoothramp": [-0.72, -0.2, 0, 0.3, 1.5, 2, 2.5, 3]
}

# NOTE: Add subcases for each station
for case in STATION:
    for i, station in enumerate(STATION[case]):
        # TURB_CASES_TREE[case][f'station_{i}'] = {}
        TURB_CASES.append(f'{case}_station_{i}')
        DATASET_PLOT_TITLE[f'{case}_station_{i}'] = f'{DATASET_PLOT_TITLE[case]}, Station: {station}'

