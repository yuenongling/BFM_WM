import sys
import os

from pandas.core.common import temp_setattr
from src.wall_model import WallModel
from wall_model_cases import TURB_CASES, TURB_CASES_TREE, print_dataset_tree, DATASET_PLOT_TITLE , STATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_name = sys.argv[1] if len(sys.argv) > 1 else None
model_path = os.path.join('./models', model_name) if model_name else None
wall_model = WallModel.load_compact(model_path, device="cpu")

training_datasets = ['CH','SYN','TBL_-4','TBL_-3','TBL_-2','TBL_-1','TBL_5','TBL_10','TBL_15','TBL_20', 'gaussian_2M_MAPG','gaussian_2M_FPG','gaussian_2M_APG','gaussian_2M_SEP', 'FS_ZPG','FS_FPG','FS_APG','bub_K']

test_datasets = ['CH',
 'SYN',
 'PIPE',
 'naca_0012',
 'hump',
 # 'hump_station_0','hump_station_1','hump_station_2','hump_station_3','hump_station_4','hump_station_5',
 'backstep',
 # 'backstep_station_0','backstep_station_1','backstep_station_2','backstep_station_3','backstep_station_4',
 'ph_B',
 # 'ph_B_station_0','ph_B_station_1','ph_B_station_2','ph_B_station_3',
 'bend',
 # 'bend_station_0','bend_station_1','bend_station_2','bend_station_3','bend_station_4','bend_station_5','bend_station_6',
 'convdiv',
 # 'convdiv_station_0','convdiv_station_1','convdiv_station_2','convdiv_station_3','convdiv_station_4',
 'round_step',
 'smoothramp',
 'TBL_-4',
 'TBL_-3',
 'TBL_-2',
 'TBL_-1',
 'TBL_5',
 'TBL_10',
 'TBL_15',
 # 'TBL_20',
 'apg_b1n',
 'apg_b2n',
 'apg_m13n',
 'apg_m16n',
 'apg_m18n',
 'aairfoil_10M',
 # 'aairfoil_10M_station_0','aairfoil_10M_station_1','aairfoil_10M_station_2','aairfoil_10M_station_3','aairfoil_10M_station_4','aairfoil_10M_station_5','aairfoil_10M_station_6','aairfoil_10M_station_7',
 'aairfoil_2M',
 # 'aairfoil_2M_station_0','aairfoil_2M_station_1','aairfoil_2M_station_2','aairfoil_2M_station_3','aairfoil_2M_station_4','aairfoil_2M_station_5',
 'gaussian_2M_MAPG',
 # 'gaussian_2M_MAPG_station_0','gaussian_2M_MAPG_station_1',
 'gaussian_2M_FPG',
 # 'gaussian_2M_FPG_station_0','gaussian_2M_FPG_station_1',
 'gaussian_2M_APG',
 # 'gaussian_2M_APG_station_0','gaussian_2M_APG_station_1',
 'gaussian_2M_SEP',
 'gaussian_1M_MAPG',
 'gaussian_1M_FPG_concave',
 'gaussian_1M_FPG_convex',
 'gaussian_1M_APG_stable',
 'gaussian_1M_APG',
 # 'gaussian_2M_SEP_station_0','gaussian_2M_SEP_station_1',
 'bub_A',
 # 'bub_A_station_0','bub_A_station_1','bub_A_station_2','bub_A_station_3','bub_A_station_4','bub_A_station_5',
 'bub_B',
 # 'bub_B_station_0','bub_B_station_1','bub_B_station_2','bub_B_station_3','bub_B_station_4','bub_B_station_5',
 'bub_C',
 'bub_K',
 'naca_4412_10',
 'naca_4412_4',
 'naca_4412_2',
 'naca_4412_1',
 'FS_ZPG',
 'FS_FPG',
 'FS_APG',
]

# NOTE: The following code is used to add the station datasets to the test_datasets list
for dataset in test_datasets:
    if dataset in STATION:
        dataset_index = test_datasets.index(dataset)
        len_station = len(STATION[dataset])
        for i in range(len_station):
            test_datasets.insert(dataset_index+i+1, dataset + '_station_' + str(i))

data = {}

BFM_error = []
log_error = []
abs_error = []

for test_dataset in test_datasets:
# Test each dataset
    print(f"\nTesting on {test_dataset}...")
    results = wall_model.test_external_dataset(
        tauw=True,
        dataset_key=test_dataset,
        mask_threshold=2e-4,
        save_path='./dummy/'
    )

    plt.close()

    if results is None:
        BFM_error.append(np.nan)
        log_error.append(np.nan)
        abs_error.append(False)
    else:
        BFM_err = results['metrics']['model']['mean_rel_error']
        log_err = results['metrics']['loglaw']['mean_rel_error']

        if BFM_err == 0 and log_err == 0: # It means that no relative error has been stored here
            print('!!!!!!!!!!! WARNING !!!!!!!!!!!')
            print("NOTE: Use absolute error instead of relative error")
            print('!!!!!!!!!!! WARNING !!!!!!!!!!!')

            BFM_err = results['metrics']['model']['mean_abs_error']
            log_err = results['metrics']['loglaw']['mean_abs_error']
            abs_error.append(True)
        else:
            abs_error.append(False)

        BFM_error.append(BFM_err)
        log_error.append(log_err)

data['Test Case'] = [DATASET_PLOT_TITLE[test_dataset] for test_dataset in test_datasets]
data['BFM'] = BFM_error
data['Log law'] = log_error
data['abs_error'] = abs_error

df = pd.DataFrame(data)

# # Save testing results to CSV
# Remove extension from model name
model_name = model_name.split('.')[0]
with open(f'./testing_results/{model_name}.csv', 'w') as f:
    pd.DataFrame(data).to_csv(f, index=False)

#########################################
# HTML Table Generation

def color_bars(val, max_val):
    """Generate HTML for a colored bar."""
    percentage = (val / max_val) * 100
    color = f'hsl({100 - percentage}, 100%, 50%)' # Green to red
    return f'<div style="background:linear-gradient(to right, {color} {percentage}%, white {percentage}%); padding: 5px; border-radius: 3px;">{val:.3e}</div>'

def checkmark(is_true):
    return '&#10004;' if is_true else ''

def highlight_trained(row):
    if row['Test Case'] in [DATASET_PLOT_TITLE[dataset] for dataset in training_datasets]:
        return ['background-color: rgba(255, 255, 0, 0.3)'] * len(row)  # Light yellow, transparent background
    else:
        return [''] * len(row)

max_val = max(df[['BFM', 'Log law']].max())

styled = (
    df.style
    .apply(highlight_trained, axis=1)
    .format({'BFM': lambda x: color_bars(x, max_val), 'Log law': lambda x: color_bars(x, max_val), 'abs_error': checkmark})
    .set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('font-size', '18px'),
                ('padding', '12px'),
                ('background-color', '#f0f0f0'),
                ('text-align', 'center')
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('font-size', '18px'),
                ('padding', '12px'),
                ('text-align', 'center')
            ]
        },
        {
            'selector': 'table',
            'props': [
                ('border-collapse', 'collapse'),
                ('box-shadow', '0 4px 8px rgba(0, 0, 0, 0.1)'),
                ('border-radius', '8px'),
                ('overflow', 'hidden')
            ]
        },
        {
            'selector': 'th, td',
            'props': [('border', '1px solid #ddd')]
        }
    ])
    .set_properties(**{'text-align': 'center'})
)

html_code = styled.to_html(escape=False)

with open(f'./tables/{model_name}_test.html', 'w') as f:
    f.write(html_code)

print("Saved 'fancy_gradient_table.html' with colored bars and improved styling.")
