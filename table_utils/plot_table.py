'''
Read in testing results and generate a table with colored bars
'''

import sys
from wall_model_cases import DATASET_PLOT_TITLE 
import numpy as np
import pandas as pd

training_datasets = ['CH','SYN','TBL_-4','TBL_-3','TBL_-2','TBL_-1','TBL_5','TBL_10','TBL_15','TBL_20', 'gaussian_2M_MAPG','gaussian_2M_FPG','gaussian_2M_APG','gaussian_2M_SEP', 'FS_ZPG','FS_FPG','FS_APG','bub_K']
# # Save testing results to CSV
# Remove extension from model name

model_name = sys.argv[1] if len(sys.argv) > 1 else None
df = pd.read_csv(f'./testing_results/renamed/{model_name}.csv')

df['diff'] = df['BFM'] - df['Log law']

df['Training?'] = [dataset in [DATASET_PLOT_TITLE[dataset] for dataset in training_datasets] for dataset in df['Test Case']]

df['abs_error'] = df['abs_error']


#########################################
# HTML Table Generation

def color_bars(val, max_val):
    """Generate HTML for a colored bar."""
    percentage = (val / max_val) * 100
    color = f'hsl({100 - percentage}, 100%, 50%)' # Green to red
    return f'<div style="background:linear-gradient(to right, {color} {percentage}%, white {percentage}%); padding: 5px; border-radius: 3px;">{val:.2f}</div>'

def highlight_trained(row):
    if row['Test Case'] in [DATASET_PLOT_TITLE[dataset] for dataset in training_datasets]:
        return ['background-color: rgba(255, 255, 0, 0.3)'] * len(row)  # Light yellow, transparent background
    else:
        return [''] * len(row)

def checkmark(is_trained):
    return '&#10004;' if is_trained else ''

def highlight_positive_diff(row):
    if row['diff'] > 0:
        return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row) # Light green for positive diff
    else:
        return [''] * len(row)

styled = (
    df.style
    # .apply(highlight_trained, axis=1)
    .apply(highlight_positive_diff, axis=1)
    .format({'BFM': lambda x: color_bars(x, 100), 'Log law': lambda x: color_bars(x, 100), 
             'Training?': checkmark, 'abs_error?': checkmark})
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

# Add the note to the HTML code at the top
note_html = '<p style="text-align: left; font-size: 30px;"><b>Note:</b> Colored bars indicate worse performance compared to the log law.</p>'
html_code = note_html + html_code


with open(f'./tables/{model_name}.html', 'w') as f:
    f.write(html_code)

print("Saved 'fancy_gradient_table.html' with colored bars, checkmarks, and diff highlighting.")
