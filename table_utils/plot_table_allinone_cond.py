import sys
sys.path.append('../')
from wall_model_cases import DATASET_PLOT_TITLE 
'''
Read in testing results and generate a table with colored bars
'''

# Assuming wall_model_cases.py exists and defines DATASET_PLOT_TITLE
# Example definition if missing:
# DATASET_PLOT_TITLE = {k: k for k in ['CH','SYN','TBL_-4','TBL_-3','TBL_-2','TBL_-1','TBL_5','TBL_10','TBL_15','TBL_20', 'gaussian_2M_MAPG','gaussian_2M_FPG','gaussian_2M_APG','gaussian_2M_SEP', 'FS_ZPG','FS_FPG','FS_APG','bub_K']}

import numpy as np
import pandas as pd

# --- Configuration ---
training_datasets = ['CH','SYN','TBL_-4','TBL_-3','TBL_-2','TBL_-1','TBL_5','TBL_10','TBL_15','TBL_20', 'gaussian_2M_MAPG','gaussian_2M_FPG','gaussian_2M_APG','gaussian_2M_SEP', 'FS_ZPG','FS_FPG','FS_APG','bub_K']
inputs_num = [0, 1, 2, 3, 4, 5, 6] # Use the final inputs_num definition from your code
input_files_path = '../testing_results/renamed/' # Adjust path if needed
output_file_path = './tables/all_model_new.html' # Output file name
max_val_for_bars = 100 # Max value for color bar percentage calculation

# --- Data Loading and Preparation ---
df_all = pd.DataFrame()

try:
    # Load initial data (assuming inputs1.csv has the base structure)
    base_file = f'{input_files_path}inputs1.csv' # Define base file for clarity
    df_base = pd.read_csv(base_file)
    df_all['Test Case'] = df_base['Test Case']

    # Replace PIPE with PIPE_2 in the 'Test Case' column
    df_all['Test Case'].replace('PIPE', 'PIPE (up to $Re_{\tau}=12,000$', inplace=True)
    df_all['Test Case'].replace('Synthetic (log law) data', 'Synthetic (log law) data (up to $Re_{\tau}=100,000$', inplace=True)

    # Ensure numeric columns are treated as such, coerce errors to NaN
    df_all['EQWM'] = pd.to_numeric(df_base['Log law'], errors='coerce')
    df_all['abs_error'] = df_base['abs_error'].astype(bool) # Ensure 'abs_error' is boolean

    # Load BFM data for each input set
    input_cols = [] # Keep track of input columns added
    for inputs in inputs_num:
        col_name = f'inputs{inputs}'
        df = pd.read_csv(f'{input_files_path}inputs{inputs}.csv')
        # Ensure numeric columns are treated as such, coerce errors to NaN
        df_all[col_name] = pd.to_numeric(df['BFM'], errors='coerce') # Best Fit Model column
        input_cols.append(col_name)

    training_titles = {DATASET_PLOT_TITLE[dataset] for dataset in training_datasets if dataset in DATASET_PLOT_TITLE}
    df_all['Training?'] = df_all['Test Case'].isin(training_titles)

    # *** NEW: Calculate if Log law is better than best input ***
    # Calculate the minimum value across input columns for each row (ignores NaN by default)
    df_all['min_input_error'] = df_all[input_cols].min(axis=1)
    # Compare Log law error with the minimum input error
    # Ensure comparison only happens when both values are valid numbers
    df_all['log_law_is_best'] = (df_all['EQWM'].notna()) & \
                                (df_all['min_input_error'].notna()) & \
                                (df_all['EQWM'] < df_all['min_input_error'])

except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure the CSV files exist at {input_files_path}")
    sys.exit(1)
except KeyError as e:
    print(f"Error: Column {e} not found in one of the CSV files.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    sys.exit(1)


# --- Formatting Functions ---
def color_bars(val, max_val):
    """Generate HTML for a colored bar."""
    if pd.isna(val):
        return "" # Handle NaN values gracefully
    percentage = max(0, min(100, (val / max_val) * 100)) # Clamp percentage between 0 and 100
    hue = 120 * (1 - percentage / 100) # Hue: 120 (green) at 0%, down to 0 (red) at 100%
    color = f'hsl({hue}, 100%, 50%)'
    # Added min-width and text alignment for better appearance inside the div
    return f'<div style="background:linear-gradient(to right, {color} {percentage:.2f}%, white {percentage:.2f}%); padding: 5px; border-radius: 3px; width: 100%; box-sizing: border-box; text-align: center; min-width: 60px;">{val:.2f}</div>'

def checkmark(is_true):
    """Return a checkmark symbol if True, otherwise empty string."""
    return '&#10004;' if is_true else ''

# --- Row-wise Conditional Formatting Function ---
def format_row_conditionally(row):
    """
    Formats numeric columns based on the 'abs_error' value in the row.
    Returns a Series with formatted strings.
    """
    formatted_row = row.copy().astype(object) # Work on a copy, ensure object dtype for mixed types
    numeric_cols = ['EQWM'] + input_cols # Use dynamically identified input cols

    # --- Handle 'Test Case' Column Color ---
    test_case_value = row.get('Test Case', '') # Get original value safely
    if pd.notna(test_case_value): # Check if it's not NaN/None
        test_case_str = str(test_case_value)
        is_station_case = "station" in test_case_str.lower()

        if is_station_case:
            # Apply lighter color (e.g., grey) using an HTML span
            color = '#555' # Darker grey for better readability than 'grey'
        else:
            color = '#FF0000' # Darker grey for better readability than 'grey'

        formatted_row['Test Case'] = f'<span style="color: {color};">{test_case_str}</span>'
    # Keep the original value if no modification needed (already copied)
    # else: keep the value already in formatted_row['Test Case']

    use_scientific = row['abs_error'] # Check the condition for the row

    for col in numeric_cols:
        if col in row.index: # Check if column exists
            value = row[col]
            if pd.notna(value):
                if use_scientific:
                    formatted_row[col] = f"{value:.2e}" # Scientific notation
                else:
                    # Use color_bars, but apply formatting only if the value isn't NaN
                    formatted_row[col] = color_bars(value, max_val_for_bars)
            else:
                formatted_row[col] = "" # Handle NaN

    # Format boolean columns to checkmarks (can also be done later with .format)
    formatted_row['Training?'] = checkmark(row['Training?'])
    formatted_row['abs_error'] = checkmark(row['abs_error'])
    # We don't need to format 'log_law_is_best' or 'min_input_error' for display here
    # They are used for styling logic later. Let's remove them from the display df.
    if 'min_input_error' in formatted_row.index:
         del formatted_row['min_input_error']
    if 'log_law_is_best' in formatted_row.index:
         del formatted_row['log_law_is_best']


    return formatted_row

# --- Apply Row-wise Formatting ---
# Create a DataFrame with the formatted strings/HTML
df_display = df_all.apply(format_row_conditionally, axis=1)


# --- HTML Table Generation ---
# Define styles
table_styles = [
    {'selector': 'th', 'props': [('font-size', '16px'), ('padding', '10px'), ('background-color', '#f0f0f0'), ('text-align', 'center'), ('font-weight', 'bold')]},
    {'selector': 'td', 'props': [('font-size', '14px'), ('padding', '0px'), ('text-align', 'center')]}, # Reduce padding for cells containing divs
    {'selector': 'td > div', 'props': [('padding', '10px')]}, # Add padding *inside* the div for color bars
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('box-shadow', '0 4px 8px rgba(0, 0, 0, 0.1)'), ('border-radius', '8px'), ('overflow', 'hidden'), ('margin', 'auto'), ('width', '90%')]}, # Center table, set width
    {'selector': 'th, td', 'props': [('border', '1px solid #ddd')]},
    {'selector': 'td:first-child, th:first-child', 'props': [('text-align', 'left'), ('padding-left', '15px')]},
    {'selector': 'td:nth-last-child(1), td:nth-last-child(2), th:nth-last-child(1), th:nth-last-child(2)', 'props': [('text-align', 'center'), ('width', '60px')]} # Center & width for checkmark columns
]


# *** NEW: Combined Row Styling Function ***
def apply_row_styles(row):
    """Applies background highlights based on Training status and Log Law performance."""
    # Default style for all cells in the row
    styles = [''] * len(row)

    # Access the pre-calculated boolean values from the original df_all using the row's name (index)
    try:
        is_trained = df_all.loc[row.name, 'Training?']
        log_law_better = df_all.loc[row.name, 'log_law_is_best']
    except KeyError:
        # Handle cases where the index might not align (shouldn't happen with this workflow)
        return styles # Return default styles if lookup fails

    # Define background colors (using rgba for transparency)
    trained_color = 'background-color: rgba(255, 255, 0, 0.3);' # Light yellow, slightly more transparent
    log_law_best_color = 'background-color: rgba(173, 216, 230, 0.5);' # Light blue, slightly transparent

    # Apply styles - Log law highlight takes precedence over training highlight if both are true
    # if is_trained:
    #     styles = [trained_color] * len(row)
    if log_law_better:
        styles = [log_law_best_color] * len(row) # Overwrites if log law is best

    return styles


# Apply final styling to the pre-formatted DataFrame
styled = (
    df_display.style
    # Apply the combined row highlighting function
    .apply(apply_row_styles, axis=1)
    .set_table_styles(table_styles)
    # No .format() needed for columns formatted by format_row_conditionally
    .set_properties(**{'text-align': 'center'}) # Default alignment, overridden by specific styles
)


# --- Output ---
# Render to HTML, ensuring HTML tags are not escaped
html_code = styled.to_html(escape=False, index=False) # Add index=False to hide the DataFrame index

# Add a title or notes if desired
title_html = '<h2 style="text-align: center; font-family: sans-serif;">Model Testing Results Comparison</h2>'
note_html = '''
<div style="width: 90%; margin: 10px auto; font-family: sans-serif; font-size: 13px; line-height: 1.5;">
    <b>Notes:</b>
    <ul>
        <li>Rows highlighted <span style="background-color: rgba(173, 216, 230, 0.5); padding: 0 3px;">light blue</span> indicate cases where the 'Log law' model performed better (lower error) than all 'inputs' models for that test case.</li>
        <li>Numeric cells show model error. For rows where 'abs_error' is True (indicated by &#10004; in the 'abs_error' column), errors are shown in scientific notation (e.g., 1.23e+02).</li>
        <li>Other numeric cells show performance relative to a max value (assumed 100). The colored bar indicates error magnitude (green=low error, red=high error).</li>
        <li>&#10004; in the 'Training?' column indicates the test case data was part of the training set.</li>
    </ul>
</div>
'''
html_code = title_html + note_html + html_code

# Save the HTML file
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(html_code)
    print(f"Saved styled table with conditional formatting and highlighting to '{output_file_path}'")
except IOError as e:
    print(f"Error writing file: {e}")
