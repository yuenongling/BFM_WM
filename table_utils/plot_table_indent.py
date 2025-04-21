import sys
sys.path.append('../')
from wall_model_cases import DATASET_PLOT_TITLE 
# Assuming wall_model_cases.py exists and defines DATASET_PLOT_TITLE
# Example definition if missing:
# DATASET_PLOT_TITLE = {k: k for k in ['CH','SYN','TBL_-4','TBL_-3','TBL_-2','TBL_-1','TBL_5','TBL_10','TBL_15','TBL_20', 'gaussian_2M_MAPG','gaussian_2M_FPG','gaussian_2M_APG','gaussian_2M_SEP', 'FS_ZPG','FS_FPG','FS_APG','bub_K']}

import numpy as np
import pandas as pd
import time # Import time module for date


# --- Configuration ---
training_datasets = ['CH','SYN','TBL_-4','TBL_-3','TBL_-2','TBL_-1','TBL_5','TBL_10','TBL_15','TBL_20', 'gaussian_2M_MAPG','gaussian_2M_FPG','gaussian_2M_APG','gaussian_2M_SEP', 'FS_ZPG','FS_FPG','FS_APG','bub_K']
inputs_num = [0, 1, 2] # Use the final inputs_num definition from your code
input_files_path = '../testing_results/renamed/' # Adjust path if needed
output_file_path = './tables/all_model_012_conditional_indent.html' # Output file name
max_val_for_bars = 100 # Max value for color bar percentage calculation
indent_value_px = 25 # Indentation value in pixels for subcases

# --- Data Loading and Preparation ---
# (Keep the data loading section exactly as in the previous version)
# ... (load df_all, calculate Training?, min_input_error, log_law_is_best) ...
df_all = pd.DataFrame() # Placeholder for demonstration

try:
    # Load initial data (assuming inputs1.csv has the base structure)
    base_file = f'{input_files_path}inputs1.csv' # Define base file for clarity
    # --- !!! --- Make sure path is correct --- !!! ---
    # Example: Create dummy data if files don't exist for testing
    try:
        df_base = pd.read_csv(base_file)
    except FileNotFoundError:
        print(f"Warning: Base file {base_file} not found. Creating dummy data.")
        dummy_data = {
            'Test Case': ['Case A', 'Case B', 'Case B station 1', 'Case C', 'Case C station X', 'Case D - Train'],
            'Log law': [10, 55, 60, 95, 8, 12],
            'abs_error': [False, True, True, False, False, True],
            'BFM': [8, 50, 58, 90, 9, 15] # Dummy BFM for base file if needed
        }
        df_base = pd.DataFrame(dummy_data)


    df_all['Test Case'] = df_base['Test Case']
    # Ensure numeric columns are treated as such, coerce errors to NaN
    df_all['Log law'] = pd.to_numeric(df_base['Log law'], errors='coerce')
    df_all['abs_error'] = df_base['abs_error'].astype(bool) # Ensure 'abs_error' is boolean

    # Load BFM data for each input set
    input_cols = [] # Keep track of input columns added
    for inputs in inputs_num:
        col_name = f'inputs{inputs}'
        # --- !!! --- Make sure path is correct --- !!! ---
        # Example: Use dummy data if files don't exist for testing
        try:
            df_input = pd.read_csv(f'{input_files_path}inputs{inputs}.csv')
        except FileNotFoundError:
             print(f"Warning: Input file for inputs{inputs} not found. Using dummy data.")
             # Use dummy BFM, ensure length matches df_base
             dummy_bfm = {
                0: [8, 50, 58, 90, 9, 15],
                1: [9, 48, 55, 88, 10, 14],
                2: [7, 52, 62, 91, 8, 13]
             }.get(inputs, [np.nan]*len(df_base)) # Default to NaN if input num not covered
             df_input = pd.DataFrame({'BFM': dummy_bfm[:len(df_base)]}) # Ensure matching length


        # Ensure numeric columns are treated as such, coerce errors to NaN
        df_all[col_name] = pd.to_numeric(df_input['BFM'], errors='coerce') # Best Fit Model column
        input_cols.append(col_name)


    # Determine if the test case was used in training
    if 'DATASET_PLOT_TITLE' not in globals():
        # print("Warning: DATASET_PLOT_TITLE not found. Using identity mapping.")
        DATASET_PLOT_TITLE = {k: k for k in training_datasets} # Simple fallback
    # Add dummy training case for testing highlighting
    training_datasets.append('Case D - Train')
    DATASET_PLOT_TITLE['Case D - Train'] = 'Case D - Train'


    training_titles = {DATASET_PLOT_TITLE[dataset] for dataset in training_datasets if dataset in DATASET_PLOT_TITLE}
    df_all['Training?'] = df_all['Test Case'].isin(training_titles)

    # Calculate if Log law is better than best input
    if input_cols: # Only calculate if there are input columns
         df_all['min_input_error'] = df_all[input_cols].min(axis=1)
         df_all['log_law_is_best'] = (df_all['Log law'].notna()) & \
                                    (df_all['min_input_error'].notna()) & \
                                    (df_all['Log law'] < df_all['min_input_error'])
    else:
        # Handle case with no input columns
        df_all['min_input_error'] = np.nan
        df_all['log_law_is_best'] = False


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
# (color_bars and checkmark functions remain the same)
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
# (format_row_conditionally remains the same - it prepares content)
def format_row_conditionally(row):
    """
    Formats numeric columns based on the 'abs_error' value in the row.
    Returns a Series with formatted strings.
    """
    formatted_row = row.copy().astype(object) # Work on a copy, ensure object dtype for mixed types
    # Make sure input_cols is accessible here or passed as argument if needed
    numeric_cols = ['Log law'] + [col for col in input_cols if col in row.index] # Use dynamically identified input cols

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
# (Create df_display exactly as before)
df_display = df_all.apply(format_row_conditionally, axis=1)


# --- HTML Table Generation ---
# Define styles
table_styles = [
    {'selector': 'th', 'props': [('font-size', '16px'), ('padding', '10px'), ('background-color', '#f0f0f0'), ('text-align', 'center'), ('font-weight', 'bold')]},
    # Base padding for all td cells - indentation will add to this
    {'selector': 'td', 'props': [('font-size', '14px'), ('padding', '0px'), ('text-align', 'center'), ('padding-left', '10px'), ('padding-right', '10px')]},
    # Padding inside the div for color bars should not affect indentation
    {'selector': 'td > div', 'props': [('padding-top', '10px'), ('padding-bottom', '10px')]},
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('box-shadow', '0 4px 8px rgba(0, 0, 0, 0.1)'), ('border-radius', '8px'), ('overflow', 'hidden'), ('margin', 'auto'), ('width', '95%')]}, # Center table, set width
    {'selector': 'th, td', 'props': [('border', '1px solid #ddd')]},
    # First column left-aligned, slightly more base padding
    {'selector': 'td:first-child, th:first-child', 'props': [('text-align', 'left'), ('padding-left', '15px')]},
    {'selector': 'td:nth-last-child(1), td:nth-last-child(2), th:nth-last-child(1), th:nth-last-child(2)', 'props': [('text-align', 'center'), ('width', '60px'), ('padding-left', '10px'), ('padding-right', '10px')]} # Reset padding for checkmark columns if needed
]


# *** UPDATED: Combined Row Styling Function with Indentation ***
def apply_row_styles(row):
    """Applies background highlights and indentation based on row properties."""
    # Default style for all cells in the row
    base_styles = [''] * len(row) # Start with no specific style for each cell

    # --- Access Original Data for Conditions ---
    try:
        # Use row.name (which is the index from df_all) to look up original values
        original_row = df_all.loc[row.name]
        test_case_name = original_row['Test Case']
        is_trained = original_row['Training?']
        log_law_better = original_row['log_law_is_best']
    except KeyError:
        # Fallback if index lookup fails (should not happen normally)
        return base_styles

    # --- Determine Styles based on Conditions ---
    bg_color = ''
    indentation = ''

    # 1. Background Color (Log law best takes precedence)
    if is_trained:
        bg_color = 'background-color: rgba(255, 255, 0, 0.3);' # Light yellow
    if log_law_better:
        bg_color = 'background-color: rgba(173, 216, 230, 0.5);' # Light blue

    # 2. Indentation
    if "Station" in str(test_case_name).lower(): # Check if 'station' is in the name (case-insensitive)
        indentation = f'padding-left: {indent_value_px}px;'

    # --- Combine Styles for Each Cell ---
    final_styles = []
    for i in range(len(row)):
        # Start with background color
        current_style = bg_color

        # Add indentation - check if it's the first column
        if indentation:
             # For the first column ('Test Case'), add to its specific base padding
             if i == 0:
                  base_padding_first_col = 15 # From table_styles['td:first-child']
                  total_indent = base_padding_first_col + indent_value_px
                  # Overwrite padding-left specifically for the first cell
                  # Need to handle existing bg_color carefully
                  style_parts = [bg_color] if bg_color else []
                  style_parts.append(f'padding-left: {total_indent}px;')
                  current_style = " ".join(style_parts)

             else:
                 # For other columns, add the indentation value to their base padding
                 base_padding_other_cols = 10 # From table_styles['td'] padding-left
                 total_indent = base_padding_other_cols + indent_value_px
                 # Add padding-left rule, preserving background color if any
                 style_parts = [bg_color] if bg_color else []
                 style_parts.append(f'padding-left: {total_indent}px;')
                 current_style = " ".join(style_parts).strip()

        final_styles.append(current_style)


    # --- Special Handling for First Column (Non-indented rows) ---
    # Need to re-apply the specific left-alignment and base padding for non-indented first column cells
    # Styler applies styles sequentially, so general TD styles might overwrite specific first-child styles
    # It's often better to handle column-specific alignment/padding in set_table_styles if possible.
    # Let's simplify: apply indentation directly to the padding value from the base style.

    # --- REVISED Combine Styles Logic ---
    final_styles = []
    is_station_case = "station" in str(test_case_name).lower()

    for i in range(len(row)):
        style_string = bg_color # Start with potential background color

        # Determine base padding
        base_padding_left = 10 # Default from 'td' selector
        if i == 0:
            base_padding_left = 15 # Specific base for first column
        elif len(row) - i <= 2: # Last two columns (checkmarks)
             # Use their specific padding if defined, otherwise default
             base_padding_left = 10 # As defined in example 'td:nth-last-child...'

        # Calculate final padding
        current_padding_left = base_padding_left
        if is_station_case:
            current_padding_left += indent_value_px

        # Add padding style (always add, even if 0, to ensure override if needed?)
        # Let's only add if indentation is applied or if we need to explicitly set base padding
        # This gets complex with Styler's precedence rules. Let's just add the *additional* indent.

        # --- YET ANOTHER REVISED LOGIC ---
        # Let set_table_styles handle base padding and alignment.
        # Here, *only* add background color and *additional* indent padding.

    final_styles = []
    is_station_case = "station" in str(test_case_name).lower()
    additional_indent_style = f'padding-left: {indent_value_px}px;' # Relative indent

    for i in range(len(row)):
        current_style = bg_color # Start with background

        if is_station_case:
             # Simply append the additional padding needed for indentation
             if current_style: # If background exists
                 current_style = f"{current_style.rstrip(';')} {additional_indent_style}"
             else:
                 current_style = additional_indent_style

        final_styles.append(current_style)

    return final_styles


# Apply final styling to the pre-formatted DataFrame
styled = (
    df_display.style
    # Apply the combined row highlighting and indentation function
    .apply(apply_row_styles, axis=1)
    .set_table_styles(table_styles) # Base styles including alignment and base padding
    # No .format() needed for columns formatted by format_row_conditionally
    .set_properties(**{'text-align': 'center'}) # Default alignment, potentially overridden
)


# --- Output ---
# Render to HTML, ensuring HTML tags are not escaped
html_code = styled.to_html(escape=False, index=False) # Add index=False to hide the DataFrame index

# Add a title or notes if desired
current_date = time.strftime("%Y-%m-%d %H:%M:%S %Z") # Get current date and time
title_html = '<h2 style="text-align: center; font-family: sans-serif;">Model Testing Results Comparison</h2>'
note_html = f'''
<div style="width: 95%; margin: 10px auto; font-family: sans-serif; font-size: 13px; line-height: 1.5;">
    <p>Generated on: {current_date}</p>
    <b>Notes:</b>
    <ul>
        <li>Rows containing "station" in the 'Test Case' name are indented to indicate subcases.</li>
        <li>Rows highlighted <span style="background-color: rgba(173, 216, 230, 0.5); padding: 0 3px;">light blue</span> indicate cases where the 'Log law' model performed better (lower error) than all 'inputs' models for that test case.</li>
        <li>Rows highlighted <span style="background-color: rgba(255, 255, 0, 0.3); padding: 0 3px;">light yellow</span> indicate test cases used during training (unless overridden by the blue highlight).</li>
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
    print(f"Saved styled table with conditional formatting, highlighting, and indentation to '{output_file_path}'")
except IOError as e:
    print(f"Error writing file: {e}")
