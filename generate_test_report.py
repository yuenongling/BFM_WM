import os
import re
import sys
import subprocess
import shutil
from collections import defaultdict

def generate_latex_report(image_files, folder_path):
    """
    Generate LaTeX report where files with the same case prefix are grouped as subfigures.
    Does not differentiate subcases, only groups by the case name before wm/loglaw.
    Works with both PDF and PNG files.
    
    Args:
        image_files (list): List of image filenames (PDF and/or PNG)
        folder_path (str): Path to the folder containing the image files
    
    Returns:
        str: Complete LaTeX document content
    """
    # Group files by their case prefix (everything before _wm or _loglaw)
    grouped_files = defaultdict(list)
    
    for file in image_files:
        file_lower = file.lower()
        # Extract the case name (everything before _wm or _loglaw)
        # Handle different file extensions (pdf, png)
        if "_wm." in file_lower or "_loglaw." in file_lower:
            # Find the base name by removing the _wm or _loglaw suffix and extension
            base_name = file_lower
            for suffix in ["_wm.pdf", "_loglaw.pdf", "_wm.png", "_loglaw.png"]:
                base_name = base_name.replace(suffix, "")
            grouped_files[base_name].append(file)
    
    # Sort the keys alphabetically, with numeric parts sorted numerically
    def sort_key(key):
        # Extract numeric prefix if it exists
        match = re.match(r'^(-?\d+)', key)
        if match:
            return (int(match.group(1)), key)
        return (float('inf'), key)  # Non-numeric prefixes come last
        
    sorted_keys = sorted(grouped_files.keys(), key=sort_key)
    
    # Begin LaTeX document
    latex_content = []
    latex_content.append(r'\documentclass{article}')
    latex_content.append(r'\usepackage{graphicx}')
    latex_content.append(r'\usepackage{subcaption}')
    latex_content.append(r'\usepackage[margin=1in]{geometry}')
    latex_content.append(r'\begin{document}')
    latex_content.append(r'\title{PDF Report}')
    latex_content.append(r'\author{Generated Report}')
    latex_content.append(r'\maketitle')
    
    # Create a figure for each group of files with the same case name
    for case_key in sorted_keys:
        files = sorted(grouped_files[case_key])
        
        # Generate a human-readable case title based on the case prefix
        if "naca" in case_key:
            # Extract NACA number but handle potential additional underscores
            naca_parts = case_key.split('_')
            if len(naca_parts) >= 2:
                # Get everything after "naca_" and replace any remaining underscores with spaces
                naca_identifier = ' '.join(naca_parts[1:])
                case_title = f"NACA {naca_identifier}"
            else:
                case_title = case_key.replace('_', ' ')
        elif "apg" in case_key:
            # Extract APG identifier but handle potential additional underscores
            apg_parts = case_key.split('_')
            if len(apg_parts) >= 2:
                # Get everything after "apg_" and replace any remaining underscores with spaces
                apg_identifier = ' '.join(apg_parts[1:])
                case_title = f"APG {apg_identifier}"
            else:
                case_title = case_key.replace('_', ' ')
        elif "airfoil" in case_key:
            # Extract airfoil identifier but handle potential additional underscores
            airfoil_parts = case_key.split('_')
            if len(airfoil_parts) >= 2:
                # Get everything after "airfoil_" and replace any remaining underscores with spaces
                airfoil_identifier = ' '.join(airfoil_parts[1:])
                case_title = f"Airfoil {airfoil_identifier}"
            else:
                case_title = case_key.replace('_', ' ')
        elif re.match(r'^-?\d+$', case_key):
            # Simple numeric case
            case_title = f"Case {case_key}"
        else:
            # Default case title
            case_title = case_key.replace('_', ' ')
        
        # Begin figure environment
        latex_content.append(r'\begin{figure}[htbp]')
        latex_content.append(r'\centering')
        latex_content.append(r'\setlength{\belowcaptionskip}{10pt}')  # Add space below captions
        
        # Add subfigures
        for file in files:
            # Determine custom caption based on filename
            if 'wm' in file.lower():
                subcaption = "BFM wall model"
            elif 'loglaw' in file.lower():
                subcaption = "Log law"
            else:
                # Fallback caption - replace underscores with spaces instead of escaping
                subcaption = file.replace('.pdf', '').replace('_', ' ')
            
            # For two subfigures, use 0.48\textwidth to fit side by side with small gap
            latex_content.append(r'\begin{subfigure}{0.96\textwidth}')
            latex_content.append(r'\centering')
            latex_content.append(r'\includegraphics[width=\textwidth]{' + os.path.join('./', file) + '}')
            latex_content.append(r'\caption{' + subcaption + '}')
            latex_content.append(r'\end{subfigure}')
            latex_content.append(r'~')  # Add space between subfigures
        
        # Add overall figure caption with the case name
        latex_content.append(r'\caption{' + case_title + '}')
        latex_content.append(r'\end{figure}')
        
        # Add page break after each figure
        latex_content.append(r'\clearpage')
    
    # End document
    latex_content.append(r'\end{document}')
    
    return '\n'.join(latex_content)

def compile_latex(tex_file_path, folder_path):
    """
    Compile LaTeX file using pdflatex and remove auxiliary files
    
    Args:
        tex_file_path (str): Path to the LaTeX file
        folder_path (str): Path to the folder containing the file
    
    Returns:
        bool: True if compilation was successful, False otherwise
    """
    # Get the filename without extension for later use
    base_name = os.path.splitext(os.path.basename(tex_file_path))[0]
    
    try:
        # Change to the directory containing the tex file
        original_dir = os.getcwd()
        os.chdir(folder_path)
        
        # Run pdflatex twice to resolve references
        for _ in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', base_name + '.tex'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                print("LaTeX compilation error:")
                print(result.stderr)
                os.chdir(original_dir)
                return False
        
        # Remove auxiliary files
        aux_extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.fls', 
                         '.fdb_latexmk', '.synctex.gz', '.nav', '.snm', '.vrb']
        
        for ext in aux_extensions:
            aux_file = os.path.join(folder_path, base_name + ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)
                
        # Change back to the original directory
        os.chdir(original_dir)
        return True
        
    except Exception as e:
        print(f"Error during compilation: {str(e)}")
        # Make sure we return to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

def main():
    # Check command line arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <folder_path> [--png-only | --include-png]")
        sys.exit(1)
    
    # Get folder path from command line argument
    folder_path = sys.argv[1]
    
    # Check for file type options
    file_mode = "pdf"  # Default mode
    if len(sys.argv) == 3:
        if sys.argv[2] == "--png-only":
            file_mode = "png"
        elif sys.argv[2] == "--include-png":
            file_mode = "both"
    
    # Verify folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    # Get list of files in the specified directory
    all_files = os.listdir(folder_path)
    
    # Get files based on selected mode
    if file_mode == "pdf":
        image_files = [f for f in all_files if f.lower().endswith('.pdf')]
        file_type_str = "PDF"
    elif file_mode == "png":
        image_files = [f for f in all_files if f.lower().endswith('.png')]
        file_type_str = "PNG"
    else:  # both
        image_files = [f for f in all_files if f.lower().endswith(('.pdf', '.png'))]
        file_type_str = "PDF and PNG"
    
    if not image_files:
        print(f"No {file_type_str} files found in {folder_path}")
        sys.exit(1)
    
    # Generate LaTeX content
    latex_content = generate_latex_report(image_files, folder_path)
    
    # Write output to file in the same folder
    output_name = 'image_report.tex'
    output_path = os.path.join(folder_path, output_name)
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX report generated in {output_path}")
    print(f"Processed {len(image_files)} {file_type_str} files grouped by case")
    
    # Compile the LaTeX file and remove auxiliary files
    print("Compiling LaTeX document...")
    if compile_latex(output_path, folder_path):
        print(f"Successfully compiled the report to {os.path.join(folder_path, 'image_report.pdf')}")
        print("Auxiliary files have been removed")
    else:
        print("Failed to compile the LaTeX document")
        print("You may need to compile it manually with 'pdflatex pdf_report.tex'")

if __name__ == "__main__":
    main()
