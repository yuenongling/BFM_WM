# BFM_WM

## Data

Data are stored in `data/` directories. They cover data summarized in [Table](https://docs.google.com/spreadsheets/d/18hcHtwxN7-uUAkYSo2UL1Tny2SpPDvQiWsKEqjHOlh0/edit?usp=sharing) and the data generation repository [Repo](https://github.com/yuenongling/WM_data_creation?tab=readme-ov-file).

## Training Models

### Configuration

Training models requires a TOML configuration file in the `inputfiles/` directory. The configuration file defines:
- General settings (verbosity, GPU usage)
- Data selection (channel flow, synthetic data, TBL angles)
- Model architecture (layers, activation functions)
- Input and output scaling methods
- Training parameters (loss function, optimizer, scheduler)

### Running Training

```bash
# Train a single model
python train_wm_proposal.py input_file.toml

# Train multiple models with different configurations
python train_wm_proposal.py input1.toml
python train_wm_proposal.py input2.toml
```

## Testing Models

### Interactive Testing

```bash
# Load model and test on selected datasets
python test_model.py model_name.pth
```
When prompted, enter a dataset name or 'P' to print available datasets.

### Generating Test Reports

```bash
# Test model and generate HTML report
python test_model_summary_table.py model_name.pth
```

This generates:
- CSV file with performance metrics
- HTML table visualization showing performance across datasets
