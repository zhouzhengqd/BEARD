# Robustness Metrics Evaluation Toolkit

A toolkit for evaluating model robustness against adversarial attacks in dataset distillation scenarios. Calculates Robustness Ratio (RR), Attack Efficiency (AE), and Comprehensive Robustness-Efficiency Index (CREI).

## Overview

This toolkit provides:
- **RR (Robustness Ratio)**: Model robustness using BEARD or ROME metrics
- **AE (Attack Efficiency)**: Computational efficiency based on time costs  
- **CREI**: Combined metric with adjustable weighting
- Support for **white-box**, **black-box query**, and **black-box transfer** attacks

## Project Structure

```
metrics/
├── README.md                          # This file
├── black_box/                         # Black-box evaluation tools
│   ├── README.md
│   ├── auto_metrics.py               # Core metric functions
│   ├── evaluate_metrics.py           # Log parser and evaluator
│   ├── convert_to_white_box.py        # Convert Transfer logs to white-box format
│   ├── Query/BACON/                   # Query attack logs (Spsa, Square)
│   └── Transfer/BACON/                # Transfer attack logs (FGSM, PGD, etc.)
│
└── white_box/                         # White-box evaluation tools
    ├── README.md
    ├── auto_metrics.py
    ├── evaluate_metrics.py
    └── BACON/                         # White-box attack logs
```

**Note:** BACON is used as a reference example in the provided logs and commands.  
The metrics are method-agnostic and can be applied to other dataset distillation methods.


## Quick Start

### Prerequisites

```bash
pip install numpy pandas
# Optional for Excel output:
pip install openpyxl
```

### White-Box Evaluation

```bash
cd white_box
python evaluate_metrics.py \
    --log-files ./BACON/ConvNet_CIFAR10_1_evaluation_log.txt \
              ./BACON/ConvNet_CIFAR10_10_evaluation_log.txt \
              ./BACON/ConvNet_CIFAR10_50_evaluation_log.txt \
    --output-dir ./output
```

### Black-Box Query Evaluation

Query attacks (Spsa, Square) use the same evaluation logic as white-box:

```bash
cd black_box
python evaluate_metrics.py \
    --log-files ./Query/BACON/ConvNet_CIFAR10_50_evaluation_log.txt \
    --output-dir ./output
```

### Black-Box Transfer Evaluation + Conversion

Transfer attack logs contain multiple target models' results and need conversion:

```bash
cd black_box

# Step 1: Convert Transfer logs to white-box format
python convert_to_white_box.py --input-dir ./Transfer

# Step 2: Evaluate the converted logs
python evaluate_metrics.py \
    --log-files ./Transfer/BACON/white_box_format/ConvNet_CIFAR10_50_evaluation_log_BACON.txt \
    --output-dir ./output
```

## Command Line Options

```bash
python evaluate_metrics.py [OPTIONS]

Options:
  -c, --config PATH     Configuration file path
  --log-files PATHS...  Input log files
  --output-dir PATH     Output directory
  --metrics-type TYPE   Metrics type: beard or rome
  --alpha FLOAT         CREI weight coefficient (0.0-1.0)
  --create-config       Create default config file
```

## Attack Types

| Type | Attacks | Description |
|------|---------|-------------|
| White-box | FGSM, PGD, PGD_L2, Deepfool, CW, AA | Full model knowledge |
| Black-box Query | Spsa, Square | Query-based attacks |
| Black-box Transfer | FGSM, PGD, PGD_L2, Deepfool, CW, AA | Transfer from surrogate |

## Output

```
output/
├── data_ipc_*.json           # Per-IPC results
├── *_metrics_*.json          # Combined metrics
└── *_metrics_*.xlsx/.csv     # Summary table
```

## Documentation

- **[White-Box Guide](white_box/README.md)** - White-box evaluation details
- **[Black-Box Guide](black_box/README.md)** - Black-box evaluation and conversion

## License

MIT License
