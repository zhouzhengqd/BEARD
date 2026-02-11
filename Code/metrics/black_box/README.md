# Black-Box Attack Evaluation

Evaluate model robustness in black-box scenarios where the attacker has limited or no knowledge of the target model.

## Files

| File | Description |
|------|-------------|
| `auto_metrics.py` | Core metric calculation (RR, AE, CREI) |
| `evaluate_metrics.py` | Log parser and evaluator |
| `convert_to_white_box.py` | Convert Transfer logs to white-box format |

## Black-Box Attack Types

### 1. Query-Based Attacks

Attacks that query the target model to estimate gradients.

**Attack methods:** Spsa, Square

**Usage:** Same as white-box evaluation

```bash
python evaluate_metrics.py \
    --log-files ./Query/BACON/ConvNet_CIFAR10_50_evaluation_log.txt \
    --output-dir ./output
```

### 2. Transfer Attacks

Attacks trained on a surrogate model, evaluated on multiple target models.

**Attack methods:** FGSM, PGD, PGD_L2, Deepfool, CW, AA

**Log format difference:** Contains `eval_accuracies_target` with multiple models' results

```
eval_accuracies_target: [{'DC': 0.5431, 'DSA': 0.6048, 'BACON': 0.6995, ...}]
```

**Usage:** Requires conversion before evaluation

```bash
# Step 1: Convert Transfer logs to white-box format
python convert_to_white_box.py --input-dir ./Transfer

# Creates: ./Transfer/BACON/white_box_format/ConvNet_CIFAR10_50_evaluation_log_{MODEL}.txt

# Step 2: Evaluate specific target model
python evaluate_metrics.py \
    --log-files ./Transfer/BACON/white_box_format/ConvNet_CIFAR10_50_evaluation_log_BACON.txt \
    --output-dir ./output
```

## Directory Structure

```
black_box/
├── Query/BACON/                    # Query attack logs
│   └── ConvNet_CIFAR10_50_evaluation_log.txt
├── Transfer/BACON/                 # Transfer attack logs
│   └── ConvNet_CIFAR10_50_evaluation_log.txt
└── Transfer/BACON/white_box_format/  # Converted logs (generated)
    ├── ConvNet_CIFAR10_50_evaluation_log_DC.txt
    ├── ConvNet_CIFAR10_50_evaluation_log_BACON.txt
    └── ...
```

## Convert Tool

The `convert_to_white_box.py` extracts individual target model results from Transfer logs.

**Supported target models:** DC, DSA, MTT, DM, IDM, BACON, ROME, VULCAN

```bash
python convert_to_white_box.py --input-dir ./Transfer
```

Output: One white-box format log per target model in `white_box_format/` subdirectory.

## Configuration

Example `config.json` for Query evaluation:

```json
{
    "input_files": [
        "./Query/BACON/ConvNet_CIFAR10_50_evaluation_log.txt"
    ],
    "output_dir": "./output",
    "metrics_type": "beard",
    "alpha": 0.5,
    "dataset": "CIFAR10",
    "method": "BACON"
}
```

## Command Line Options

```bash
python evaluate_metrics.py [OPTIONS]

Options:
  -c, --config PATH     Configuration file
  --log-files PATHS...  Input log files
  --output-dir PATH     Output directory
  --metrics-type TYPE   beard or rome
  --alpha FLOAT         CREI weight (0.0-1.0)
  --create-config       Create default config
```

## Output

Same format as [white-box evaluation](../white_box/README.md#output):
- JSON files per IPC
- Combined metrics JSON
- Excel/CSV summary table

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No Clean attack data" | Verify log contains "Clean attack" line |
| "File not found" | Check paths relative to script location |
| Convert tool finds no data | Ensure logs have `eval_accuracies_target` field |

## See Also

- [Main README](../README.md)
- [White-Box README](../white_box/README.md)
