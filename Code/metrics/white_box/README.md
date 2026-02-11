# White-Box Attack Evaluation

Evaluate model robustness in white-box scenarios where the attacker has full knowledge of the target model.

## Files

| File | Description |
|------|-------------|
| `auto_metrics.py` | Core metric calculation (RR, AE, CREI) |
| `evaluate_metrics.py` | Main evaluation script |

## Usage

### Basic Evaluation

```bash
python evaluate_metrics.py \
    --log-files ./BACON/ConvNet_CIFAR10_1_evaluation_log.txt \
              ./BACON/ConvNet_CIFAR10_10_evaluation_log.txt \
              ./BACON/ConvNet_CIFAR10_50_evaluation_log.txt \
    --output-dir ./output
```

### With Custom Metrics

```bash
python evaluate_metrics.py \
    --log-files ./BACON/ConvNet_CIFAR10_50_evaluation_log.txt \
    --output-dir ./output \
    --metrics-type rome \
    --alpha 0.7
```

## Log Format

White-box logs contain attack results on a single target model:

```
TIMESTAMP - INFO - Clean attack on METHOD: final acc is:  target_attack: XX.XX +- 0.00, attack time is X.XX; non_target_attack: XX.XX +- 0.00, attack time is X.XX, dataset: DATASET, IPC: N, ...
TIMESTAMP - INFO - FGSM attack on METHOD: final acc is:  target_attack: XX.XX +- 0.00, attack time is X.XX; non_target_attack: XX.XX +- 0.00, attack time is X.XX, dataset: DATASET, IPC: N, ...
```

**Attacks:** Clean, FGSM, PGD, PGD_L2, Deepfool, CW, AA

**Note:** Deepfool and AA are excluded from targeted calculations (set to null).

## Configuration

Create `config.json`:

```json
{
    "input_files": [
        "./BACON/ConvNet_CIFAR10_1_evaluation_log.txt",
        "./BACON/ConvNet_CIFAR10_10_evaluation_log.txt",
        "./BACON/ConvNet_CIFAR10_50_evaluation_log.txt"
    ],
    "output_dir": "./output",
    "metrics_type": "beard",
    "alpha": 0.5,
    "dataset": "CIFAR10",
    "method": "BACON"
}
```

Then run:
```bash
python evaluate_metrics.py --config config.json
```

## Output

- **JSON files**: Detailed per-IPC results with RR, AE, CREI values
- **Combined metrics**: Aggregated across all IPCs
- **Excel/CSV**: Summary table

Example JSON output:
```json
{
    "individual_results": {
        "ipc_50": {
            "targeted": {
                "RR": 19.78,
                "AE": 29.70,
                "CREI": 24.74
            }
        }
    },
    "combined_results": {
        "targeted": {
            "RRM": 36.83,
            "AEM": 29.83,
            "CREIM": 33.33
        }
    }
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No Clean attack data found" | Ensure log contains "Clean attack" line |
| "File not found" | Use absolute or relative paths from script location |
| Excel save fails | Install `openpyxl` or use CSV fallback |

## See Also

- [Main README](../README.md)
- [Black-Box README](../black_box/README.md)
