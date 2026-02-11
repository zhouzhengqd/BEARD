#!/usr/bin/env python3
"""
Parse BACON evaluation logs, convert results to JSON,
and compute RR, AE, and CREI metrics.
"""

import re
import json
import os
import argparse
import pandas as pd
from black_box.auto_metrics import calculate_rr, calculate_ae, calculate_crei


def load_config(config_path='parse_bacon_config.json'):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Loaded configuration: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found; using defaults")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Config file is invalid JSON: {e}")
        print("Using default configuration")
        return get_default_config()


def get_default_config():
    """Return the default configuration dictionary."""
    return {
        "input_files": [
            "ConvNet_CIFAR10_1_evaluation_log.txt",
            "ConvNet_CIFAR10_10_evaluation_log.txt",
            "ConvNet_CIFAR10_50_evaluation_log.txt"
        ],
        "output_dir": "CIFAR-10/BACON",
        "metrics_type": "beard",
        "alpha": 0.5,
        "dataset": "CIFAR-10",
        "method": "BACON",
        "output_files": {
            "json_template": "data_{ipc_name}.json",
            "metrics_file": "{method}_metrics_{metrics_type}.json",
            "summary_file": "{method}_{dataset}_Results_Summary.md"
        }
    }


def parse_log_file(log_file_path):
    """Parse a log file and extract clean/attack results."""

    # Regex pattern that supports all method names
    # Uses \w+ to match any method name, avoiding hardcoding
    pattern = r'(\w+) attack on (\w+): final acc is:\s+target_attack: ([\d.]+) \+- [\d.]+, attack time is ([\d.]+); non_target_attack: ([\d.]+) \+- [\d.]+, attack time is ([\d.]+)'

    data = {
        "clean": {
            "accuracy": None,
            "target_time": None,
            "non_target_time": None,
            "time": None  # Stores the minimum clean time
        },
        "attacks": []
    }

    # Targeted results are excluded for specific attacks
    # Deepfool and AutoAttack targeted values are set to null
    excluded_targeted_attacks = {'Deepfool', 'AA'}  # Excluded in targeted mode

    # Attack name mapping (ROME format to standard names)
    attack_name_mapping = {
        'CW': 'C&W',
        'AA': 'AutoAttack'
    }

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all matching attack results
        matches = re.findall(pattern, content)

        for match in matches:
            attack_name, method_name, target_acc, target_time, untarget_acc, untarget_time = match

            # Optional method name check (for debugging)
            # print(f"Found method: {method_name}, attack: {attack_name}")

            # Convert to floats
            target_acc = float(target_acc)
            target_time = float(target_time)
            untarget_acc = float(untarget_acc)
            untarget_time = float(untarget_time)

            if attack_name.lower() == 'clean':
                # Clean data
                data["clean"]["accuracy"] = target_acc
                data["clean"]["target_time"] = target_time
                data["clean"]["non_target_time"] = untarget_time
                # Use the smaller time as the clean time
                data["clean"]["time"] = min(target_time, untarget_time)
            else:
                # Apply attack name mapping
                mapped_attack_name = attack_name_mapping.get(attack_name, attack_name)

                # Attack data - targeted
                if attack_name in excluded_targeted_attacks:
                    # Excluded targeted attack; add null values
                    attack_entry_targeted = {
                        "type": "targeted",
                        "name": mapped_attack_name,
                        "accuracy": None,
                        "time": None
                    }
                else:
                    # Normal targeted attack
                    attack_entry_targeted = {
                        "type": "targeted",
                        "name": mapped_attack_name,
                        "accuracy": target_acc,
                        "time": target_time
                    }
                data["attacks"].append(attack_entry_targeted)

                # Attack data - untargeted (all attack methods)
                attack_entry_untargeted = {
                    "type": "untargeted",
                    "name": mapped_attack_name,
                    "accuracy": untarget_acc,
                    "time": untarget_time
                }
                data["attacks"].append(attack_entry_untargeted)

        return data

    except Exception as e:
        print(f"Error parsing file {log_file_path}: {e}")
        return None


def save_json_data(data, output_path):
    """Save data to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")


def create_metrics_table(all_ipc_data, config):
    """Create a metrics summary table across IPC values."""

    # Table headers
    columns = ['Method', 'Dataset', 'Metrics_Type', 'IPC', 'Attack_Type', 'RR (%)', 'AE (%)', 'CREI (%)']
    table_data = []

    # Add header row
    table_data.append(columns)

    # Add rows per IPC
    for ipc_name in sorted(all_ipc_data.keys()):
        ipc_data = all_ipc_data[ipc_name]

        # Extract IPC value
        ipc_value = ipc_name.replace('ipc_', '')

        # Add targeted attack metrics
        if 'targeted' in ipc_data:
            targeted_row = [
                config['method'],                    # Method
                config['dataset'],                  # Dataset
                config['metrics_type'].upper(),     # Metrics_Type
                ipc_value,                          # IPC
                'Targeted',                         # Attack_Type
                f"{ipc_data['targeted']['RR']:.2f}",         # RR
                f"{ipc_data['targeted']['AE']:.2f}",         # AE
                f"{ipc_data['targeted']['CREI']:.2f}"        # CREI
            ]
            table_data.append(targeted_row)

        # Add untargeted attack metrics
        if 'untargeted' in ipc_data:
            untargeted_row = [
                config['method'],                    # Method
                config['dataset'],                  # Dataset
                config['metrics_type'].upper(),     # Metrics_Type
                ipc_value,                          # IPC
                'Untargeted',                       # Attack_Type
                f"{ipc_data['untargeted']['RR']:.2f}",       # RR
                f"{ipc_data['untargeted']['AE']:.2f}",       # AE
                f"{ipc_data['untargeted']['CREI']:.2f}"      # CREI
            ]
            table_data.append(untargeted_row)

    return table_data


def create_raw_data_table(all_json_data, config):
    """Create a raw-data table matching the data.xlsx layout."""

    # Table headers
    columns = ['Dataset', 'IPC', 'Attack', 'DD-Method', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']
    table_data = []

    # Add header row
    table_data.append(columns)

    # Add rows per IPC
    for ipc_name in sorted(all_json_data.keys()):
        json_data = all_json_data[ipc_name]

        # Extract IPC value
        ipc_value = ipc_name.replace('ipc_', '')

        # Add Clean row
        clean_acc = json_data['clean']['accuracy']
        clean_time = json_data['clean']['time']

        clean_row = [
            config['dataset'],  # Dataset
            ipc_value,          # IPC
            'Clean',            # Attack
            clean_acc,          # DD-Method (targeted accuracy)
            clean_acc,          # Unnamed: 4 (untargeted accuracy, same as targeted for clean)
            clean_time,         # Unnamed: 5 (targeted time)
            clean_time          # Unnamed: 6 (untargeted time, same as targeted for clean)
        ]
        table_data.append(clean_row)

        # Collect attack data
        attack_data = {}

        # Collect data from the attacks list
        for attack in json_data['attacks']:
            attack_name = attack['name']
            attack_type = attack['type']

            if attack_name not in attack_data:
                attack_data[attack_name] = {}

            if attack_type == 'targeted':
                attack_data[attack_name]['targeted_acc'] = attack['accuracy']
                attack_data[attack_name]['targeted_time'] = attack['time']
            elif attack_type == 'untargeted':
                attack_data[attack_name]['untargeted_acc'] = attack['accuracy']
                attack_data[attack_name]['untargeted_time'] = attack['time']

        # Add attack rows
        for attack_name, data in attack_data.items():
            attack_row = [
                '',                                    # Dataset (empty for attack rows)
                '',                                    # IPC (empty for attack rows)
                attack_name,                          # Attack
                data.get('targeted_acc', ''),        # DD-Method (targeted accuracy)
                data.get('untargeted_acc', ''),      # Unnamed: 4 (untargeted accuracy)
                data.get('targeted_time', ''),       # Unnamed: 5 (targeted time)
                data.get('untargeted_time', '')      # Unnamed: 6 (untargeted time)
            ]
            table_data.append(attack_row)

    return table_data


def print_table(table_data):
    """Print the table data to the terminal."""
    if not table_data:
        print("No data to display")
        return

    # Compute max width per column
    col_widths = []
    for col_idx in range(len(table_data[0])):
        max_width = max(len(str(row[col_idx])) for row in table_data)
        col_widths.append(max(max_width, 10))  # Minimum width is 10

    # Print header
    header = table_data[0]
    print("\n" + "="*80)
    print("Generated table data (data.xlsx layout)")
    print("="*80)

    # Print header row
    header_str = " | ".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header)))
    print(header_str)
    print("-" * len(header_str))

    # Print data rows
    for row in table_data[1:]:
        row_str = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        print(row_str)

    print("="*80)


def save_excel_table(table_data, output_path):
    """Save table data to an Excel file (CSV fallback)."""
    try:
        # Create DataFrame
        df = pd.DataFrame(table_data[1:], columns=table_data[0])

        # Save to Excel
        df.to_excel(output_path, index=False)
        print(f"Table data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        # If pandas is unavailable, save as CSV
        try:
            import csv
            csv_path = output_path.replace('.xlsx', '.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(table_data)
            print(f"Saved as CSV: {csv_path}")
        except Exception as csv_e:
            print(f"Failed to save CSV as well: {csv_e}")


def calculate_metrics_for_ipc(json_data, ipc_name, metrics_type='beard', alpha=0.5):
    """Compute metrics for a single IPC entry."""
    
    results = {}
    
    # Process targeted and untargeted attacks
    for attack_type in ['targeted', 'untargeted']:
        print(f"\n=== Calculating {ipc_name} - {attack_type} metrics ===")
        
        # Extract clean data
        clean_acc = json_data["clean"]["accuracy"]
        clean_time = json_data["clean"]["time"]
        
        # Extract attack data
        attack_accs = []
        attack_times = []
        
        for attack in json_data["attacks"]:
            if attack["type"] == attack_type:
                attack_accs.append(attack["accuracy"])
                attack_times.append(attack["time"])
        
        if not attack_accs:
            print(f"Warning: No {attack_type} attack data found")
            continue
        
        try:
            # Calculate RR
            rr = calculate_rr((clean_acc, attack_accs), metrics_type=metrics_type)
            print(f"RR ({metrics_type}): {rr:.2f}%")
            
            # Calculate AE
            ae = calculate_ae((clean_time, attack_times))
            print(f"AE: {ae:.2f}%")
            
            # Calculate CREI
            crei = calculate_crei(rr, ae, alpha=alpha)
            print(f"CREI (α={alpha}): {crei:.2f}%")
            
            results[attack_type] = {
                "RR": round(rr, 2),
                "AE": round(ae, 2),
                "CREI": round(crei, 2),
                "clean_accuracy": clean_acc,
                "clean_time": clean_time,
                "attack_accuracies": attack_accs,
                "attack_times": attack_times
            }
            
        except Exception as e:
            print(f"Error calculating {attack_type} metrics: {e}")
    
    return results


def calculate_combined_metrics(all_ipc_data, metrics_type='beard', alpha=0.5):
    """Compute combined metrics across multiple IPC values."""
    
    combined_results = {}
    
    for attack_type in ['targeted', 'untargeted']:
        print(f"\n=== Calculating combined metrics - {attack_type} attacks ===")
        
        # Collect data across IPCs
        all_rr_data = []
        all_ae_data = []
        
        for ipc_name, ipc_data in all_ipc_data.items():
            if attack_type in ipc_data:
                data = ipc_data[attack_type]
                
                # Prepare RR input
                all_rr_data.append({
                    'clean_acc': data['clean_accuracy'],
                    'attack_accs': data['attack_accuracies']
                })
                
                # Prepare AE input
                all_ae_data.append({
                    'clean_acc': data['clean_time'],  # Time is stored in clean_acc for AE
                    'attack_accs': data['attack_times']
                })
        
        if not all_rr_data:
            print(f"Warning: No {attack_type} attack data for combined metrics")
            continue
        
        try:
            # Calculate combined RR
            combined_rr = calculate_rr(all_rr_data, metrics_type=metrics_type)
            print(f"RRM ({metrics_type}): {combined_rr:.2f}%")
            
            # Calculate combined AE
            combined_ae = calculate_ae(all_ae_data)
            print(f"AEM: {combined_ae:.2f}%")
            
            # Calculate combined CREI
            combined_crei = calculate_crei(combined_rr, combined_ae, alpha=alpha)
            print(f"CREIM (α={alpha}): {combined_crei:.2f}%")
            
            combined_results[attack_type] = {
                "RRM": round(combined_rr, 2),
                "AEM": round(combined_ae, 2),
                "CREIM": round(combined_crei, 2)
            }
            
        except Exception as e:
            print(f"Error calculating combined {attack_type} metrics: {e}")
    
    return combined_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Parse BACON logs and compute metrics')
    parser.add_argument('-c', '--config', default='parse_bacon_config.json',
                       help='Path to the configuration file')
    parser.add_argument('--log-files', nargs='+',
                       help='List of log file paths (overrides config)')
    parser.add_argument('--output-dir',
                       help='Output directory (overrides config)')
    parser.add_argument('--metrics-type', choices=['beard', 'rome'],
                       help='Metrics type (overrides config)')
    parser.add_argument('--alpha', type=float,
                       help='CREI weight (overrides config)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a default configuration file')

    args = parser.parse_args()

    # Create the default config file
    if args.create_config:
        config = get_default_config()
        with open('parse_bacon_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("Created default config: parse_bacon_config.json")
        return

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.log_files:
        config['input_files'] = args.log_files
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.metrics_type:
        config['metrics_type'] = args.metrics_type
    if args.alpha is not None:
        config['alpha'] = args.alpha
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Store all IPC data and results
    all_ipc_data = {}
    all_results = {}
    all_json_data = {}  # Store raw JSON data for table generation

    print(f"\nConfiguration:")
    print(f"  Input files: {config['input_files']}")
    print(f"  Output dir: {config['output_dir']}")
    print(f"  Metrics type: {config['metrics_type']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Method: {config['method']}")

    # Process each log file
    for log_file in config['input_files']:
        if not os.path.exists(log_file):
            print(f"Warning: File {log_file} not found; skipping")
            continue
        
        # Derive IPC value from file name
        if 'CIFAR10_1_' in log_file:
            ipc_name = 'ipc_1'
        elif 'CIFAR10_10_' in log_file:
            ipc_name = 'ipc_10'
        elif 'CIFAR10_50_' in log_file:
            ipc_name = 'ipc_50'
        else:
            ipc_name = os.path.splitext(os.path.basename(log_file))[0]
        
        print(f"\n{'='*50}")
        print(f"Processing file: {log_file} (IPC: {ipc_name})")
        print(f"{'='*50}")
        
        # Parse log file
        json_data = parse_log_file(log_file)
        if json_data is None:
            continue

        # Store raw JSON for table generation
        all_json_data[ipc_name] = json_data

        # Save JSON file
        json_filename = config['output_files']['json_template'].format(ipc_name=ipc_name)
        json_output_path = os.path.join(config['output_dir'], json_filename)
        save_json_data(json_data, json_output_path)

        # Compute metrics for this IPC
        ipc_results = calculate_metrics_for_ipc(json_data, ipc_name,
                                               config['metrics_type'], config['alpha'])

        all_ipc_data[ipc_name] = ipc_results
        all_results[ipc_name] = ipc_results
    
    # Compute combined metrics
    if len(all_ipc_data) > 1:
        print(f"\n{'='*50}")
        print("Computing multi-IPC combined metrics")
        print(f"{'='*50}")
        
        combined_results = calculate_combined_metrics(all_ipc_data,
                                                     config['metrics_type'], config['alpha'])
        all_results['combined'] = combined_results

    # Save final results
    metrics_filename = config['output_files']['metrics_file'].format(
        method=config['method'].lower(),
        metrics_type=config['metrics_type']
    )
    results_output_path = os.path.join(config['output_dir'], metrics_filename)

    final_results = {
        "configuration": {
            "metrics_type": config['metrics_type'],
            "alpha": config['alpha'],
            "dataset": config['dataset'],
            "method": config['method']
        },
        "individual_results": {k: v for k, v in all_results.items() if k != 'combined'},
        "combined_results": all_results.get('combined', {})
    }
    
    save_json_data(final_results, results_output_path)

    # Generate and display summary table
    metrics_output_path = None
    if all_ipc_data:
        print(f"\n{'='*80}")
        print(f"Generating metrics summary table for {config['method']} on {config['dataset']}")
        print(f"{'='*80}")

        # Create metrics summary table
        metrics_table_data = create_metrics_table(all_ipc_data, config)

        # Print metrics table to console
        print_table(metrics_table_data)

        # Save metrics table file
        metrics_filename = f"{config['method'].lower()}_{config['dataset'].lower()}_metrics_{config['metrics_type']}.xlsx"
        metrics_output_path = os.path.join(config['output_dir'], metrics_filename)
        save_excel_table(metrics_table_data, metrics_output_path)

    print(f"\n{'='*50}")
    print("Processing complete!")
    print(f"JSON results saved to: {results_output_path}")
    if metrics_output_path:
        print(f"Metrics summary table saved to: {metrics_output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
