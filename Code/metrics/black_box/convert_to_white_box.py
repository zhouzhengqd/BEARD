#!/usr/bin/env python3
"""
Convert black-box attack logs to the white-box log format for reuse
in the same evaluation pipeline.
"""

import re
import json
import os
import argparse
import datetime

def parse_log_file(log_file_path, target_model):
    """Parse a log file and extract attack results for a target model."""
    pattern = r'(\w+) attack on \w+: final acc is:\s+target_attack: ([\d.]+) \+- [\d.]+, attack time is ([\d.]+); non_target_attack: ([\d.]+) \+- [\d.]+, attack time is ([\d.]+).*?eval_accuracies_target: \{(\[.*?\}\])\}, eval_accuracies_non_target: \{(\[.*?\}\])\}'
    
    # Initialize result structure
    results = {
        "targeted": {
            "clean": None,
            "attacks": {}
        },
        "non_targeted": {
            "clean": None,
            "attacks": {}
        }
    }

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for match in re.finditer(pattern, content, re.DOTALL):
        attack_name, source_target_acc, target_time, source_non_target_acc, non_target_time, target_accs, non_target_accs = match.groups()
        
        # Extract target-model accuracies from the string blocks
        target_acc = None
        non_target_acc = None
        
        # Parse targeted and non-targeted accuracies
        target_matches = re.findall(r"'([^']+)':\s*([\d.]+)", target_accs)
        non_target_matches = re.findall(r"'([^']+)':\s*([\d.]+)", non_target_accs)
        
        for model, acc in target_matches:
            if model == target_model:
                target_acc = float(acc)
                break
                
        for model, acc in non_target_matches:
            if model == target_model:
                non_target_acc = float(acc)
                break
        
        if target_acc is None or non_target_acc is None:
            continue
            
        # Handle Clean results
        if attack_name.lower() == 'clean':
            results["targeted"]["clean"] = {
                "accuracy": target_acc * 100,  # Convert to percentage
                "time": float(target_time)
            }
            results["non_targeted"]["clean"] = {
                "accuracy": non_target_acc * 100,  # Convert to percentage
                "time": float(non_target_time)
            }
            continue
                
        # Normalize attack names
        attack_name_mapping = {
            'CW': 'CW',
            'AA': 'AA',
            'C&W': 'CW',
            'AutoAttack': 'AA'
        }
        mapped_attack_name = attack_name_mapping.get(attack_name, attack_name)
        
        # Store attack results
        results["targeted"]["attacks"][mapped_attack_name] = {
            "accuracy": target_acc * 100,  # Convert to percentage
            "time": float(target_time)
        }
        results["non_targeted"]["attacks"][mapped_attack_name] = {
            "accuracy": non_target_acc * 100,  # Convert to percentage
            "time": float(non_target_time)
        }
        
    return results

def generate_white_box_format(results, target_model="ROME"):
    """Convert results into the white-box log format."""
    output_lines = []
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,000")
    common_params = "dataset: CIFAR10, IPC: 50, DSA:True, num_eval: 0, aug:False , model: ConvNet, attack_type: ['Clean', 'FGSM', 'PGD', 'PGD_L2', 'Deepfool', 'CW', 'AA'], target_attack: [True, False], src_dataset: True"
    
    # Add the logging setup line
    output_lines.append(f"{current_time} - INFO - Logging setup complete.")
    
    # Add Clean results
    target_clean = results["targeted"]["clean"]
    non_target_clean = results["non_targeted"]["clean"]
    if target_clean and non_target_clean:
        clean_line = f"{current_time} - INFO - Clean attack on {target_model}: final acc is:  "
        clean_line += f"target_attack: {target_clean['accuracy']:.2f} +- 0.00, attack time is {target_clean['time']:.2f}; "
        clean_line += f"non_target_attack: {non_target_clean['accuracy']:.2f} +- 0.00, attack time is {non_target_clean['time']:.2f}, "
        clean_line += common_params
        output_lines.append(clean_line)
    
    # Add per-attack results
    attack_order = ['FGSM', 'PGD', 'PGD_L2', 'Deepfool', 'CW', 'AA']  # Fixed order, white-box naming
    for attack_name in attack_order:
        target_data = results["targeted"]["attacks"].get(attack_name)
        non_target_data = results["non_targeted"]["attacks"].get(attack_name)
        
        if target_data and non_target_data:
            attack_line = f"{current_time} - INFO - {attack_name} attack on {target_model}: final acc is:  "
            attack_line += f"target_attack: {target_data['accuracy']:.2f} +- 0.00, attack time is {target_data['time']:.2f}; "
            attack_line += f"non_target_attack: {non_target_data['accuracy']:.2f} +- 0.00, attack time is {non_target_data['time']:.2f}, "
            attack_line += common_params
            output_lines.append(attack_line)
    
    output_lines.append("")  # Add trailing blank line
    return "\n".join(output_lines)

def process_method(method_dir, method_name):
    """Process one method directory of logs."""
    # Target model list
    target_models = ['DC', 'DSA', 'MTT', 'DM', 'IDM', 'BACON', 'ROME', 'VULCAN']
    
    # Create the white_box_format directory under the method folder
    white_box_dir = os.path.join(method_dir, 'white_box_format')
    os.makedirs(white_box_dir, exist_ok=True)
    
    # Locate the evaluation log in the method folder
    log_file = os.path.join(method_dir, "ConvNet_CIFAR10_50_evaluation_log.txt")
    
    if not os.path.exists(log_file):
        print(f"Warning: Evaluation log not found for method {method_name}")
        return
        
    # Process each target model
    for model in target_models:
        print(f"\nProcessing target model for {method_name}: {model}")
        
        # Parse the log file
        results = parse_log_file(log_file, model)
        
        # Generate white-box format logs (targeted and non-targeted results)
        output_path = os.path.join(white_box_dir, f"ConvNet_CIFAR10_50_evaluation_log_{model}.txt")
        content = generate_white_box_format(results, model)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Saved converted output to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert black-box attack logs to white-box format')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(script_dir, 'black_box_file')
    
    parser.add_argument('--input-dir', default=default_input_dir,
                      help='Input directory containing per-method subfolders')
    args = parser.parse_args()
    
    # Iterate over all method folders
    for method_name in os.listdir(args.input_dir):
        method_dir = os.path.join(args.input_dir, method_name)
        if not os.path.isdir(method_dir):
            continue
            
        print(f"\nProcessing method: {method_name}")
        process_method(method_dir, method_name)
    
    print("\nAll methods processed.")

if __name__ == '__main__':
    main()
