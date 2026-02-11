import numpy as np
import json
import argparse
import os

def calculate_rr(input_data, metrics_type):
    """
    Calculate Robustness Ratio (RR), supporting both single and multiple IPC scenarios.

    Parameters:
    input_data: Can be one of two formats:
        1. For single IPC:
           - tuple/list: (clean_acc, attack_accs) where:
             * clean_acc (float): Model accuracy on clean data
             * attack_accs (list): List of model accuracies under different attacks, can contain None values
        2. For multiple IPCs:
           - list of dict: Each dict contains 'clean_acc' and 'attack_accs'
             Example: [{'clean_acc': 0.9, 'attack_accs': [0.1, 0.2, 0.15]}, ...]

    metrics_type: str
        The type of RR calculation method to use: 'beard', 'rome', or 'rome+'

    Returns:
    float: Robustness Ratio (RR) value as a percentage
    """
    # Process input data, unify format
    all_clean_accs = []
    all_attack_accs_flat = []

    # Process input based on type
    if isinstance(input_data, (tuple, list)) and len(input_data) == 2 and not isinstance(input_data[0], dict):
        # Single IPC case
        clean_acc, attack_accs = input_data
        clean_acc = float(clean_acc)
        # Filter out None values
        attack_accs = [acc for acc in attack_accs if acc is not None]
        if not attack_accs:
            raise ValueError("All attack accuracies are None, cannot calculate RR.")
        all_clean_accs = [clean_acc]
        all_attack_accs_flat = attack_accs
    elif isinstance(input_data, list) and all(isinstance(x, dict) for x in input_data):
        # Multiple IPC case
        if not input_data:
            raise ValueError("Input IPC data list cannot be empty.")
        for data in input_data:
            if not isinstance(data.get('clean_acc'), (int, float)) or not isinstance(data.get('attack_accs'), (list, tuple)):
                raise ValueError("Each IPC data must contain 'clean_acc' and 'attack_accs' fields in correct format.")
            all_clean_accs.append(float(data['clean_acc']))
            # Filter out None values
            valid_attacks = [acc for acc in data['attack_accs'] if acc is not None]
            all_attack_accs_flat.extend(valid_attacks)
    else:
        raise ValueError("Invalid input format. Must be (clean_acc, attack_accs) tuple or list of dicts with accuracy data.")

    # Convert to numpy arrays for calculation
    all_clean_accs = np.array(all_clean_accs)
    all_attack_accs_flat = np.array(all_attack_accs_flat)

    if len(all_attack_accs_flat) == 0:
        raise ValueError("No valid attack accuracy data for calculation.")

    # Calculate averages and minimum
    avg_clean_acc = np.mean(all_clean_accs)
    avg_attack_acc = np.mean(all_attack_accs_flat)
    min_attack_acc = np.min(all_attack_accs_flat)

    # Ensure clean_acc is greater than or equal to all attack_accs
    if np.any(avg_clean_acc < all_attack_accs_flat):
        print("Warning: Clean accuracy is lower than some attack accuracies, this might indicate data anomalies.")

    # Avoid division by zero
    if avg_clean_acc == min_attack_acc:
        return 0.0

    # Calculate and return RR value
    if not metrics_type:
        raise ValueError("metrics_type is required")
    metrics_type_lower = metrics_type.lower()
    if metrics_type_lower == 'beard':
        rr = 1 - ((avg_clean_acc - avg_attack_acc) / (avg_clean_acc - min_attack_acc))
    elif metrics_type_lower == 'rome':
        rr = 1 - ((avg_clean_acc - avg_attack_acc) * (avg_clean_acc - min_attack_acc)) / (avg_clean_acc ** 2)
    elif metrics_type_lower == 'rome+':
        # ROME+ with penalty for negative ASR values
        # ASR (Attack Success Rate) = clean_acc - attack_acc for each attack
        asr_values = avg_clean_acc - all_attack_accs_flat

        # Calculate penalty for negative ASR values
        # P_neg = (1/N) * sum(1(ASR_i < 0) * |ASR_i|)
        negative_asr_mask = asr_values < 0
        penalty_neg = np.mean(np.where(negative_asr_mask, np.abs(asr_values), 0))

        # Calculate average ASR and maximum ASR (most successful attack)
        avg_asr = np.mean(asr_values)
        min_asr = np.max(asr_values)  # Most successful attack (highest ASR)

        # I-RR formula: 1 - (ASR̄ * ASR*) / (ACC̄^2) - P_neg
        # Apply a scaled penalty for negative ASR values
        rr = 1 - (avg_asr * min_asr) / (avg_clean_acc ** 2) - penalty_neg * 0.01
    else:
        raise ValueError("Unsupported metrics_type. Please use 'beard', 'rome', or 'rome+'.")
    return rr * 100  # Multiply by 100 as per definition

def calculate_ae(input_data):
    """
    Calculate Attack Efficiency Ratio based on time cost, supporting both single and multiple IPC scenarios.

    Parameters:
    input_data: Can be one of two formats:
        1. For single IPC:
           - tuple/list: (clean_time, attack_times) where:
             * clean_time (float): Model inference time on clean data (milliseconds)
             * attack_times (list): List of model time costs under different attacks, can contain None values
        2. For multiple IPCs:
           - list of dict: Each dict contains 'clean_acc' and 'attack_accs' (storing time values)
             Example: [{'clean_acc': 10.5, 'attack_accs': [15.2, 20.1, 18.5]}, ...]

    Returns:
    float: Attack Efficiency Ratio (AE) value as a percentage
    """
    # Process input data, unify format
    all_clean_times = []
    all_attack_times_flat = []

    # Process input based on type
    if isinstance(input_data, (tuple, list)) and len(input_data) == 2 and not isinstance(input_data[0], dict):
        # Single IPC case
        clean_time, attack_times = input_data
        clean_time = float(clean_time)
        # Filter out None values
        attack_times = [t for t in attack_times if t is not None]
        if not attack_times:
            raise ValueError("All attack times are None, cannot calculate AE.")
        all_clean_times = [clean_time]
        all_attack_times_flat = attack_times
    elif isinstance(input_data, list) and all(isinstance(x, dict) for x in input_data):
        # Multiple IPC case
        if not input_data:
            raise ValueError("Input IPC data list cannot be empty.")
        for data in input_data:
            if not isinstance(data.get('clean_acc'), (int, float)) or not isinstance(data.get('attack_accs'), (list, tuple)):
                raise ValueError("Each IPC data must contain 'clean_acc' and 'attack_accs' fields in correct format.")
            all_clean_times.append(float(data['clean_acc']))  # Actually storing time values
            # Filter out None values
            valid_times = [t for t in data['attack_accs'] if t is not None]
            all_attack_times_flat.extend(valid_times)
    else:
        raise ValueError("Invalid input format. Must be (clean_time, attack_times) tuple or list of dicts with time data.")

    # Convert to numpy arrays for calculation
    all_clean_times = np.array(all_clean_times)
    all_attack_times_flat = np.array(all_attack_times_flat)

    if len(all_attack_times_flat) == 0:
        raise ValueError("No valid attack time data for calculation.")

    # Calculate averages and extremes
    avg_clean_time = np.mean(all_clean_times)
    avg_attack_time = np.mean(all_attack_times_flat)
    min_attack_time = np.min(all_attack_times_flat)
    max_attack_time = np.max(all_attack_times_flat)

    # Check time data reasonability
    if np.any(avg_clean_time > all_attack_times_flat):
        print("Warning: Clean data time cost is higher than some attack times, this might indicate anomalies.")

    # Avoid division by zero
    if max_attack_time == avg_clean_time:
        return 0.0

    # Calculate and return AE value
    # print(f"Clean time: {avg_clean_time}, Average attack time: {avg_attack_time}, Minimum attack time: {min_attack_time}, Maximum attack time: {max_attack_time}")
    # Linear AE (current default)
    ae = (avg_attack_time - avg_clean_time) / (max_attack_time - avg_clean_time)
    return ae * 100  # Multiply by 100 as per definition

def calculate_crei(rr, ae, alpha=0.5):
    """
    Calculate Comprehensive Robustness-Efficiency Index (CREI)

    Parameters:
    rr (float): Robustness Ratio value
    ae (float): Attack Efficiency value
    alpha (float): Weight coefficient, default is 0.5, should be in range [0,1]

    Returns:
    float: CREI value as a percentage
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    
    return alpha * rr + (1 - alpha) * ae

def read_json_data(file_path, attack_type='targeted'):
    """
    Read data from a TXT log file and return clean_acc, attack_accs, and attack_times
    For clean time, select the minimum between target and non-target attack times.

    Parameters:
    file_path (str): Path to the log file containing the data
    attack_type (str): Type of attack data to read ('targeted' or 'untargeted')

    Returns:
    tuple: (clean_acc, clean_time, attack_accs, attack_times)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Locate the Clean attack line
    clean_line = None
    attack_lines = []
    for line in lines:
        if "Clean attack" in line:
            clean_line = line
        elif "attack on ROME:" in line:
            attack_lines.append(line)
    
    if not clean_line:
        raise ValueError("No Clean attack data found in the file")
    
    # Parse the Clean line
    # Example format: "Clean attack on ROME: final acc is:  target_attack: 43.49 +- 0.00, attack time is 5.16; non_target_attack: 43.49 +- 0.00, attack time is 2.75"
    target_time = float(clean_line.split("attack time is ")[1].split(";")[0].strip())
    non_target_time = float(clean_line.split("attack time is ")[-1].split(",")[0].strip())
    
    # Use the smaller time as the clean time
    clean_time = min(target_time, non_target_time)
    
    # Extract clean accuracy from the Clean line
    clean_acc = float(clean_line.split("target_attack: ")[1].split(" +-")[0].strip())
    
    # Parse attack data
    attack_accs = []
    attack_times = []
    
    for line in attack_lines:
        if attack_type == 'targeted':
            # Get target_attack data
            acc = float(line.split("target_attack: ")[1].split(" +-")[0].strip())
            time = float(line.split("attack time is ")[1].split(";")[0].strip())
        else:
            # Get non_target_attack data
            acc = float(line.split("non_target_attack: ")[1].split(" +-")[0].strip())
            time = float(line.split("attack time is ")[-1].split(",")[0].strip())
        
        attack_accs.append(acc)
        attack_times.append(time)
    
    # Filter attacks based on type
    if not attack_accs:
        raise ValueError(f"No {attack_type} attack data found in the file")
    
    return float(clean_acc), float(clean_time), attack_accs, attack_times

def save_results_to_json(results, output_file):
    """
    Save calculation results to JSON file

    Parameters:
    results (dict): Dictionary containing calculation results
    output_file (str): Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

def load_config(config_path='config.json'):
    """
    Load configuration from JSON file and validate settings

    Parameters:
    config_path (str): Path to the configuration file

    Returns:
    dict: Validated configuration settings
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['metrics_type', 'alpha', 'root_path', 'input_files', 'output_file']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config file")
        
        # Validate metrics_type
        if config['metrics_type'] not in ['beard', 'rome', 'rome+']:
            raise ValueError("metrics_type must be either 'beard', 'rome', or 'rome+'")
        
        # Validate alpha
        if not 0 <= config['alpha'] <= 1:
            raise ValueError("alpha must be between 0 and 1")
        
        # Process paths
        config['root_path'] = os.path.expanduser(config['root_path'])
        
        # Process output file name - replace ${metrics_type} with actual value
        output_file = config['output_file'].replace('${metrics_type}', config['metrics_type'])
        config['output_file'] = os.path.join(config['root_path'], output_file)
        
        # Process input file paths
        processed_input_files = {}
        for key, file_path in config['input_files'].items():
            processed_input_files[key] = os.path.join(config['root_path'], file_path)
        config['input_files'] = processed_input_files
        
        return config
    except Exception as e:
        raise Exception(f"Error loading config file: {str(e)}")

# --- Main entry point ---
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config()
        
        # List of attack types to process
        attack_types = ['targeted', 'untargeted']
        
        for attack_type in attack_types:
            print(f"\n=== Processing {attack_type} attacks ===")
            
            # Store all IPC data
            all_ipc_data_rr = []  # For RR calculation
            all_ipc_data_ae = []  # For AE calculation
            
            # Store all results
            results = {
                "configuration": {
                    "metrics_type": config['metrics_type'],
                    "alpha": config['alpha'],
                    "root_path": config['root_path'],
                    "attack_type": attack_type
                },
                "individual_results": {},
                "combined_results": {}
            }
            
            # Calculate RR and AE for each IPC
            for ipc_name, file_path in config['input_files'].items():
                try:
                    print(f"\n=== Processing {ipc_name} data ===")
                    print(f"Reading data from: {file_path}")
                    clean_acc, clean_time, attack_accs, attack_times = read_json_data(file_path, attack_type)
                    
                    # Calculate RR (based on accuracy)
                    rr = calculate_rr((clean_acc, attack_accs), metrics_type=config['metrics_type'])
                    print(f"{ipc_name} Robustness Ratio (RR): {rr:.2f}%")
                    
                    # Calculate AE (based on time)
                    ae = calculate_ae((clean_time, attack_times))
                    print(f"{ipc_name} Attack Efficiency (AE): {ae:.2f}%")
                    
                    # Calculate CREI
                    crei = calculate_crei(rr, ae, alpha=config['alpha'])
                    print(f"{ipc_name} Comprehensive Robustness-Efficiency Index (CREI): {crei:.2f}%")
                    
                    # Save individual IPC results
                    results["individual_results"][ipc_name] = {
                        "RR": round(rr, 2),
                        "AE": round(ae, 2),
                        "CREI": round(crei, 2),
                        "data": {
                            "clean": {
                                "accuracy": clean_acc,
                                "time": clean_time
                            },
                            "attacks": {
                                "accuracy": attack_accs,
                                "time": attack_times
                            }
                        }
                    }
                    
                    # Save data for combined metrics calculation
                    all_ipc_data_rr.append({
                        'clean_acc': clean_acc,
                        'attack_accs': attack_accs
                    })
                    all_ipc_data_ae.append({
                        'clean_acc': clean_time,
                        'attack_accs': attack_times
                    })
                    
                except Exception as e:
                    print(f"Error processing {ipc_name} data: {str(e)}")
                    continue
            
            # Calculate combined metrics for all IPCs
            if all_ipc_data_rr and all_ipc_data_ae:
                try:
                    print("\n=== Calculating combined metrics for all IPCs ===")
                    # Calculate combined RR
                    combined_rr = calculate_rr(all_ipc_data_rr, metrics_type=config['metrics_type'])
                    print(f"Combined Robustness Ratio (RR): {combined_rr:.2f}%")
                    
                    # Calculate combined AE
                    combined_ae = calculate_ae(all_ipc_data_ae)
                    print(f"Combined Attack Efficiency (AE): {combined_ae:.2f}%")
                    
                    # Calculate combined CREI
                    combined_crei = calculate_crei(combined_rr, combined_ae, alpha=config['alpha'])
                    print(f"Combined Comprehensive Robustness-Efficiency Index (CREI): {combined_crei:.2f}%")
                    
                    # Save combined results
                    results["combined_results"] = {
                        "RR": round(combined_rr, 2),
                        "AE": round(combined_ae, 2),
                        "CREI": round(combined_crei, 2)
                    }
                    
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(config['output_file'])
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate output filename with attack type
                    output_file = config['output_file'].replace('.json', f'_{attack_type}.json')
                    
                    # Save all results to JSON file
                    save_results_to_json(results, output_file)
                    print(f"\nResults saved to {output_file}")
                    
                except Exception as e:
                    print(f"Error calculating combined metrics: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
