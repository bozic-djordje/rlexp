import os
import csv
import yaml
import argparse
from collections import defaultdict

# Argument parser
parser = argparse.ArgumentParser(description="Compare hyperparameters across training runs.")
parser.add_argument("input_file", type=str, help="Path to the input .txt file")
parser.add_argument("--output_file", type=str, default=None, help="Optional path to output .csv file")
args = parser.parse_args()

# Determine output file path
if args.output_file:
    output_csv = args.output_file
else:
    output_csv = os.path.splitext(args.input_file)[0] + ".csv"

# Store all hyperparameter dicts and metadata
run_ids = []  # For column headers
hyperparams_list = []  # List of dictionaries
success_indicators = []

# Read each line from the input file
with open(args.input_file, 'r') as f:
    for line in f:
        path, success = line.strip().split(',')
        success = success.strip()

        # Extract run ID from last 15 characters of folder name
        run_folder = os.path.basename(os.path.dirname(path))
        run_id = run_folder[-15:]
        run_ids.append(run_id)

        # Load YAML
        with open(path, 'r') as yaml_file:
            hyperparams = yaml.safe_load(yaml_file)["experiment"]
        hyperparams_list.append(hyperparams)
        success_indicators.append(success)

# Get all keys that appear in any of the hyperparameter dicts
all_keys = set()
for params in hyperparams_list:
    all_keys.update(params.keys())

# Only keep keys that differ across runs
differing_keys = []
for key in all_keys:
    values = [params.get(key, None) for params in hyperparams_list]
    if len(set(map(str, values))) > 1:
        differing_keys.append(key)

# Write output CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header
    writer.writerow(['hyperparameter'] + run_ids)

    # Write success indicator row
    writer.writerow(['success_indicator'] + success_indicators)

    # Write rows for differing hyperparameters
    for key in differing_keys:
        row = [key] + [hyperparams.get(key, None) for hyperparams in hyperparams_list]
        writer.writerow(row)

print(f"CSV saved to {output_csv}")