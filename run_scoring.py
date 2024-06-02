import sys

sys.path.append("..")
import argparse
import pathlib
import os
import numpy as np
import json

root_dir_name = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="This is script to generate data for the HEP competition."
)
parser.add_argument(
    "--prediction",
    "-p",
    type=pathlib.Path,
    help="Prediction file location",
    default=os.path.join(root_dir_name, "sample_result_submission"),
)
parser.add_argument(
    "--output",
    "-o",
    help="Output file location",
    default=os.path.join(root_dir_name, "scoring_output"),
)
parser.add_argument(
    "--reference",
    "-r",
    help="Reference file location",
    default=os.path.join(root_dir_name, "reference_data"),
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)
args = parser.parse_args()

if not args.codabench:
    prediction_dir = args.prediction
    output_dir = args.output
    reference_dir = args.reference
    program_dir = os.path.join(root_dir_name, "ingestion_program")
else:
    prediction_dir = "/app/input/res"
    output_dir = "/app/output"
    reference_dir = "/app/input/ref"
    program_dir = os.path.join(root_dir_name, "ingestion_program")

sys.path.append(program_dir)

settings_file = os.path.join(prediction_dir, "test_settings.json")
print(settings_file)
try:
    with open(settings_file) as f:
        test_settings = json.load(f)
except FileNotFoundError :
    settings_file = os.path.join(reference_dir, "settings", "data.json")
    try:
        with open(settings_file) as f:
            test_settings = json.load(f)
    except FileNotFoundError:
        print("Settings file not found. Please provide the settings file.")
        sys.exit(1)


from HiggsML.score import Scoring


# Init scoring
scoring = Scoring()

# Start timer
scoring.start_timer()

# Load ingestion duration
ingestion_duration_file = os.path.join(prediction_dir, "ingestion_duration.json")
scoring.load_ingestion_duration(ingestion_duration_file)

print(prediction_dir)

# Load ingestions results
scoring.load_ingestion_results(prediction_dir,output_dir)

# Compute Scores
scoring.compute_scores(test_settings)

# Write scores
scoring.write_scores()

# Stop timer
scoring.stop_timer()

# Show duration
scoring.show_duration()

print("\n----------------------------------------------")
print("[✔] Scoring Program executed successfully!")
print("----------------------------------------------\n\n")
