"""
PROCESSING
----------------------------------------------------------------------------------
Applying a SWAP test circuit to a quantum circuit to estimate the purity of the output state.
"""
# =========================== Packages ===========================
import argparse

from libIO import get_metrics_fullfilename, dump_to_json, load_from_json
from libSWAP import compute_metrics_from_csv_circuit_SWAP
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i", "--input", type=str, required=True, help="Json file with parameters from experiment")
    parser.add_argument("-o", "--output", type=str, required=True, help="Folder where to store the results")

    return parser

parser = build_parser('Computing metrics')
args = parser.parse_args()
verbose = args.verbose
jsonfilename = args.input
resultdir = args.output

# ========================= Read parameters of experiment from json file =============================
experiment_params = ExperimentParams.from_dict(load_from_json(jsonfilename))
if experiment_params == None:
    print ("ERROR: reading json file, no experiment can be loaded")
    exit()
experiment_params.results_dir = resultdir

# ==================================================================
# Get metrics
metrics = compute_metrics_from_csv_circuit_SWAP(experiment_params)

# Save metrics
fullfilename = get_metrics_fullfilename(experiment_params)
dump_to_json(metrics, fullfilename)
experiment_params.metrics_file = fullfilename

# Save experiment
dump_to_json(experiment_params.to_dict(), jsonfilename)