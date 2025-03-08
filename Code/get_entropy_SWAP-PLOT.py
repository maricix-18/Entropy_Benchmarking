"""
PLOT
----------------------------------------------------------------------------------
Applying a SWAP test circuit to a quantum circuit to estimate the purity of the output state.
"""
# =========================== Packages ===========================
import argparse

from libIO import load_from_json
from libDensMat import get_metrics_DensMat
from libPlot import plot_metrics
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i", "--input", type=str, required=True, help="Json file with parameters from experiment")
    parser.add_argument("-o", "--output", type=str, required=True, help="Folder where to store the results")

    return parser

parser = build_parser('Plotting metrics for SWAP')
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
backend_params = experiment_params.backend_params
circuit_params = experiment_params.circuit_params
noise_params = experiment_params.noise_params
SWAP_params = experiment_params.protocol_params

save_fig = True
show_fully_mixed_state = True
compute_exact = True

# ==================================================================
# Load experiment results (metrics)
metrics = load_from_json(experiment_params.metrics_file)
metrics_exact = get_metrics_DensMat(experiment_params)

# ==================================================================
# Plot
plot_metrics(metrics, metrics_exact, experiment_params, show_fully_mixed_state, compute_exact, save_fig)