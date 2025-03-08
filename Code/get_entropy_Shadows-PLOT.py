"""
PLOT
----------------------------------------------------------------------------------
A script where our implementation of the classical shadows protocol is 
applied to the output of a quantum circuit to obtain a purity (and second-order Renyi
entropy density) estimate
== WITH STATS (mean and std) computed over several purity/entropy estimate samples 
(including ERROR BARS)
== WITH POSSIBILITY TO DERANDOMIZE the shadows (i.e., consider each possible 
random unitary one after the other and only once)
"""
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

parser = build_parser('Plotting metrics for Shadows')
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

show_fully_mixed_state = True
compute_exact = True
save_fig = True

# ==================================================================
# Load experiment results (metrics)
metrics = load_from_json(experiment_params.metrics_file)

# Compute exact
metrics_exact = get_metrics_DensMat(experiment_params)

# ==================================================================
# Plot
plot_metrics(metrics, metrics_exact, experiment_params, show_fully_mixed_state, compute_exact, save_fig)