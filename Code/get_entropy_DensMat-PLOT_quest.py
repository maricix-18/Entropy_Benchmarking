"""
PLOT
----------------------------------------------------------------------------------
A script where a density matrix simulation is used to access the purity and
Renyi-2 entropy density of the output of a quantum circuit under a noise model
"""
import argparse

from libIO import load_from_json
from libPlot_quest import plot_metrics_DensMat
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i", "--input", type=str, required=True, help="Json file with parameters from experiment")
    parser.add_argument("-o", "--output", type=str, required=True, help="Folder where to store the results")

    return parser

parser = build_parser('Plotting metrics for DensMat')
args = parser.parse_args()
verbose = args.verbose
jsonfilename = args.input
resultdir = args.output

# ========================= Read parameters of experiment from json file =============================
# experiment_params = ExperimentParams.from_dict(load_from_json(jsonfilename))
# if experiment_params == None:
#     print ("ERROR: reading json file, no experiment can be loaded")
#     exit()
# experiment_params.results_dir = resultdir

save_fig = True
show_maximally_mixed_state = True

# ==================================================================
# Load experiment results (metrics)
## for quest take json files from folder
metrics = load_from_json(jsonfilename)
#metrics = load_from_json(experiment_params.metrics_file)

# ==================================================================
# Plot
plot_metrics_DensMat( metrics, show_maximally_mixed_state, resultdir,  save_fig)