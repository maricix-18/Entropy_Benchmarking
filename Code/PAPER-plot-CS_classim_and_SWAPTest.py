"""
PAPER

Comparison between Classical Shadows and SWAP test via classical simulation
"""
import argparse
import matplotlib.pyplot as plt
import os

from libUtils import get_metrics_specific_width, renyi_entropy_from_purity
from libPlot import compute_xticks, compute_depth_tab, compute_filename, plot_params
from libExperiment import ExperimentParams
from libIO import load_from_json

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i1", "--input1", type=str, required=False, default='C:/results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-04-06_n5-5_D15_M3072_K100_grps3_wo.json', help="Json file for HEA_RIGETTI - CS - without measurement error and meas circuit errors")
    parser.add_argument("-i2", "--input2", type=str, required=False, default='C:/results/Aer_sim/SWAP_HEA_RIGETTI/experiment_2025-04-09_n5-5_D15_meas307200_grps3_wo.json', help="Json file for HEA_RIGETTI - SWAP - without measurement error and meas circuit errors")
    parser.add_argument("-i3", "--input3", type=str, required=False, default='C:/results/Aer_sim/DensMat_HEA_RIGETTI/experiment_2025-03-18_n5-5_D15.json', help="Json file for DensMat")
    parser.add_argument("-o", "--output", type=str, required=False, default='Paper/Methods_plot', help="Folder where to store the results")

    return parser

parser = build_parser('Plotting comparison between two protocols (for PAPER)')
args = parser.parse_args()
verbose = args.verbose
jsonfilename1 = args.input1
jsonfilename2 = args.input2
jsonfilename3 = args.input3
resultdir = args.output

save_fig = True

# ========================= Experiment1 (HEA_RIGETTI - Classical Shadows - without measurement error and meas circuit errors)) =============================
experiment1 = ExperimentParams.from_dict(load_from_json(jsonfilename1))
if experiment1 == None:
    print ("ERROR: reading json file, no experiment #1 can be loaded")
    exit()

circuit_params = experiment1.circuit_params
noise_params = experiment1.noise_params
CS_params = experiment1.protocol_params
# CS_params = CSParams(num_samples=3, num_groups=3, M=729, K=100, protocol_choice='randomized')

num_qubits = circuit_params.num_qubits_min

# Get metrics
metrics_CS = load_from_json(experiment1.metrics_file)
short_metrics_CS = get_metrics_specific_width(metrics_CS, num_qubits, num_qubits)

# ========================= Experiment2 (HEA_RIGETTI - SWAP - without measurement error and meas circuit errors)) =============================
experiment2 = ExperimentParams.from_dict(load_from_json(jsonfilename2))
if experiment2 == None:
    print ("ERROR: reading json file, no experiment #2 can be loaded")
    exit()

# Get metrics
metrics_SWAP = load_from_json(experiment2.metrics_file)
short_metrics_SWAP = get_metrics_specific_width(metrics_SWAP, num_qubits, num_qubits)

# ========================= Experiment3 (HEA_RIGETTI - DensMat =============================

experiment3 = ExperimentParams.from_dict(load_from_json(jsonfilename3))
if experiment3 == None:
    print ("ERROR: reading json file, no experiment #3 can be loaded")
    exit()

# Get metrics
metrics_exact = load_from_json(experiment3.metrics_file)
short_metrics_exact = get_metrics_specific_width(metrics_exact, num_qubits, num_qubits)

"""
Plots
"""
if save_fig:
    # Prepare directory - check if directory exists, if not create it
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

filename = 'n%d_M%d_K%d_grps%d_spls%d.pdf' % (num_qubits, CS_params.M, CS_params.K, CS_params.num_groups, CS_params.num_samples)

depth_tab = compute_depth_tab(circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step)
xticks = compute_xticks(depth_tab, circuit_params.depth_step)

full_filename_puri = compute_filename (resultdir, 'pur', filename)
full_filename_R2d = compute_filename (resultdir, 'R2d', filename)

# Purity
plt.figure()
plt.errorbar(x=depth_tab, y=short_metrics_CS['all_pur_mean_diff_n'], yerr=short_metrics_CS['all_pur_std_diff_n'], label="sim shadows", ecolor="steelblue", capsize=3, ls='none') #Aer Sim CS
plt.errorbar(x=depth_tab, y=short_metrics_SWAP['all_pur_mean_diff_n'], yerr=short_metrics_SWAP['all_pur_std_diff_n'], label="sim swap", ecolor="salmon", capsize=3, ls='none') #Aer Sim SWAP test

plt.plot(depth_tab, short_metrics_exact['all_pur_diff_n'], label="density matrix sim", color="black")
plt.axhline(y=1/2**num_qubits, label="maximally mixed state", color='black', linestyle='-.')
plot_params('Depth', 'Purity', xticks, 1.1, save_fig, full_filename_puri)
plt.show()

# Renyi-2 entropy density
plt.figure()
plt.errorbar(x=depth_tab, y=short_metrics_CS['all_R2d_mean_diff_n'], yerr=short_metrics_CS['all_R2d_std_diff_n'], label="sim shadows", ecolor="steelblue", capsize=3, ls='none') #Aer Sim CS
plt.errorbar(x=depth_tab, y=short_metrics_SWAP['all_R2d_mean_diff_n'], yerr=short_metrics_SWAP['all_R2d_std_diff_n'], label="sim swap", ecolor="salmon", capsize=3, ls='none') #Aer Sim SWAP test

plt.plot(depth_tab, short_metrics_exact['all_R2d_diff_n'], label="density matrix sim", color="black")
plt.axhline(y=renyi_entropy_from_purity(1/2**num_qubits)/num_qubits, label="maximally mixed state", color='black', linestyle='-.')
plot_params('Depth', 'Renyi-2 entropy density', xticks, 1.1, save_fig, full_filename_R2d)
plt.show()