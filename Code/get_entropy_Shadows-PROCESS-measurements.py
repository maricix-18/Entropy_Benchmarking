"""
PROCESSING (from circuit to measurement outcomes saved in csv files)
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
import numpy as np

from libIO import get_experiment_fullfilename, dump_to_json
from libUtils import make_time_stamp
from libQC import NoiseParams, CircuitParams3Qbits, BackendParams
from libShadows import get_and_save_measurements_circuit_CS, CSParams
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-o", "--output", type=str, required=True, help="Folder where to store the results")

    return parser

parser = build_parser('Running circuit for Shadows')
args = parser.parse_args()
verbose = args.verbose
resultdir = args.output

# ========================= Parameters =============================
seed = 837
np.random.seed(seed) #reference

# Backend
backend_params = BackendParams('Aer_sim')

# Quantum circuit
circuit_params = CircuitParams3Qbits(choice='HEA_RIGETTI', depth_max=15)

# Noise model
noise_params = NoiseParams(p_DP1 = 0.01, p_DP2=0.1, p_AD1=0, p_AD2=0, p_meas=[[1, 0], [0, 1]])

# Classical shadows
CS_params = CSParams(num_samples=3, num_groups=1, M=50, K=1000, protocol_choice='randomized')

# Create experiment
experiment_params = ExperimentParams(backend_params, circuit_params, noise_params, CS_params, seed, make_time_stamp())
experiment_params.results_dir = resultdir

# ==================================================================
# Get measurement outcomes and save them to csv file
experiment_params = get_and_save_measurements_circuit_CS(experiment_params, verbose=False)

# Save experiment
jsonfilename = get_experiment_fullfilename(experiment_params)
dump_to_json(experiment_params.to_dict(), jsonfilename)