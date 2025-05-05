"""
PROCESSING
----------------------------------------------------------------------------------
Applying a SWAP test circuit to a quantum circuit to estimate the purity of the output state.
"""
# =========================== Packages ===========================
import argparse
import numpy as np

from libUtils import make_time_stamp
from libIO import get_experiment_fullfilename, dump_to_json
from libQC import NoiseParams, BackendParams, CircuitParams5Qbits, CircuitParams3Qbits
from libSWAP import SWAPParams
from libCircuitMeasures import get_and_save_circuit_measurements
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-o", "--output", type=str, required=True, help="Folder where to store the results")

    return parser

parser = build_parser('Running circuit for SWAP')
args = parser.parse_args()
verbose = args.verbose
resultdir = args.output

# ==========================  Parameters ===========================
seed = 837
np.random.seed(seed) #reference

# Backend
backend_params = BackendParams('Aer_sim')

# Quantum circuit
circuit_params = CircuitParams3Qbits('HEA_RIGETTI', depth_max=15)
#circuit_params = CircuitParams5Qbits('HEA_RIGETTI', depth_max=15)
# Noise model
noise_params = NoiseParams()


# SWAP test
SWAP_params = SWAPParams(num_samples=3, num_measurements=320000, num_groups=5)

# Create experiment
experiment_params = ExperimentParams(backend_params, circuit_params, noise_params, SWAP_params, seed, make_time_stamp())
experiment_params.results_dir = resultdir

# ==================================================================
# Get measurement outcomes and save them to csv file
experiment_params = get_and_save_circuit_measurements(experiment_params)

# Save experiment
jsonfilename = get_experiment_fullfilename(experiment_params)
dump_to_json(experiment_params.to_dict(), jsonfilename)
