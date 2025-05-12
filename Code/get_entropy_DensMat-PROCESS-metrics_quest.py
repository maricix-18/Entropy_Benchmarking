"""
PROCESSING
----------------------------------------------------------------------------------
A script where a density matrix simulation is used to access the purity and
Renyi-2 entropy density of the output of a quantum circuit under a noise model
"""

import argparse
import numpy as np

from libIO import get_metrics_fullfilename, get_experiment_fullfilename, dump_to_json
from libUtils import make_time_stamp
from libQC import NoiseParams, CircuitParams, ProtocolParams, BackendParams
from libDensMat_quest import get_metrics_DensMat
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-o", "--output", type=str, required=True, help="Folder where to store the results")

    return parser

parser = build_parser('Running circuit and computing metrics')
args = parser.parse_args()
verbose = args.verbose
resultdir = args.output

# ========================= Parameters =============================
seed = 837
np.random.seed(seed) #reference

# Backend
backend_params = BackendParams('Aer_sim')

# Quantum circuit
circuit_params = CircuitParams('HEA_RIGETTI', num_qubits_min = 5, num_qubits_max = 5, num_qubits_step=1, depth_min = 0, depth_max = 15)

# Noise model
noise_params = NoiseParams()

# Protocol params
DensMat_params = ProtocolParams(name='DensMat', num_samples=0)

# Create experiment
experiment_params = ExperimentParams(backend_params, circuit_params, noise_params, DensMat_params, seed, make_time_stamp())
experiment_params.results_dir = resultdir

# ==================================================================
# Get metrics
metrics = get_metrics_DensMat(experiment_params)

# ==================================================================
# Save metrics
fullfilename = get_metrics_fullfilename(experiment_params)
dump_to_json(metrics, fullfilename)
experiment_params.metrics_file = fullfilename

# Save experiment
jsonfilename = get_experiment_fullfilename(experiment_params)
dump_to_json(experiment_params.to_dict(), jsonfilename)