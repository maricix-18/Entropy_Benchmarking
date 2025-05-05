"""
LIBRARY for metrics computation once circuit measures are available.
"""

import time
import numpy as np

from libIO import read_df_from_csv
from libUtils import compute_stats
from libShadows import compute_metrics_per_depth_CS
from libSWAP import compute_metrics_per_depth_SWAP

class Metrics(dict):
    def __init__(self, metric_list):
        for m in metric_list:
            self[m] = []

def compute_metrics_per_depth(df, depth, num_qubits, protocol_params, verbose=False):
    if protocol_params.name == 'CS':
        return compute_metrics_per_depth_CS(df, depth, num_qubits, protocol_params, verbose=verbose)
    elif protocol_params.name == 'SWAP':
        return compute_metrics_per_depth_SWAP(df, depth, num_qubits, protocol_params, verbose=verbose)

    return None, None
 
def compute_metrics_from_measures(experiment_params, verbose=False):
    """
    Returns data in the form of the purity of the output state of noisy circuit of 
    a family given by circuit_choice, for width between num_qubits_min and num_qubits_max, 
    for depth between 0 and depth max with step depth_step.

    The purity is obtained depending on a given protocol.
    """

    protocol_params = experiment_params.protocol_params
    if protocol_params.name == 'CS':
        np.random.seed(experiment_params.seed) #used for artif1 and artif2

    circuit_params = experiment_params.circuit_params

    metrics = Metrics(['all_pur_mean_diff_n', 'all_pur_std_diff_n', 'all_R2d_mean_diff_n', 'all_R2d_std_diff_n'])

    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1):
        print("\n Number of qubits: ", num_qubits)
        start = time.time()

        # GET DATAFRAME
        df = read_df_from_csv(experiment_params, num_qubits) 
        if df.empty:
            if verbose: print("The original dataframe df is empty.")
            return (None, None, None, None)            
    
        # Lists to store purity for different depths
        all_pur_mean, all_pur_std, all_R2d_mean, all_R2d_std = [], [], [], []

        for depth_index in range(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step):
            print("depth : ", depth_index)

            pur_samples, R2d_samples = [], []
            start_depth = time.time()

            pur_samples, R2d_samples = compute_metrics_per_depth(df, depth_index, num_qubits, protocol_params)

            if verbose: print("\n TIME depth (sec)", (time.time() - start_depth))

            pur_mean, pur_std = compute_stats(pur_samples)
            R2d_mean, R2d_std = compute_stats(R2d_samples)
            
            all_pur_mean.append(pur_mean)
            all_pur_std.append(pur_std)
            all_R2d_mean.append(R2d_mean)
            all_R2d_std.append(R2d_std)

        metrics['all_pur_mean_diff_n'].append(all_pur_mean)
        metrics['all_pur_std_diff_n'].append(all_pur_std)
        metrics['all_R2d_mean_diff_n'].append(all_R2d_mean)
        metrics['all_R2d_std_diff_n'].append(all_R2d_std)

        # Runtime
        elapsed_time = time.time() - start
        print("\n", "Elapsed time in seconds: ", elapsed_time)

    return metrics

