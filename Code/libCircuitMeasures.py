"""
LIBRARY to get circuit measurements.
"""
import time
import numpy as np

from libIO import init_csv_file
from libQC import define_gates, define_backend, init_circuit, add_circuit_layer
from libShadows import get_circuit_measurements_per_depth_CS
from libSWAP import get_circuit_measurements_per_depth_SWAP

def get_circuit_measurements_per_depth(depth, num_qubits, qc, circuit_params, protocol_params, backend, backend_params, fullfilename, verbose=False):
    if protocol_params.name == 'CS':
        return get_circuit_measurements_per_depth_CS(depth, num_qubits, qc, protocol_params, backend, backend_params, fullfilename, verbose=verbose)
    elif protocol_params.name == 'SWAP':
        return get_circuit_measurements_per_depth_SWAP(depth, num_qubits, qc, circuit_params, protocol_params, backend, backend_params, fullfilename, verbose=verbose)

    return None, None


def get_and_save_circuit_measurements(experiment_params, verbose=False):
    """
    Gets and saves to a csv file the measurement outcomes of a given protocol, circuit and noise model.
    with K shots per random measurement and M random measurements. 
    This set of measurements is run num_samples times.
    """
    circuit_params = experiment_params.circuit_params
    protocol_params = experiment_params.protocol_params
    backend_params = experiment_params.backend_params
    protocol_params = experiment_params.protocol_params
    if protocol_params.name == 'CS':
        np.random.seed(experiment_params.seed) # used for randomized protocol

    # Backend
    gates_1Q, gates_2Q = define_gates(circuit_params.choice, protocol_params.name)
    basis_gates = [gates_1Q, gates_2Q]
    backend = define_backend(backend_params, experiment_params.noise_params, basis_gates)

    # ==================================================================

    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1):
        print('\n=========> num qubits = ', num_qubits)

        start = time.time()

        fullfilename = init_csv_file(experiment_params, num_qubits)
        experiment_params.csv_files[str(num_qubits)] = fullfilename
        # ------------------------------------------------------------------
        
        #Initialise circuit
        qc = init_circuit(circuit_params, num_qubits)

        for depth_index in range(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step):
            print("depth : ", depth_index)
            # Build parameterised quantum circuit with depth "depth_index" by adding depth_step layers to previous circuit
            if depth_index>circuit_params.depth_min:
                for index in range(circuit_params.depth_step):
                    qc = add_circuit_layer(circuit_params, num_qubits, qc, depth_index - 1 + index)

            get_circuit_measurements_per_depth(depth_index, num_qubits, qc, circuit_params, protocol_params, backend, backend_params, fullfilename, verbose=verbose)
                                 
        # Runtime
        elapsed_time = time.time() - start
        if verbose: print("\n", "Elapsed time in seconds: ", elapsed_time)
    
    return experiment_params
