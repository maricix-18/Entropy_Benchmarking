"""
This library contains the functions used to estimate the purity of a state using the SWAP test
"""
import time
import pandas as pd

from qiskit import QuantumRegister, QuantumCircuit

from libIO import init_csv_file, read_df_from_csv
from libUtils import renyi_entropy_from_purity, compute_stats, bitwise_AND, parity_bit
from libQC import define_backend, define_gates, init_circuit, add_circuit_layer, add_native_ionq_bell_meas
from libQC import Metrics, ProtocolParams, measure_Zbasis
from libShadows import get_df_specific_depth_from_df

# Classes ==========================================================================
class SWAPParams(ProtocolParams):
    def __init__(self, num_samples, num_measurements):
        self.name = 'SWAP'
        self.num_samples = num_samples # Number of samples for the purity/entropy estimate
        self.num_measurements = num_measurements #10**4 (resp. 10**6) for one (resp. two) correct digits in the fractional part of the purity estimate
    @staticmethod
    def from_dict(temp):
        return SWAPParams(temp['num_samples'], temp['num_measurements'])

# Functions ===========================================================================

def swap_test_outcome (outcome):
    """
    returns the parity bit of the bitwise AND of the two halves of the string outcome
    """
    num_qubits = int(len(outcome)/2)
    result = parity_bit(bitwise_AND(outcome[0:num_qubits], outcome[num_qubits:2*num_qubits]))
    return (result)

def estimate_purity_from_swap_test (counts):
    """
    returns the purity estimate from the SWAP test circuit measurements outcomes counts
    """
    nb_outcome_0 = 0
    nb_outcome_1 = 0
    for outcome, nb_times_outcome in counts.items():
        if swap_test_outcome(outcome) == 0:
            nb_outcome_0 += nb_times_outcome
        else:
            nb_outcome_1 += nb_times_outcome
    nb_measurements = nb_outcome_0 + nb_outcome_1
    pur_estimate = 2*nb_outcome_0/nb_measurements - 1
    return (pur_estimate)


def extract_swap_counts_from_df(original_df, wanted_depth_index, sample_index):
    df_fixed_depth = get_df_specific_depth_from_df(original_df, wanted_depth_index)
    df_fixed_depth_sample = df_fixed_depth.iloc[sample_index]
    swap_counts = df_fixed_depth_sample.T[1] #the first column is not counts (depth_index)
    return(eval(swap_counts))

def concatenate_swap_circuit(qc, circuit_choice, num_qubits):
    # Creating the SWAP test circuit used to estimate the purity of the above circuit output
    swap_test_qr1 = QuantumRegister(num_qubits, 'q1')
    swap_test_qr2 = QuantumRegister(num_qubits, 'q2')
    swap_test_circuit = QuantumCircuit(swap_test_qr1, swap_test_qr2, name="circuit + SWAP Circuit")

    qubit_index_list = [i for i in range(0, 2*num_qubits)]
    swap_test_circuit.compose(qc, qubits=qubit_index_list[0:num_qubits], inplace=True)
    swap_test_circuit.compose(qc, qubits=qubit_index_list[num_qubits:2*num_qubits], inplace=True)

    # SWAP test part of the circuit
    for i in range(num_qubits):
        if circuit_choice == 'HEA_IONQ':
            swap_test_circuit = add_native_ionq_bell_meas(swap_test_circuit, i, i+num_qubits)
        else:    
            swap_test_circuit.cx(i, i+num_qubits)
            swap_test_circuit.h(i)

    # print("full circuit : \n", swap_test_circuit)
    return swap_test_circuit

def get_and_save_measurements_circuit_SWAP(experiment_params):
    """
    Gets and saves to a csv file the measurement outcomes of a SWAP test measurement applied to a noisy circuit
    with num_measurements shots. This set of measurements is run num_samples times.
    """
    circuit_params = experiment_params.circuit_params

    # Backend
    gates_1Q, gates_2Q = define_gates(circuit_params.choice, 'SWAP')
    basis_gates = [gates_1Q, gates_2Q]
    backend = define_backend(experiment_params.backend_params, experiment_params.noise_params, basis_gates)
    
    # ==================================================================

    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1):
        print('\n=========> num qubits = ', num_qubits)
        
        start = time.time()

        fullfilename = init_csv_file(experiment_params, num_qubits)
        experiment_params.csv_files[str(num_qubits)] = fullfilename
        # ------------------------------------------------------------------
        
        #Initialise circuit
        qc = init_circuit(num_qubits, circuit_params.depth_min, circuit_params.choice)

        for depth_index in range(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step):
            print("depth : ", depth_index)
            # Build parameterised quantum circuit with depth "depth_index" by adding depth_step layers to previous circuit
            if depth_index>circuit_params.depth_min:
                for index in range(circuit_params.depth_step):
                    qc = add_circuit_layer(circuit_params, num_qubits, qc, depth_index - 1 + index)
            
            swap_test_circuit = concatenate_swap_circuit(qc, circuit_params.choice, num_qubits)

            for _ in range(experiment_params.protocol_params.num_samples):
                IONQ =  experiment_params.backend_params.type == 'IonQ_sim' or experiment_params.backend_params.type == 'IonQ_QPU'

                counts = measure_Zbasis(swap_test_circuit, experiment_params.protocol_params.num_measurements, backend, experiment_params.backend_params.initial_layout, False, IONQ)
               
                # print("\n full circuit which should be unchanged after execution: \n", swap_test_circuit)

                # Save results to csv file
                df = pd.DataFrame([depth_index, counts]).T
                df.to_csv(fullfilename, mode='a', index=False, header=False)
                print("Data saved successfully.")

        # Runtime
        elapsed_time = time.time() - start
        print("\n", "Elapsed time in seconds: ", elapsed_time)

    return experiment_params

def compute_metrics_from_csv_circuit_SWAP(experiment_params):

    metrics = Metrics(['all_pur_mean_diff_n', 'all_pur_std_diff_n', 'all_R2d_mean_diff_n', 'all_R2d_std_diff_n'])

    for num_qubits in range(experiment_params.circuit_params.num_qubits_min, experiment_params.circuit_params.num_qubits_max+1):
        print("\n Number of qubits: ", num_qubits)
        start = time.time()

        # GET DATAFRAME
        df = read_df_from_csv(experiment_params, num_qubits) 
    
        # Lists to store purity for different depths
        all_pur_mean, all_pur_std, all_R2d_mean, all_R2d_std = [], [], [], []

        for depth_index in range(experiment_params.circuit_params.depth_min, experiment_params.circuit_params.depth_max+1, experiment_params.circuit_params.depth_step):
            print("depth : ", depth_index)

            pur_estimate_samples, R2d_estimate_samples = [], []

            for sample_index in range(experiment_params.protocol_params.num_samples):
                
                counts = extract_swap_counts_from_df(df, depth_index, sample_index)
   
                # Purity & Renyi density estimates
                pur_estimate = estimate_purity_from_swap_test(counts)
                R2d_estimate = renyi_entropy_from_purity(pur_estimate)/num_qubits
                #print("\n Purity estimate from SWAP test circuit : ", pur_estimate)
                pur_estimate_samples.append(pur_estimate)
                R2d_estimate_samples.append(R2d_estimate)

            pur_mean, pur_std = compute_stats(pur_estimate_samples)
            R2d_mean, R2d_std = compute_stats(R2d_estimate_samples)
            
            all_pur_mean.append(pur_mean)
            all_pur_std.append(pur_std)
            all_R2d_mean.append(R2d_mean)
            all_R2d_std.append(R2d_std)

        metrics['all_pur_mean_diff_n'].append(all_pur_mean)
        metrics['all_pur_std_diff_n'].append(all_pur_std)
        metrics['all_R2d_mean_diff_n'].append(all_R2d_mean)
        metrics['all_R2d_std_diff_n'].append(all_R2d_std)

        print("=============================================")

        # print("\n Purity estimate from SWAP test circuit : \n", all_pur_mean)

        # Runtime
        elapsed_time = time.time() - start
        elapsed_time_min = int(elapsed_time/60)
        print("\n", "Elapsed time in seconds: ", elapsed_time)
        print("\n", "Elapsed time in minutes: ", elapsed_time_min)

    return metrics
