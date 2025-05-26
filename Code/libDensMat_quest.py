"""
Useful functions to perform density matrix simulations
"""
import numpy as np
from qiskit.quantum_info import DensityMatrix
from qiskit import qasm
from libQC import init_circuit, add_circuit_layer
from libQC import define_gates, define_backend
from scipy.linalg import eigvalsh
from libMetric import Metrics
from qiskit.quantum_info.states.utils import shannon_entropy

def read_matrix(filepath, num_qubits):
    dim = 2 ** num_qubits
    matrix = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            row = []
            entries = line.split(', ')
            for val in entries:
                val = val.replace("i", "j")      # replace imaginary unit
                val = val.replace(" ", "")        # remove all spaces
                val = val.replace("+-", "-")      # fix invalid complex form
                try:
                    row.append(np.complex128(val))
                except ValueError:
                    print(f"Invalid complex number: '{val}'")
                    raise
            matrix.append(row)

    matrix = np.array(matrix, dtype=complex)

    # Validate size
    if matrix.shape != (dim, dim):
        raise ValueError(f"Matrix shape {matrix.shape} doesn't match 2^{num_qubits} x 2^{num_qubits}")

    return matrix

def get_metrics_DensMat_single_width (circuit_params, num_qubits):
    """
    *returns 3 lists all_vNd, all_pur, all_R2d corresponding
    to the values of the von Neumann entropy density, Purity and second-order Renyi 
    entropy density for a quantum circuit of different depths and fixed width num_qubits.
    """

    all_vNd = []
    all_pur = []
    all_R2d = []

    #Initialise circuit
    # qc = init_circuit(circuit_params, num_qubits)

    for depth_index in range(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step):
        print("Depth = ", depth_index)
        
        # get density matrix
        # C:\Users\maria\Desktop\Entropy_Benchmarking\Entropy_Benchmarking\Entropy_Benchmarking\Code\Quest_Q2_D15_DensityMatrix_NoiseModel\Data_fixed\DensMat_qc_Q2_D1.csv
        filepath = "./Quest_Q6_D15_DensityMatrix_NoiseModel/data_fixed/DensMat_qc_Q6_D"+str(depth_index)+".csv"
      
        density_matrix = read_matrix(filepath, num_qubits)
        # normalise density matrix
        density_matrix=density_matrix/np.trace(density_matrix)
        # force hermitian
        density_matrix = (density_matrix + density_matrix.conj().T) / 2
        #print("density matrix type: ", type(density_matrix))
        #flat = density_matrix.flatten()
        eigenvalues = eigvalsh(density_matrix)
        print("eingenvalues: ", eigenvalues)
        # calculate entropy 
        eingv = eigenvalues[eigenvalues > 1e-10] # Remove zero entries to avoid log(0)
        print("eingv: ", eingv)
        #probs = eingv / np.sum(eingv) # normalize -> get prob distribution
        vNd = np.sum(-eingv * np.log2(eingv))/num_qubits
        #vNd = entropy(density_matrix, base=2) / num_qubits
        #print("vNd : ",vNd)
        print("vNd /n qubits:", vNd)
        #print("type vNd:",type(vNd))
        # purity 
        pur = np.real(np.trace(density_matrix@density_matrix))#, density_matrix.data)))
        print("pur : ",pur)
        #print("type pur:",type(pur))
        R2d = -1 * np.log2(pur) / num_qubits
        print("R2d : ",R2d)
        #print("type R2d:",type(R2d))

        all_vNd.append(vNd)
        all_pur.append(pur)
        all_R2d.append(R2d)  

    return(all_vNd, all_pur, all_R2d)


def get_metrics_DensMat(experiment_params):
    """
    *same as get_metrics_DensMat_single_width but for different circuit widths
    (from num_qubits_min to num_qubits_max)
    """
    circuit_params = experiment_params.circuit_params
    noise_params = experiment_params.noise_params
    backend_params = experiment_params.backend_params


    basis_gates = define_gates(circuit_params.choice, 'DensMat')
    backend = define_backend(backend_params, noise_params, basis_gates)
    backend.set_options(method="density_matrix")

    metrics = Metrics(['all_vNd_diff_n', 'all_pur_diff_n', 'all_R2d_diff_n'])
 
    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1, circuit_params.num_qubits_step):
        print("-- \n Number of qubits : ", num_qubits)

        # get data from the saved files 
        all_vNd, all_pur, all_R2d = get_metrics_DensMat_single_width(circuit_params, num_qubits)
        metrics['all_vNd_diff_n'].append(all_vNd)
        metrics['all_pur_diff_n'].append(all_pur)
        metrics['all_R2d_diff_n'].append(all_R2d)
        
    return metrics

