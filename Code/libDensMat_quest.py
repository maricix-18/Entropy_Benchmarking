"""
Useful functions to perform density matrix simulations
"""
import numpy as np
from qiskit.quantum_info import purity, entropy, DensityMatrix
from qiskit import qasm
from libQC import init_circuit, add_circuit_layer
from libQC import define_gates, define_backend
from libMetric import Metrics

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
        #density_matrix = get_output_density_matrix(qc, backend)
        filepath = "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15_results_Quest_fixed/Qasm_qc_Q5_D"+str(depth_index)+".csv"
        density_matrix = read_matrix(filepath, num_qubits)
        #density_matrix = sanitize_to_density_matrix(matrix)
        # force hermitian
        density_matrix = (density_matrix + density_matrix.conj().T) /2
        # normalise
        density_matrix /= np.trace(density_matrix)
        # Repair PSD if needed (clip negative eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(density_matrix)
        eigvals = np.clip(eigvals, 0, None)  # remove tiny negative values
        density_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        # make correct type
        density_matrix = DensityMatrix(density_matrix)
        # Metrics
        # eigvals = np.linalg.eigvalsh(rho)
        # Von Neumann entropy
        # vNd = -np.sum(eigvals * np.log2(eigvals + 1e-16)) / num_qubits
        # Purity
        # pur = np.real(np.trace(rho @ rho))
        # Rényi-2 entropy
        # R2d = -np.log2(pur + 1e-16) / num_qubits

        vNd = entropy(density_matrix, base=2) / num_qubits
        print("vNd : ",vNd)
        print("type vNd:",type(vNd))
        pur = np.real(purity(density_matrix))
        print("pur : ",pur)
        print("type pur:",type(pur))
        R2d = -1 * np.log2(pur) / num_qubits
        print("R2d : ",R2d)
        print("type R2d:",type(R2d))

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

