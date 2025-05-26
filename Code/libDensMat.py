"""
Useful functions to perform density matrix simulations
"""
import numpy as np
from qiskit.quantum_info import purity, entropy
from qiskit import qasm
from libQC import init_circuit, add_circuit_layer
from libQC import define_gates, define_backend
from scipy.linalg import eigvalsh
from libMetric import Metrics

def get_output_density_matrix(qc, backend):
    """
    returns the output density matrix of the quantum circuit qc under some noise model
    defined by the choice of backend, and using Qiskit density matrix simulator
    """

    qc_copy = qc.copy("qc_copy")
    qc_copy.save_density_matrix()

    result = backend.run(qc_copy).result()
    density_matrix = result.data()['density_matrix']

    return(density_matrix)

def get_metrics_DensMat_single_width (backend, circuit_params, num_qubits):
    """
    *returns 3 lists all_vNd, all_pur, all_R2d corresponding
    to the values of the von Neumann entropy density, Purity and second-order Renyi 
    entropy density for a quantum circuit of different depths and fixed width num_qubits.
    """

    all_vNd = []
    all_pur = []
    all_R2d = []

    #Initialise circuit
    qc = init_circuit(circuit_params, num_qubits)

    for depth_index in range(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step):
        print("depth = ", depth_index)

        if depth_index>circuit_params.depth_min:
            for index in range(circuit_params.depth_step):
                qc = add_circuit_layer(circuit_params, num_qubits, qc, depth_index - 1 + index)
                # # for each depth size, save the circuit in qasm format
                # dumped = qc.qasm()
                
                # filename= "Qasm_Q6_D15_DensityMatrix/Qasm_qc_Q6_D"+str(depth_index)+".txt"
                # with open(filename, "w") as file:
                #     file.write(dumped)      
                # print("\n Quantum circuit \n", qc)

        density_matrix = get_output_density_matrix(qc, backend)
        # SAVE density mat data for quest comparison
        # filename= "QiskitDenMat_NoiseModel_data_Q6_D15/Qasm_qc_Q6_D"+str(depth_index)+".txt"
        # with open(filename, "w") as file:
        #     dens_mat = np.array(density_matrix)
        #     for row in dens_mat:
        #         line = ", ".join(
        #             f"{val.real:+.8f}{val.imag:+.8f}i"  # Format: +0.12345678+0.12345678i
        #             for val in row
        #         )
        #         file.write(line + "\n")
        dens_mat = np.array(density_matrix)
        
        # Metrics
        print("density matrix type: ", type(dens_mat))
        vNd = entropy(density_matrix, base=2) / num_qubits
        print("vNd : ",vNd)
        print("type vNd:",type(vNd))
        pur = np.real(purity(density_matrix))
        print("pur : ",pur)
        print("type pur:",type(pur))
        R2d = -1 * np.log2(pur) / num_qubits
        print("R2d : ",R2d)
        print("type R2d:",type(R2d))

        # eigenvalues = eigvalsh(dens_mat)
        # print("eingenvalues: ", eigenvalues)
        # # calculate entropy 
        # eingv = eigenvalues[eigenvalues > 1e-10] # Remove zero entries to avoid log(0)
        # print("eing: ", eingv)
        # #probs = eingv / np.sum(eingv) # normalize -> get prob distribution
        # vNd = np.sum(-eingv * np.log2(eingv))
        # #vNd = entropy(density_matrix, base=2) / num_qubits
        # #print("vNd : ",vNd)
        # #print("vNd /n qubits:", vNd/num_qubits)
        # #print("type vNd:",type(vNd))
        # # purity 
        # pur = np.real(np.trace(dens_mat@dens_mat))#, density_matrix.data)))
        # #print("pur : ",pur)
        # #print("type pur:",type(pur))
        # R2d = -1 * np.log2(pur) / num_qubits
        #print("R2d : ",R2d)


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

        all_vNd, all_pur, all_R2d = get_metrics_DensMat_single_width(backend, circuit_params, num_qubits)
        metrics['all_vNd_diff_n'].append(all_vNd)
        metrics['all_pur_diff_n'].append(all_pur)
        metrics['all_R2d_diff_n'].append(all_R2d)
        
    return metrics

