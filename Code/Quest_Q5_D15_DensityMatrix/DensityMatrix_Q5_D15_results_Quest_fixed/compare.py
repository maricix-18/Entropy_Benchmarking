import numpy as np

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

    matrix = np.array(matrix, dtype=np.complex128)

    # Validate size
    if matrix.shape != (dim, dim):
        raise ValueError(f"Matrix shape {matrix.shape} doesn't match 2^{num_qubits} x 2^{num_qubits}")

    return matrix

num_qubits = 5
for i in range(16):
    filepath_quest = "Qasm_qc_Q5_D"+str(i)+".csv"
    rho_quest_fixed = read_matrix(filepath_quest, num_qubits)
    #\Qiskit_DensityMatrix_Q5_D15\QiskitDensMat_data_densmat
    filepath_qiskit = "../../Qiskit_DensityMatrix_Q5_D15/QiskitDensMat_data_densmat/Qasm_qc_Q5_D"+str(i)+".txt"
    rho_qiskit = read_matrix(filepath_qiskit, num_qubits)
    print("For Depth ",i)
    print(np.allclose(rho_quest_fixed, rho_qiskit, atol=1e-12))  # → True
    print(np.array_equal(rho_quest_fixed, rho_qiskit))