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

def bit_reverse_indices(n):
    """Return bit-reversed indices for n-dimensional matrix"""
    bits = int(np.log2(n))
    return [int('{:0{w}b}'.format(i, w=bits)[::-1], 2) for i in range(n)]

def reorder_density_matrix_quest_to_qiskit(rho):
    n = rho.shape[0]
    perm = bit_reverse_indices(n)
    return rho[np.ix_(perm, perm)]

num_qubits=5

for i in range(16):
    filepath = "Qasm_qc_Q5_D"+str(i)+".csv"
    rho_quest = read_matrix(filepath, num_qubits)
    #rho_qiskit = get_output_density_matrix(qc, backend)
    # C:\Users\maria\Desktop\Entropy_Benchmarking\Qasm_Q5_D15_results_Quest_fixed
    rho_quest_fixed = reorder_density_matrix_quest_to_qiskit(rho_quest)
    filename= "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15_results_Quest_fixed/Qasm_qc_Q5_D"+str(i)+".csv"
    with open(filename, "w") as file:
        dens_mat = np.array(rho_quest_fixed)
        for row in dens_mat:
            line = ", ".join(
                f"{val.real:+.8f}{val.imag:+.8f}i"  # Format: +0.12345678+0.12345678i
                for val in row
            )
            file.write(line + "\n")
# Now they should match numerically
#print(np.allclose(rho_quest_fixed, rho_qiskit, atol=1e-6))  # → True