"""
A library dedicated to functions related to the purity model
"""
import numpy as np

def purity_model_globalDP (num_qubits, depth, alpha_1, alpha_2):
    """
    Returns a purity value based on a global depolarising noise model
    """
    return (1 - 2**(-num_qubits))*(np.exp(-2*(2*alpha_1*num_qubits + alpha_2 * (num_qubits - 1))* depth) - 1) + 1

def purity_model_globalDP_CS_circuit_measerr (num_qubits, depth, alpha_1, alpha_2, beta):
    """
    Returns a purity value based on a global depolarising noise model
    where beta accounts for CS circuit+measurement error
    """ 
    return (1 - 2**(-num_qubits))*(np.exp(-2*(alpha_1*num_qubits*(2*depth) + alpha_2*(num_qubits-1)*depth + beta*num_qubits)) - 1) + 1

def purity_model_globalDP_from_local_dep_prob (num_qubits, depth, p_DP1Q, p_DP2Q):
    """
    Returns a purity value based on a global depolarising noise model function of local depolarising probabilities
    """
    alpha_1 = np.log(1/(1 - p_DP1Q))
    alpha_2 = np.log(1/(1 - p_DP2Q))
    return (1 - 2**(-num_qubits))*(np.exp(-2*(2*alpha_1*num_qubits + alpha_2 * (num_qubits - 1))* depth) - 1) + 1
