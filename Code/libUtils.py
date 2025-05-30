import networkx as nx #graphs
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def make_time_stamp():
    current_datetime = datetime.now().strftime("%Y-%m-%d")
    return (str(current_datetime))

# print(make_time_stamp())

def hamming_distance (string1, string2):
    """
    TODO move this function to libUtils
    returns the hamming distance (number of positions where bits differ)
    between string1 and string2
    """
    return sum(s1 != s2 for s1, s2 in zip(string1, string2))
# print("hamming distance : ", hamming_distance('011101', '010011'))

def ternary_list_to_decimal (ternary_list):
    """
    From a ternary list (e.g., [1, 2, 0]) returns the corresponding decimal (e.g., 15)
    """
    decimal = 0
    num_digits = len(ternary_list)
    for digit_index in range(num_digits):
        decimal += ternary_list[num_digits - digit_index - 1] * 3**digit_index
    return decimal
# print(ternary_list_to_decimal([1, 2, 0])) # 15

def bitwise_AND(str1, str2):
    """
    returns the bitwise AND of two strings
    """
    result = ""
    for i in range(len(str1)):
        if str1[i] == '1' and str2[i] == '1':
            result += str1[i]
        else:
            result += '0'
    return (result)

def parity_bit(str):
    """
    returns the parity bit of a string
    """
    result = 0
    for i in range(len(str)):
        if str[i] == '1':
            result += 1
    return (result % 2)

#my_example_outcome = '001101'
#example_outcome_swap_test = parity_bit(bitwise_AND(my_example_outcome[0:3], my_example_outcome[3:6]))


def create_cycle_graph(num_qubits):
    # Example graph from qiskit tutorials (https://qiskit.org/textbook/ch-applications/qaoa.html)
    # -- this is a cycle graph with num_qubits nodes
    nodes = [i for i in range(num_qubits)]
    edges = [(i, (i+1)%num_qubits) for i in range(num_qubits)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # nx.draw(G, with_labels=True, alpha=0.8, node_size=500) # to draw graph

    return G

def get_metrics_specific_width(metrics, num_qubits_min, num_qubits):
    short_metrics = dict()
    for key in metrics.keys():
        short_metrics[key] = metrics[key][num_qubits-num_qubits_min]
    return short_metrics

def compute_stats(samples:list):
    """returns the mean and standard deviation of the list of samples"""
    mean = np.mean(samples)
    std = np.std(samples)
    return (mean, std)

def renyi_entropy_from_purity (purity):
    """returns the value of the second order Renyi entropy corresponding to purity value purity"""
    return (-1*np.log2(purity))


def get_cumulative_of_list (my_list):
    """
    Compute the cumulative of the list my_list
    """
    return [sum(my_list[:i+1]) for i in range(len(my_list))]
# print("cumulative : ", get_cumulative_of_list([0, 2, 6, 3, 1])) # expect [0, 2, 8, 11, 12]

def get_sum_list (my_list):
    return [sum(my_list) for _ in range(len(my_list))]
# print("sum list : ", get_sum_list([1,3,2])) # expect [6, 6, 6]

def get_cumulative_error_of_list (my_list):
    """
    Compute the cumulative error of the list my_list
    """
    return [sum(my_list[i+1:]) for i in range(len(my_list))]
# print("cumulative error : ", get_cumulative_error_of_list([0, 2, 6, 3, 1])) # expect [12, 10, 4, 1, 0]

def get_dataframe_specific_depth(df, wanted_depth_index):
    sub_df = df.loc[ df ["depth_index"] == wanted_depth_index]
    #print("sub dataframe: \n", sub_df)
    if sub_df.empty:
        print("Please check the wanted_depth_index as it is not in the original dataframe df.")
        return (None)
    return (sub_df)