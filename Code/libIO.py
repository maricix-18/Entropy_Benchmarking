"""
Useful functions to input/output from/to files
"""

import os
import pandas as pd
import json

# GET DIR
def get_experiment_dir(experiment_params):
    return experiment_params.results_dir + experiment_params.backend_params.type + '/'+ experiment_params.protocol_params.name + '_' + experiment_params.circuit_params.choice + '/'

def get_experiment_csv_dir(experiment_params, num_qubits):
    dirname = get_experiment_dir(experiment_params) + str(num_qubits) + 'Q/Data/'
    if experiment_params.protocol_params.name == 'CS' or experiment_params.protocol_params.name == 'Brydges':
        return  dirname + experiment_params.protocol_params.protocol_choice[:6] +'/'
    elif experiment_params.protocol_params.name == 'SWAP':
        return dirname
    
def get_experiment_metrics_dir(experiment_params):
    return get_experiment_dir(experiment_params) + 'metrics/'

def get_experiment_plot_dir(experiment_params, num_qubits):
    if experiment_params.protocol_params.name == 'DensMat':
        return get_experiment_dir(experiment_params) + 'Plots/'
    else:
        return get_experiment_dir(experiment_params) + str(num_qubits) + 'Q/Plots/'


# GET FILENAMES

def get_prefix_num_qubits(circuit_params):
    return 'n%d-%d_' % (circuit_params.num_qubits_min, circuit_params.num_qubits_max)

def get_base_depth(circuit_params):
    return 'D%d-%d-%d_' % (circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step)

def get_base_noise(experiment_params):
    noise_params = experiment_params.noise_params
    if noise_params:
        if experiment_params.protocol_params.name == 'DensMat':
            return 'DP%s-%s_AD%s-%s' % (str(noise_params.p_DP1), str(noise_params.p_DP2),
                                        str(noise_params.p_AD1), str(noise_params.p_AD2))
        else:
            return 'DP%s-%s_AD%s-%s_meas%s-%s'  % (str(noise_params.p_DP1), str(noise_params.p_DP2), 
                                                   str(noise_params.p_AD1), str(noise_params.p_AD2), 
                                                   str(noise_params.p_meas[0][1]), str(noise_params.p_meas[1][0]))
    else:
        return '%s' % (experiment_params.timestamp)

def get_base_filename(experiment_params):
    protocol_params = experiment_params.protocol_params
    circuit_params = experiment_params.circuit_params
    base_depth = get_base_depth(circuit_params)

    if protocol_params.name == 'CS':
        if protocol_params.artif_randomized:
            last_id = protocol_params.artif_randomized[:6]
        else:
            last_id = protocol_params.protocol_choice[:6]
        base_filename = 'M%d_K%d_grps%d_spls%d_%s_' % (protocol_params.M, protocol_params.K, protocol_params.num_groups, protocol_params.num_samples, last_id)
    elif protocol_params.name == 'Brydges':
        base_filename = 'M%d_K%d_grps%d_spls%d_%s_' % (protocol_params.M, protocol_params.K, protocol_params.num_groups, protocol_params.num_samples, protocol_params.protocol_choice[:6])
    elif protocol_params.name == 'SWAP':
        base_filename = 'meas%d_spls%d_' % (protocol_params.num_measurements, protocol_params.num_samples)
    elif protocol_params.name == 'DensMat':
        base_filename = ''
    
    return base_depth + base_filename

def get_metrics_filename(experiment_params):
    prefix = get_prefix_num_qubits(experiment_params.circuit_params)
    base_filename = get_base_filename(experiment_params)
    base_noise = get_base_noise(experiment_params)
    return prefix + base_filename + base_noise + '.json'

def get_CS_csv_filename(experiment_params):
    depth_max = experiment_params.circuit_params.depth_max
    M = experiment_params.protocol_params.M
    K = experiment_params.protocol_params.K
    base_noise = get_base_noise(experiment_params)
    filename = 'D%d_M%d_K%d_' %(depth_max, M, K)

    return filename + base_noise + '.csv'

def get_SWAP_csv_filename(experiment_params):
    depth_max = experiment_params.circuit_params.depth_max
    num_measurements = experiment_params.protocol_params.num_measurements
    base_noise = get_base_noise(experiment_params)
    filename = 'D%d_meas%d_' %(depth_max, num_measurements)

    return filename + base_noise + '.csv'

def get_experiment_filename(experiment_params):
    filename = 'experiment_%s_' %(experiment_params.timestamp)
    filename += 'n%d-%d_D%d.json' % (experiment_params.circuit_params.num_qubits_min, experiment_params.circuit_params.num_qubits_max, experiment_params.circuit_params.depth_max)

    return filename

#GET FULL FILENAMES
def get_metrics_fullfilename(experiment_params):
    dirname = get_experiment_metrics_dir(experiment_params)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = get_metrics_filename(experiment_params)
    return dirname + filename


def get_experiment_fullfilename(experiment_params):
    return get_experiment_dir(experiment_params) + get_experiment_filename(experiment_params)

#INIT & LOAD csv FILES
def init_csv_file(experiment_params, num_qubits):
    # Prepare directory - check if directory exists, if not create it
    dirname = get_experiment_csv_dir(experiment_params, num_qubits)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    protocol = experiment_params.protocol_params.name
    if protocol == 'CS' or protocol == 'Brydges':
        filename = get_CS_csv_filename(experiment_params)
    elif protocol == 'SWAP':
        filename = get_SWAP_csv_filename(experiment_params)
    # Prepare csv file to save results (initialise with header)
    csv_file = dirname + filename
    try:
        open(csv_file, 'r')
    except FileNotFoundError:
        print("File not found. Creating a new file.")
        if protocol == 'CS' or protocol == 'Brydges':
            names = ["depth_index"]+[str(i) for i in range(experiment_params.protocol_params.M)]
        elif protocol == 'SWAP':
            names = ["depth_index","counts"]
        df = pd.DataFrame(names).T
        df.to_csv(csv_file, index=False, header=0)
        print("File created (header initialised).")

    return csv_file

def read_df_from_csv(experiment_params, num_qubits):
    fullfilename = experiment_params.csv_files[str(num_qubits)]
    try:
        df = pd.read_csv(fullfilename)
    except FileNotFoundError:
        print("File not found. Please generate corresponding data first.")
        df = pd.DataFrame()
    return (df)


# LOAD & DUMP json FILES
def dump_to_json(data, jsonfilename):
    with open(jsonfilename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    return

def load_from_json(jsonfilename):
    with open(jsonfilename) as jsonfile:
        data_dict = json.load(jsonfile)
    return data_dict