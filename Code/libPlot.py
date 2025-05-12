"""
A library dedicated to functions related to plots
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from libIO import  get_experiment_plot_dir, get_base_filename, get_base_noise, get_prefix_num_qubits
from libUtils import renyi_entropy_from_purity, get_metrics_specific_width

# =============================================================================================================
# Utility functions 

def compute_xticks(depth_tab, depth_step):
    if max(depth_tab)-min(depth_tab)+1<10:
        xticks = range(min(depth_tab), max(depth_tab)+1, depth_step)
    else:
        xticks = range(min(depth_tab), max(depth_tab)+1, int((max(depth_tab)-min(depth_tab)+1)/10))
    return xticks

def compute_depth_tab(depth_min, depth_max, depth_step):
    return np.arange(depth_min, depth_max+1, depth_step)

def compute_depth_tab_more_points(depth_min, depth_max):
    return np.linspace(depth_min, depth_max+1, 1000)

def compute_filename(output_dir, metric, filename):
    return output_dir+'/'+metric+'-'+filename

def plot_params(xlabel, ylabel, xticks, ylim_max, save_fig, filename):
    plt.xlabel(xlabel)
    plt.xticks(xticks)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0, top=ylim_max)
    plt.legend()
    if save_fig:
        plt.savefig(filename)

# =============================================================================================================
# Main functions for Shadows and SWAP

def plot_metric(metric:str, metric_mean:list, metric_std:list, metric_exact:list, num_qubits:int, depth_min:int, depth_max:int, depth_step:int, show_fully_mixed_state:bool, compute_exact:bool, save_fig:bool, output_dir:str, filename:str):
    plt.figure()
    depth_tab = np.arange(depth_min, depth_max+1, depth_step)
    plt.errorbar(x=depth_tab, y=metric_mean, yerr=metric_std, ecolor="red", capsize=3, ls='none', label='estimate')
    if compute_exact:
        plt.plot(depth_tab, metric_exact, color="black", label='exact')
    if show_fully_mixed_state:
        if metric == 'Purity':
            plt.axhline(y=1/2**num_qubits, color='black', linestyle='-.', label='maximally mixed state')
        elif metric == 'Renyi-2 entropy':
            plt.axhline(y=renyi_entropy_from_purity(1/2**num_qubits), color='black', linestyle='-.', label='maximally mixed state')
        elif metric == 'Renyi-2 entropy density':
            plt.axhline(y=renyi_entropy_from_purity(1/2**num_qubits)/num_qubits, color='black', linestyle='-.', label='maximally mixed state')
    plt.xlabel('Depth')
    xticks = compute_xticks(depth_tab, depth_step)
    plt.xticks(xticks)
    plt.ylabel(metric)
    if metric == 'Renyi-2 entropy':
        plt.ylim(bottom=0, top=num_qubits+0.1)
    else:
        plt.ylim(bottom=0, top=1.1)
    plt.legend()
    if save_fig:
        if metric == 'Renyi-2 entropy density':
            metric = 'R2d'
        elif metric == 'Renyi-2 entropy':
            metric = 'R2'
        elif metric == 'Purity':
            metric = 'pur'
        plt.savefig(output_dir+'/'+metric+'-'+filename)
    plt.show()
    return()

def plot_metrics(metrics, metrics_exact, experiment_params, show_fully_mixed_state, compute_exact, save):

    circuit_params = experiment_params.circuit_params
     
    filename = get_base_filename(experiment_params) + get_base_noise(experiment_params) + '.pdf' #svg'

    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1):
        print('\n Number of qubits = ', num_qubits)

        # Extract purity, renyi entropy density (average, std and exact)
        short_metrics = get_metrics_specific_width(metrics, circuit_params.num_qubits_min, num_qubits)
        short_metrics_exact = get_metrics_specific_width(metrics_exact, circuit_params.num_qubits_min, num_qubits)
        
        # PLOTS ==========================================================================
        if save:
            # Prepare directory - check if directory exists, if not create it
            output_dir = get_experiment_plot_dir(experiment_params, num_qubits)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            output_dir = None

        # Purity
        plot_metric('Purity', short_metrics['all_pur_mean_diff_n'], short_metrics['all_pur_std_diff_n'], short_metrics_exact['all_pur_diff_n'], 
                    num_qubits, circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step, show_fully_mixed_state, compute_exact, save, output_dir, filename)

        # Renyi-2 entropy
        plot_metric('Renyi-2 entropy', [r2d*num_qubits for r2d in short_metrics['all_R2d_mean_diff_n']], 
                    [r2d_std*num_qubits for r2d_std in short_metrics['all_R2d_std_diff_n']], 
                    [r2d_exact*num_qubits for r2d_exact in short_metrics_exact['all_R2d_diff_n']], num_qubits, 
                    circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step, show_fully_mixed_state, compute_exact, save, output_dir, filename)
        
        # Renyi-2 entropy density
        plot_metric('Renyi-2 entropy density', short_metrics['all_R2d_mean_diff_n'], short_metrics['all_R2d_std_diff_n'], short_metrics_exact['all_R2d_diff_n'], 
                    num_qubits, circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step, show_fully_mixed_state, compute_exact, save, output_dir, filename)

# =============================================================================================================
# Main functions for DensMat

def plot_metric_DensMat(metric, experiment_params, metrics, plot_dir, show_maximally_mixed_state, save):
    if metric == 'Renyi-2 entropy density':
        metric_short = 'R2d'
    elif metric == 'von Neumann entropy density':
        metric_short = 'vNd'
    elif metric == 'Purity':
        metric_short = 'pur'

    circuit_params = experiment_params.circuit_params
    noise_params = experiment_params.noise_params

    plt.figure()
    d = np.arange(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step)
    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1, circuit_params.num_qubits_step):
        if num_qubits == 1 and not metric == 'von Neumann entropy density':
            if metric == 'Purity':
                metric_lim = (1/4)*(1+noise_params.p_AD1/(1 - (1 - noise_params.p_DP1)*(1 - noise_params.p_AD1)))**2 + (1/4)*(1/(1 + noise_params.p_AD1/(noise_params.p_DP1 * (1 - noise_params.p_AD1))))**2
            if metric == 'Renyi-2 entropy density':
                pur_lim = (1/4)*(1+noise_params.p_AD1/(1 - (1 - noise_params.p_DP1)*(1 - noise_params.p_AD1)))**2 + (1/4)*(1/(1 + noise_params.p_AD1/(noise_params.p_DP1 * (1 - noise_params.p_AD1))))**2    
                metric_lim = -np.log2(pur_lim)/num_qubits
            plt.plot(d, [metric_lim]*len(d), linestyle='--', color='black', label='predicted limit')
        plt.plot(d, metrics['all_'+metric_short+'_diff_n'][int((num_qubits-circuit_params.num_qubits_min)/circuit_params.num_qubits_step)], label='$n = $' + str(num_qubits))
    if show_maximally_mixed_state:
        if metric == 'Purity':
            plt.axhline(y=1/2**num_qubits, color='black', linestyle='-.', label='maximally mixed state')
        elif metric == 'Renyi-2 entropy density':
            plt.axhline(y=1, color='black', linestyle='-.', label='maximally mixed state')
    plt.legend()
    plt.xlabel('Depth')
    xticks = compute_xticks(d, circuit_params.depth_step)
    plt.xticks(xticks)
    plt.ylabel(metric)
    #plt.ylim(0,1.1)
    if save:
        filename = metric_short + '-' + get_prefix_num_qubits(circuit_params) + 'D' + str(circuit_params.depth_max) + '_' + get_base_noise(experiment_params) + '.pdf'
        plt.savefig(plot_dir + filename)
    plt.show()
    return

def plot_metrics_DensMat (experiment_params, metrics, show_maximally_mixed_state, save=True):
    """
    Plots metrics (von Neumann entropy, purity and/or Renyi entropy) from the classical simulation 
    with density matrix simulation as a function of circuit depth and for different number of qubits/
    circuit widths
    """
    plot_dir = get_experiment_plot_dir(experiment_params, 0)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_metric_DensMat('von Neumann entropy density', experiment_params, metrics, plot_dir, show_maximally_mixed_state, save)
    plot_metric_DensMat('Purity', experiment_params, metrics, plot_dir, show_maximally_mixed_state, save)
    plot_metric_DensMat('Renyi-2 entropy density', experiment_params, metrics, plot_dir, show_maximally_mixed_state, save)

    return