#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:38:42 2023

@author: mrutala

Options:
    Make two programs: one which converts data into the required statistical
    measures, and one which plots these measures
        In which case, we could call the plot call outside of this
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def make_NaNFree(*arrays):
    
    nans_arr = np.zeros(len(arrays[0]))
    for arr in arrays:
        try:
            nans_arr += np.isnan(arr).astype(int)
        except:
            breakpoint()
    
    nans_arr[nans_arr > 1] = 1
    nans_arr = nans_arr.astype(bool)
    
    output_arrs = []
    for arr in arrays:
        arr = np.array(arr)
        output_arrs.append(arr[~nans_arr])
    
    return tuple(output_arrs)

def find_TaylorStatistics(test_data = 0, ref_data = 0, verbose=True):
    
    N = float(len(test_data))
    if N != float(len(ref_data)):
        print("'test_data' and 'ref_data' must have the same length.")
        return
    
    test_arr = np.array(test_data, dtype='float64')
    ref_arr = np.array(ref_data, dtype='float64')
    test_arr, ref_arr = make_NaNFree(test_arr, ref_arr)
    N = float(len(test_arr))
    if (len(test_arr) == 0.) or (len(ref_arr) == 0.):
        if verbose: 
            print('One array is all NaNs. Returning...')
        return (0., 0.), 0.
    
    test_mean = np.mean(test_arr)
    test_std = np.std(test_arr)
    
    
    ref_mean = np.mean(ref_arr)
    ref_std = np.std(ref_arr)
    
    correlation_r = np.sum((test_arr - test_mean) * (ref_arr - ref_mean)) * (1./N) * (1./(test_std * ref_std))
    
    centered_RMS_diff = np.sqrt((1./N) * np.sum(((test_arr - test_mean) - (ref_arr - ref_mean))**2))
    
    return((correlation_r, test_std), centered_RMS_diff)

def find_TaylorDiagramCoordinates(ref, *tests):
    
    df = pd.DataFrame(ref, columns=['ref'])
    for i, test in enumerate(tests):
        df['{0:02d}'.format(i)] = test
    
    df = df.dropna(axis='index')
    
    results_list = [(0, np.std(df['ref'].values), 0)]
    for column_name, column in df.items():
        if column_name != 'ref':
            (r, sig), rmsd = find_TaylorStatistics(column.values, df['ref'].values)
            results_list.append((r, sig, rmsd))
    
    return results_list

def plot_TaylorDiagram(test_data, ref_data, fig=None, ax=None, **plt_kwargs):
    
    #   Make sure the inputs are NaN free
    test_data = np.array(test_data, dtype='float64')
    ref_data = np.array(ref_data, dtype='float64')
    test_data, ref_data = make_NaNFree(test_data, ref_data)
    
    #   Get the figure and axis
    fig, ax = init_TaylorDiagram(np.std(ref_data), fig=fig, ax=ax)
    
    #   Calculate r, stddev, and RMS of the test relative to the reference
    (r, sig), RMS = find_TaylorStatistics(test_data, ref_data)
        
    # Plot model point
    ax.scatter(np.arccos(r), sig, **plt_kwargs)
    
    return(fig, ax)
   
def init_TaylorDiagram(ref_std, fig=None, ax=None, half=False, r_label='', theta_label='',
                       **plt_kwargs):

    #   If given nothing, make a figure and polar axis
    #   If given a figure, make a polar axis
    #   If given both, make nothing
    ret = []
    if fig == None:
        fig = plt.figure()
        ret.append(fig)
    if ax == None:
        ax = fig.add_subplot(111, projection='polar')
        ret.append(ax)
    
    #  Centered RMS difference circles, centered on reference
    ax.scatter(0, ref_std, marker='o', color='black')
    ax.plot(np.linspace(0, np.pi, 180), np.zeros(180)+ref_std, color='black', linewidth=1.5, linestyle='--', zorder=-2)
    
    RMS_r = np.arange(ref_std/3., (3 + 1/3.)*ref_std, ref_std/3.)
    for r in RMS_r:
        x = r * np.cos(np.linspace(0, 2*np.pi, 100))
        y = r * np.sin(np.linspace(0, 2*np.pi, 100))
        x2 = x + (ref_std)
        y2 = y #  Reference point is on y=0 by definition
        ax.plot(np.arctan2(y2,x2), np.sqrt(x2**2 + y2**2), color='gray', linestyle=':', zorder=-4)
    ax.set_rlim([0, 2.0*ref_std])
    
    #  [0,180] includes both positive and negative correlations
    ax.set_thetamin(0)
    if half: 
        ax.set_thetamax(90)
        r_label_pos = [0.5, -0.1]
        theta_ticks = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
        theta_label_pos = [0.85, 0.85]
        theta_label_rot = -45
    else:
        ax.set_thetamax(180)
        r_label_pos = [0.5, -0.1]
        theta_ticks = [-0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0, 
                       0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
        theta_label_pos = [0.5, 0.85]
        theta_label_rot = 0
        
    ax.set_xticks([np.arccos(ang) for ang in theta_ticks])
    ax.set_xticklabels(theta_ticks)
    ax.tick_params(axis='x', which='major', pad=-2, direction='inout')
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment('left')
    
    #   Set title and labels
    #   Need to set y in one command and x in another-- pyplot REALLY doesn't
    #   want to let you move the axes labels
    if r_label == '':
        r_label = r'Standard Deviation ($\sigma$)'
    
    if theta_label == '':
        theta_label = 'Pearson Correlation Coefficient (r)'
    
    ax.annotate(theta_label, theta_label_pos, xycoords='axes fraction', 
                fontsize=plt.rcParams["figure.labelsize"],
                rotation=theta_label_rot, ha='center', va='center')
    
    ax.annotate(r_label, r_label_pos, xycoords='axes fraction',
                fontsize=plt.rcParams["figure.labelsize"],
                ha='center', va='top')
    
    #ax.set_ylabel('Pearson Correlation Coefficient (r)', y=0.85, rotation=ylabel_rot)
    #ax.yaxis.set_label_coords(0.5, 0.9)
   
    ax.grid(visible=True, zorder=-100)
    
    for element in [ax.xaxis.label, ax.yaxis.label]:
        element.set_fontsize(plt.rcParams["figure.labelsize"])
    #          # ax.get_xticklabels() + ax.get_yticklabels()
    
    if len(ret) > 0:
        return ret

def example_TaylorDiagram():
    
    # Example usage
    ref_data = np.linspace(0, 100, 5)
    test_data = np.linspace(0, 200, 5) + np.random.normal(0, 5, [5])
    stats, RMS = find_TaylorStatistics(test_data, ref_data)
    
    # Create Taylor diagram plot
    #fig = plt.figure(figsize=(9,8))
    #ax = fig.add_subplot(111, projection='polar')
    
    ax = plot_TaylorDiagram(test_data, ref_data, color='red', marker='X', markersize=8)
    
    ax.set_ylim((0, 80))
    
    test_data2 = np.logspace(0, 2, 5) + np.random.normal(0, 5, [5])
    ax = plot_TaylorDiagram(test_data2, ref_data, ax=ax, marker='X', color='orange', markersize=8)
    
    plt.show()

