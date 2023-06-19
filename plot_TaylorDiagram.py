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

def find_TaylorStatistics(test_data, ref_data):
    
    N = float(len(test_data))
    if N != float(len(ref_data)):
        print("'test_data' and 'ref_data' must have the same length.")
        return
    
    
    test_data = np.array(test_data)
    test_mean = np.mean(test_data)
    test_std = np.std(test_data)
    
    ref_data = np.array(ref_data)
    ref_mean = np.mean(ref_data)
    ref_std = np.std(ref_data)
    
    correlation_r = np.sum((test_data - test_mean) * (ref_data - ref_mean)) * (1./N) * (1./(test_std * ref_std))
    
    centered_RMS_diff = np.sqrt((1./N) * np.sum(((test_data - test_mean) - (ref_data - ref_mean))**2))
    
    return((correlation_r, test_std), centered_RMS_diff)

def plot_TaylorDiagram(test_data, ref_data, ax=None, **plt_kwargs):
    
    #  If no axis is included, look for a current one or make one
    if ax == None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Set title and labels
        ax.set_title('Taylor Diagram')
        ax.set_xlabel('Standard Deviation')
        
        #  Default to include negative correlations
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        theta_ticks = [-0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0, 
                       0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
        ax.set_xticks([np.arccos(ang) for ang in theta_ticks])
        ax.set_xticklabels(theta_ticks)
        
        #  Centered RMS difference circles, centered on reference
        #plt.autoscale(False)
        RMS_r = np.arange(10, 120, 20)
        for r in RMS_r:
            x = r * np.cos(np.linspace(0, 2*np.pi, 100))
            y = r * np.sin(np.linspace(0, 2*np.pi, 100))
            x2 = x + (np.std(ref_data))
            y2 = y #  Reference point is on y=0 by definition
            ax.plot(np.arctan2(y2,x2), np.sqrt(x2**2 + y2**2), color='gray', linestyle=':')
            
        #plt.autoscale(True)
        
        ax.plot(0, np.std(ref_data), marker='o', color='black', markersize=12)

        # Show plot
        #plt.show()
    
    taylor_statistics, RMS = find_TaylorStatistics(test_data, ref_data)
        
    # Plot model point
    correlation_r, test_std = taylor_statistics
    ax.plot(np.arccos(correlation_r), test_std, **plt_kwargs)
    
    return(ax)
   

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
