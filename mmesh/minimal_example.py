#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:02:48 2024

@author: mrutala
"""

config_filepath = '/Users/mrutala/projects/MMESH/config/minimal_example.toml'

import numpy as np
import pandas as pd
import datetime
import logging
import multiprocessing as mp

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/') # !!! make relative
import MMESH as mmesh
import MMESH_context as mmesh_c
#import plot_TaylorDiagram as TD

#   Import librarires to read in data and models
import spacecraftdata
import read_SWModel

#   !!! Should be internal
reference_frame = 'SUN_INERTIAL'
observer = 'SUN'
resolution = '60Min'
padding = datetime.timedelta(days=8)


def minimal_example():
    """
    < Good description here >

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #   Create a MultiTrajectory from the config file
    #   This will have trajectories to hold all the data and models
    mtraj = mmesh.MultiTrajectory(config_filepath)
    
    #   Optional: Change logging level 
    # mtraj.logger.level = logging.WARNING
    
    #   Add data and models to each trajectory
    for trajectory_name, trajectory in mtraj.trajectories.items():
        
        starttime = trajectory.start
        stoptime = trajectory.stop
        
        #   This is a custom routine which reads the spacecraft data in as a pandas DataFrame
        #   Any function which returns a pandas DataFrame with columns of 
        #   [n_tot, u_mag, p_dyn, B_mag] will be accepted
        spacecraft = spacecraftdata.SpacecraftData(trajectory.source)
        spacecraft.read_processeddata(starttime, stoptime, resolution=resolution)
        
        breakpoint()
        
        #   Add this DataFrame to the trajectory using the 'addData()' method
        trajectory.addData(trajectory_name, spacecraft.data)
        
        #   Now add the relevant models
        for model_name, model_source in mtraj.model_sources.items():
            
            model = read_SWModel.read_model(model_source, 
                                            trajectory.source, 
                                            trajectory.start - padding, 
                                            trajectory.stop + padding, 
                                            resolution = resolution)
            
            trajectory.addModel(model_name, model, model_source=model_source)
        
        #   Finally, add context   
        #   Here, we make a CSV file to hold the context info
        #   The spatial geometry comes from SPICE (via SpiceyPy)
        #   The Solar Radio Flux comes from Pendicton/Canada
        context_path = mtraj.filepaths['output']/'context/'
        context_fullfilepath = mmesh_c.make_SpacecraftContext_CSV(trajectory.source, starttime - padding, stoptime + padding, filepath = context_path)
        context_df = pd.read_csv(context_fullfilepath)
        context_df['datetime'] = [datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in context_df['datetime']]
        context_df.set_index('datetime', inplace=True)
        context_df.index.name = None
        trajectory.context = context_df
        
        # =============================================================================
        #   Optimize the models via dynamic time warping
        #   Then plot:
        #       - The changes in correlation cooefficient 
        #         (more generically, whatever combination is being optimized)
        #       - The changes on a Taylor Diagram
        # =============================================================================
        #   Binarize the data and models !!!! Explain
        smoothing_widths = trajectory.optimize_ForBinarization('u_mag', threshold=1.05)  
        trajectory.binarize('u_mag', smooth = smoothing_widths, sigma=3)
        
        # #traj0.plot_SingleTimeseries('u_mag', starttime, stoptime, filepath=figurefilepath)
        # traj0.plot_SingleTimeseries('jumps', start, stop, fullfilepath=filepaths['figures']/'figA1_Binarization')
        
        #   Calculate a whole bunch of statistics
        #   !!! connect shifts to padding/resolution in .toml
        dtw_stats = trajectory.find_WarpStatistics('jumps', 'u_mag', 
                                                   shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)
        
        #   Write an equation describing the optimization equation
        #   This can be played with
        def optimization_eqn(df):
            f = df['r'] + (1 - (0.5*df['width_68'])/96.)
            return (f - np.min(f))/(np.max(f)-np.min(f))
        
        #   Plug in the optimization equation
        trajectory.optimize_Warp(optimization_eqn)
        
        #   This plot shows the result of the DTW optimization
        trajectory.plot_OptimizedOffset('jumps', 'u_mag', fullfilepath=mtraj.filepaths['output']/'figures/fig04_DTWIllustration')
    
    # =============================================================================
    #   Now, populate the (fore/hind/now/simul)cast interval
    #   This does not need to be done in a loop-- in fact, it is probably
    #   more likely that this will be done in once for a single cast
    # =============================================================================
    for cast_name, cast in mtraj.cast_intervals.items():
        
        #   Fill with models...
        for model_name, model_source in mtraj.model_sources.items():
            
            model = read_SWModel.read_model(model_source, cast.target, cast.start - padding, cast.stop + padding, resolution = resolution)
            
            cast.addModel(model_name, model, model_source=model_source)
    
        #   Fill with context...
        context_fullfilepath = mmesh_c.make_SpacecraftContext_CSV(cast.target, 
                                                                  cast.start, 
                                                                  cast.stop, 
                                                                  filepath=mtraj.filepaths['output']/'context/')
        cast.context = pd.read_csv(context_fullfilepath, index_col='datetime', parse_dates=True)
        
        #   Add data for comparison purposes only
        #   This data will remain unchanged
        spacecraft = spacecraftdata.SpacecraftData(cast.target)
        spacecraft.read_processeddata(cast.start, cast.stop, resolution=resolution)
        cast.addData('juno', spacecraft.data)
    
    #   Now for the actual casting! Based on a formula describing how the context should inform the uncertainty model:
    #   This translates to:
    #       empirical time delta = a * (target_sun_earth_lat) + b * (solar_radio_flux) + c*(u_mag)
    formula = "empirical_time_delta ~ target_sun_earth_lat + solar_radio_flux + u_mag"
    test = mtraj.linear_regression(formula)
    
    #   with_error? !!!! explain
    mtraj.cast_Models(with_error=True, min_rel_scale = 0.01)
    
    mtraj.ensemble()
    
    # original_TS = mtraj.taylor_statistics(cast=False, with_error=False)
    # final_TS = mtraj.taylor_statistics(cast=True, with_error=True, n_mc=10)
    
    return mtraj

if __name__ == '__main__':
    mp.freeze_support()
    mtraj = minimal_example()
