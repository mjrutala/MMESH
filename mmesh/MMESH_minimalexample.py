#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:27:46 2024

@author: mrutala
"""

config_filepath = '/Users/mrutala/projects/MMESH/config/MMESH_testing.toml'

def minimalexample():
    
    import MMESH as mmesh
    import MMESH_context as mmesh_c
    import plot_TaylorDiagram as TD
    
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
        
    mtraj = mmesh.MultiTrajectory(config_filepath)
    
    for trajectory_name, trajectory in mtraj.trajectories:
        
        starttime = trajectory.starttime
        stoptime = trajectory.stoptime
        
        spacecraft = spacecraftdata.SpacecraftData(trajectory.source)
        spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
        
        trajectory.addData(trajectory.source, spacecraft.data)
        
        
        
    for epoch_name, epoch_info in init['epochs'].items():
        start = epoch_info['start']
        stop  = epoch_info['stop']
        

        
        #  Load spacecraft data
        spacecraft = spacecraftdata.SpacecraftData(epoch_info['source'])
        spacecraft.read_processeddata(start, stop, resolution='60Min')
        pos_TSE = spacecraft.find_StateToEarth()
        
        #!!!!
        #  NEED ROBUST in-SW-subset TOOL! ADD HERE!
        
        #  Change start and stop times to reflect spacecraft limits-- with optional padding
        padding = dt.timedelta(days=8)
        start = spacecraft.data.index[0] - padding
        stop  = spacecraft.data.index[-1] + padding
        
        #  Initialize a trajectory class and add the spacecraft data
        traj0 = mmesh.Trajectory(name=epoch_name)
        traj0.addData(epoch_info['source'], spacecraft.data)
        
        # traj0.set_plotprops('color', 'data', epoch_colors[epoch_name])
        
        #  Read models and add to trajectory
        for model_ref, model_info in init['models'].items():
            
            model = read_SWModel.read_model(model_info['source'], epoch_info['source'], 
                                            start, stop, resolution='60Min')
            
            traj0.addModel(model_ref, model, model_source=model_info['source'])
    
    return