#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:23:16 2023

@author: mrutala

Process in-situ measurements of solar wind dynamic pressure and velocity
in order to return a normalized function showing where jumps in either/both
occur

This is done using a normalized derivative
"""
import datetime as dt
import pandas as pd
import numpy as np

def plot_StackedTimeseries(spacecraft_name, model_names, starttime, stoptime):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    import spacecraftdata
    import read_SWModel
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime)
    spacecraft.data = spacecraft.data.resample("15Min").mean()
    
    #  Get starttime and stoptime from the dataset
    #starttime = spacecraft.data.index[0]
    #stoptime = spacecraft.data.index[-1]
    
    #  Read models
    models = dict.fromkeys(model_names, None)
    for model in models.keys():
        models[model] = read_SWModel.choose(model, spacecraft_name, 
                                           starttime, stoptime)
    colors = ['#E63946', '#FFB703', '#46ACAF', '#386480', '#192F4D']
    model_colors = dict(zip(model_names, colors[0:len(model_names)]))
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(figsize=(8,6), nrows=4, sharex=True)
        
        #mask = np.isfinite(spacecraft.data['u_mag']) 
        axs[0].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', label='Juno/JADE (Wilson+ 2018)')
        axs[0].set(ylim=[350,550], ylabel=r'$u_{mag} [km s^{-1}]$')
        axs[1].plot(spacecraft.data.dropna(subset='n_proton').index, 
                    spacecraft.data.dropna(subset='n_proton')['n_proton'], 
                    color='gray')
        axs[1].set(ylim=[1e-3, 5e0], yscale='log', ylabel=r'$n_{proton} [cm^{-3}]$')
        axs[2].plot(spacecraft.data.dropna(subset='p_dyn').index, 
                    spacecraft.data.dropna(subset='p_dyn')['p_dyn'], 
                    color='gray')     
        axs[2].set(ylim=[1e-3, 5e0], yscale='log', ylabel=r'$p_{dyn} [nP]$')
        axs[3].plot(spacecraft.data.dropna(subset='B_mag').index, 
                    spacecraft.data.dropna(subset='B_mag')['B_mag'], 
                    color='gray')
        axs[3].set(ylim=[0,3], ylabel=r'$B_{mag} [nT]$')
        
        for model, model_info in models.items():
            if 'u_mag' in model_info.columns: 
                axs[0].plot(model_info.index, model_info['u_mag'], 
                            color=model_colors[model], label=model)
            if 'n_proton' in model_info.columns: 
                axs[1].plot(model_info.index, model_info['n_proton'],
                            color=model_colors[model], label=model)
            if 'p_dyn' in model_info.columns: 
                axs[2].plot(model_info.index, model_info['p_dyn'],
                            color=model_colors[model], label=model)
            if 'B_mag' in model_info.columns: 
                axs[3].plot(model_info.index, model_info['B_mag'],
                            color=model_colors[model], label=model)
          
        axs[0].legend(ncol=3, bbox_to_anchor=[0.0,1.05,1.0,0.15], loc='lower left', mode='expand')
        
        axs[0].set_xlim((starttime, stoptime))
        axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axs[0].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
        axs[3].set_xlabel('Day of Year 2016')
        
        fig.align_ylabels()
        plt.savefig('figures/Timeseries_Juno_Stacked.png', dpi=300)
        plt.show()
        return()

def plot_NormalizedTimeseries(spacecraft_name, model_names, starttime, stoptime):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    import spacecraftdata
    import read_SWModel
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime)
    spacecraft.data = spacecraft.data.resample("15Min").mean()
    spacecraft.data = (spacecraft.data - spacecraft.data.min()) / (spacecraft.data.max()-spacecraft.data.min())
    
    #  Get starttime and stoptime from the dataset
    #starttime = spacecraft.data.index[0]
    #stoptime = spacecraft.data.index[-1]
    
    #  Read models
    models = dict.fromkeys(model_names, None)
    for model in models.keys():
        model_in = read_SWModel.choose(model, spacecraft_name, 
                                       starttime, stoptime)
        models[model] = (model_in - model_in.min()) / (model_in.max() - model_in.min())
    colors = ['#E63946', '#FFB703', '#46ACAF', '#386480', '#192F4D']
    model_colors = dict(zip(model_names, colors[0:len(model_names)]))
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(figsize=(8,6), nrows=4, sharex=True)
        
        #mask = np.isfinite(spacecraft.data['u_mag']) 
        axs[0].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', label='Juno/JADE (Wilson+ 2018)')
        axs[0].set(ylim=[0,1], ylabel=r'$u_{mag} [arb.]$')
        axs[1].plot(spacecraft.data.dropna(subset='n_proton').index, 
                    spacecraft.data.dropna(subset='n_proton')['n_proton'], 
                    color='gray')
        axs[1].set(ylim=[1e-3, 1e0], yscale='log', ylabel=r'$n_{proton} [arb.]$')
        axs[2].plot(spacecraft.data.dropna(subset='p_dyn').index, 
                    spacecraft.data.dropna(subset='p_dyn')['p_dyn'], 
                    color='gray')     
        axs[2].set(ylim=[1e-3, 1e0], yscale='log', ylabel=r'$p_{dyn} [arb.]$')
        axs[3].plot(spacecraft.data.dropna(subset='B_mag').index, 
                    spacecraft.data.dropna(subset='B_mag')['B_mag'], 
                    color='gray')
        axs[3].set(ylim=[0,1], ylabel=r'$B_{mag} [arb.]$')
        
        for model, model_info in models.items():
            if 'u_mag' in model_info.columns: 
                axs[0].plot(model_info.index, model_info['u_mag'], 
                            color=model_colors[model], label=model)
            if 'n_proton' in model_info.columns: 
                axs[1].plot(model_info.index, model_info['n_proton'],
                            color=model_colors[model], label=model)
            if 'p_dyn' in model_info.columns: 
                axs[2].plot(model_info.index, model_info['p_dyn'],
                            color=model_colors[model], label=model)
            if 'B_mag' in model_info.columns: 
                axs[3].plot(model_info.index, model_info['B_mag'],
                            color=model_colors[model], label=model)
          
        axs[0].legend(ncol=3, bbox_to_anchor=[0.0,1.05,1.0,0.15], loc='lower left', mode='expand')
        
        axs[0].set_xlim((starttime, stoptime))
        axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axs[0].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
        axs[3].set_xlabel('Day of Year 2016')
        
        fig.align_ylabels()
        plt.savefig('figures/Timeseries_Juno_Normalized.png', dpi=300)
        plt.show()
        return()
  
def plot_JumpTimeseries(spacecraft_name, model_names, starttime, stoptime):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    import spacecraftdata
    import read_SWModel
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime)
    spacecraft.data = spacecraft.data.resample("15Min").mean()
    spacecraft.data = (spacecraft.data - spacecraft.data.min()) / (spacecraft.data.max()-spacecraft.data.min())
    #spacecraft.data = find_RollingDerivativeZScore(spacecraft.data, 'u_mag', 4)
    
    
    #  Get starttime and stoptime from the dataset
    #starttime = spacecraft.data.index[0]
    #stoptime = spacecraft.data.index[-1]
    
    #  Read models
    models = dict.fromkeys(model_names, None)
    for model in models.keys():
        model_in = read_SWModel.choose(model, spacecraft_name, 
                                       starttime, stoptime)
        models[model] = model_in
        
    colors = ['#E63946', '#FFB703', '#46ACAF', '#386480', '#192F4D']
    model_colors = dict(zip(model_names, colors[0:len(model_names)]))
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(figsize=(8,6), nrows=4, sharex=True)
        
        #mask = np.isfinite(spacecraft.data['u_mag']) 
        axs[0].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', label='Juno/JADE (Wilson+ 2018)')
        axs[0].set(ylim=[0,1], ylabel=r'$u_{mag} [arb.]$')
        axs[1].plot(spacecraft.data.dropna(subset='n_proton').index, 
                    spacecraft.data.dropna(subset='n_proton')['n_proton'], 
                    color='gray')
        axs[1].set(ylim=[-1,4], ylabel=r'$n_{proton} [arb.]$')
        axs[2].plot(spacecraft.data.dropna(subset='p_dyn').index, 
                    spacecraft.data.dropna(subset='p_dyn')['p_dyn'], 
                    color='gray')     
        axs[2].set(ylim=[-1,4], yscale='log', ylabel=r'$p_{dyn} [arb.]$')
        axs[3].plot(spacecraft.data.dropna(subset='B_mag').index, 
                    spacecraft.data.dropna(subset='B_mag')['B_mag'], 
                    color='gray')
        axs[3].set(ylim=[-1,4], ylabel=r'$B_{mag} [arb.]$')
        
        for index, (model, model_info) in enumerate(models.items()):
            print(model)
            if ('u_mag' in model_info.columns) and len(model_info['u_mag'] > 0):
                dummy = find_RollingDerivativeZScore(model_info, 'u_mag', 4)
                points = np.where(dummy['smooth_ddt_u_mag_zscore'] > 4, 0.2*(index+1), None)
                axs[0].plot(model_info.index, points, 
                            color=model_colors[model], marker='o', markersize=8, linestyle='None', label=model)
            if ('n_proton' in model_info.columns) and len(model_info['n_proton'] > 0):
                dummy = find_RollingDerivativeZScore(model_info, 'n_proton', 4)
                points = np.where(dummy['smooth_ddt_n_proton_zscore'] > 4, 0.2*(index+1), None)
                axs[1].plot(model_info.index, points,
                            color=model_colors[model], marker='|', markersize=16, linestyle='None', label=model)
            if ('p_dyn' in model_info.columns) and len(model_info['p_dyn'] > 0):
                dummy = find_RollingDerivativeZScore(model_info, 'p_dyn', 4)
                points = np.where(dummy['smooth_ddt_p_dyn_zscore'] > 4, 0.2*(index+1), None)
                axs[2].plot(model_info.index, points,
                            color=model_colors[model], marker='|', markersize=16, linestyle='None', label=model)
            if ('B_mag' in model_info.columns) and len(model_info['B_mag'] > 0):
                dummy = find_RollingDerivativeZScore(model_info, 'B_mag', 4)
                points = np.where(dummy['smooth_ddt_B_mag_zscore'] > 4, 0.2*(index+1), None)
                axs[3].plot(model_info.index, points,
                            color=model_colors[model], marker='|', markersize=8, linestyle='None', label=model)
          
        axs[0].legend(ncol=3, bbox_to_anchor=[0.0,1.05,1.0,0.15], loc='lower left', mode='expand')
        
        axs[0].set_xlim((starttime, stoptime))
        axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axs[0].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
        axs[3].set_xlabel('Day of Year 2016')
        
        fig.align_ylabels()
        plt.savefig('figures/Timeseries_Juno_Jumps.png', dpi=300)
        plt.show()
        return()
    
def plot_SpreadTimeseries(spacecraft_name, model_names, starttime, stoptime):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    import spacecraftdata
    import read_SWModel
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime)
    spacecraft.data = spacecraft.data.resample("15Min").mean()
    
    #  Get starttime and stoptime from the dataset
    #starttime = spacecraft.data.index[0]
    #stoptime = spacecraft.data.index[-1]
    
    #  Read models
    models = dict.fromkeys(model_names, None)
    for model in models.keys():
        model_in = read_SWModel.choose(model, spacecraft_name, 
                                       starttime, stoptime)
        models[model] = model_in
        
    colors = ['#E63946', '#FFB703', '#46ACAF', '#386480', '#192F4D']
    model_colors = dict(zip(model_names, colors[0:len(model_names)]))
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(figsize=(8,6), nrows=4, sharex=True)
        
        #mask = np.isfinite(spacecraft.data['u_mag']) 
        axs[0].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', alpha=0.5, label='Juno/JADE (Wilson+ 2018)')
        axs[0].set(ylim=[350,550], ylabel=r'$u_{mag}$ [km s$^{-1}$]')
        axs[1].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', alpha=0.5)
        axs[1].set(ylim=[350,550], ylabel=r'$u_{mag}$ [km s$^{-1}$]')
        axs[2].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', alpha=0.5)
        axs[2].set(ylim=[350,550], ylabel=r'$u_{mag}$ [km s$^{-1}$]')
        axs[3].plot(spacecraft.data.dropna(subset='u_mag').index, 
                    spacecraft.data.dropna(subset='u_mag')['u_mag'],
                    color='gray', alpha=0.5)
        axs[3].set(ylim=[350,550], ylabel=r'$u_{mag}$ [km s$^{-1}$]')
        
        for indx, (model, model_info) in enumerate(models.items()):
            if 'u_mag' in model_info.columns: 
                axs[indx].plot(model_info.index, model_info['u_mag'], 
                               color=model_colors[model], label=model)
        
        # labels, handles = list(zip(*[ax.get_legend_handles_labels() for ax in axs]))
        
        # axs[0].legend(handles=handles, labels=labels, ncol=3, bbox_to_anchor=[0.0,1.05,1.0,0.15], loc='lower left', mode='expand')
        
        axs[0].set_xlim((starttime, stoptime))
        axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axs[0].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
        axs[3].set_xlabel('Day of Year 2016')
        
        fig.align_ylabels()
        fig.legend(ncol=3, bbox_to_anchor=[0.1,0.9,0.8,0.1], loc='lower left', mode='expand')
        plt.savefig('figures/Timeseries_Juno_Spread_u_mag.png', dpi=300)
        plt.show()
        return()


def plot_histograms():
    
    return()

def SWData_TaylorDiagram():
    #import datetime as dt
    #import pandas as pd
    #import numpy as np
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    
    import sys
    sys.path.append('/Users/mrutala/projects/SolarWindProp/')
    #import read_SWData
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    starttime = dt.datetime(2016, 5, 1)
    stoptime = dt.datetime(2016, 7, 1)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    tag =  'u_mag'  #  solar wind property of interest
    
    juno = spacecraftdata.SpacecraftData('Juno')
    juno.read_processeddata(starttime, stoptime)
    juno.find_state(reference_frame, observer)
    
    tao = read_SWModel.TaoSW('Juno', starttime, stoptime)
    vogt = read_SWModel.VogtSW('Juno', starttime, stoptime)
    mich = read_SWModel.MSWIM2DSW('Juno', starttime, stoptime)
    huxt = read_SWModel.HUXt('Jupiter', starttime, stoptime)
    
    fig, axs = plt.subplots(nrows=3)
    
    axs[0].plot(juno.data.index, juno.data[tag])
    axs[0].plot(tao.index, tao[tag])
    axs[0].plot(vogt.index, vogt[tag])
    axs[0].plot(mich.index, mich[tag])
    if tag == 'u_mag': axs[0].plot(huxt.index, huxt[tag])
    
    tao_reindexed = tao.reindex(juno.data.index, method='nearest')
    vogt_reindexed = vogt.reindex(juno.data.index, method='nearest')
    mich_reindexed = mich.reindex(juno.data.index, method='nearest')
    huxt_reindexed = huxt.reindex(juno.data.index, method='nearest')
    
    axs[1].plot(juno.data.index, juno.data[tag])
    axs[1].plot(tao_reindexed.index, tao_reindexed[tag])
    axs[1].plot(vogt_reindexed.index, vogt_reindexed[tag])
    axs[1].plot(mich_reindexed.index, mich_reindexed[tag])
    if tag == 'u_mag': axs[1].plot(huxt_reindexed.index, huxt_reindexed[tag])
    
    plt.show()
    
    #fig = plt.figure(figsize=(16,9))
    #ax1 = plt.subplot(211, projection='polar')
    #ax2 = plt.subplot(212)
    #axs = [ax1, ax2]
    
    """
    It would be really ideal to write plot_TaylorDiagram such that there's a 
    function I could call here which "initializes" ax1 (axs[0]) to be a 
    Taylor Diagram-- i.e., I set aside the space in the figure, and then give 
    that axis to a program which adds the axes, labels, names, etc.
    Everything except the data, basically
    
    I guess it would make sense to additionally add a function to plot the 
    RMS difference, rather than doing it with everything else
    """
    
    fig, ax = TD.plot_TaylorDiagram(tao_reindexed[tag], juno.data[tag], color='red', marker='X', markersize='0')
    
    model_dict = {'tao':tao, 'vogt':vogt, 'mich':mich, 'huxt':huxt}
    marker_dict = {'tao': '^', 'vogt': 'v', 'mich':'P', 'huxt':'X'}
    label_dict = {'tao': 'Tao (2005)', 'vogt': 'SWMF-OH (BU)', 'mich':'MSWIM2D', 'huxt':'HUXt (@ Jupiter)'}
    shifted_stats = {'tao':list(), 'vogt':list(), 'mich':list(), 'huxt':list()}
    
    shifts = np.arange(-72, 72+6, 6)  #  -/+ hours of shift for the timeseries
    for model_key in model_dict.keys():
        focus_model = model_dict[model_key]
        print(model_key)
        for shift in shifts:
            shifted_model = focus_model.copy()
            shifted_model.index = shifted_model.index + pd.Timedelta(shift, 'hours')
            shifted_model_reindexed = shifted_model.reindex(juno.data.index, method='nearest')
            stats, rmse = TD.find_TaylorStatistics(shifted_model_reindexed[tag], juno.data[tag])
            shifted_stats[model_key].append(stats)
            
            if shift == 0.0:
                unshifted_plot = ax.plot(np.arccos(stats[0]), stats[1], marker='x', markersize=12, color='cyan')
         
        coords_from_stats = [(np.arccos(e1), e2) for e1, e2 in shifted_stats[model_key]]
        focus_model_plot = ax.scatter(*zip(*coords_from_stats), 
                                      c=shifts, cmap='plasma', s=24, marker=marker_dict[model_key], 
                                      label=label_dict[model_key])
             
    ax.set_ylim(0,60)
    ax.legend()
    plt.margins(0)
    plt.colorbar(focus_model_plot, location='bottom', orientation='horizontal',
                 label=r'$\Delta t$ [hours]', fraction=0.05, pad=-0.1,
                 ticks=np.arange(-72, 72+12, 12))
    plt.tight_layout()
    plt.savefig('JunoModel_u_mag_TaylorDiagram.png', dpi=200)
    plt.show()
    
    return(shifted_stats)
    
def SWData_TaylorDiagram_Binary():
    #import datetime as dt
    #import pandas as pd
    #import numpy as np
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    import scipy.signal as signal
    
    #import read_SWData
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    starttime = dt.datetime(2016, 5, 1)
    stoptime = dt.datetime(2016, 7, 1)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    tag = 'u_mag'  #  solar wind property of interest
    
    juno = spacecraftdata.SpacecraftData('Juno')
    juno.read_processeddata(starttime, stoptime)
    juno.find_state(reference_frame, observer)
    
    juno.data = find_RollingDerivativeZScore(juno.data, tag, 2)
    juno.data['binary_' + tag] = juno.data['smooth_ddt_' + tag + '_zscore'].where(juno.data['smooth_ddt_' + tag + '_zscore'] > 4, 0)
    juno.data['binary_' + tag] = juno.data['binary_' + tag].where(juno.data['binary_' + tag] == 0, 1)
    
    tao = read_SWModel.TaoSW('Juno', starttime, stoptime)
    vogt = read_SWModel.VogtSW('Juno', starttime, stoptime)
    mich = read_SWModel.MSWIM2DSW('Juno', starttime, stoptime)
    huxt = read_SWModel.HUXt('Jupiter', starttime, stoptime)
    
    tao = find_RollingDerivativeZScore(tao, tag, 2)
    vogt = find_RollingDerivativeZScore(vogt, tag, 2)
    mich = find_RollingDerivativeZScore(mich, tag, 2)
    huxt = find_RollingDerivativeZScore(huxt, tag, 2)
    
    tao['binary_' + tag] = tao['smooth_ddt_' + tag + '_zscore'].where(tao['smooth_ddt_' + tag + '_zscore'] > 4, 0)
    tao['binary_' + tag] = tao['binary_' + tag].where(tao['binary_' + tag] == 0, 1)
    vogt['binary_' + tag] = vogt['smooth_ddt_' + tag + '_zscore'].where(vogt['smooth_ddt_' + tag + '_zscore'] > 4, 0)
    vogt['binary_' + tag] = vogt['binary_' + tag].where(vogt['binary_' + tag] == 0, 1)
    mich['binary_' + tag] = mich['smooth_ddt_' + tag + '_zscore'].where(mich['smooth_ddt_' + tag + '_zscore'] > 4, 0)
    mich['binary_' + tag] = mich['binary_' + tag].where(mich['binary_' + tag] == 0, 1)
    huxt['binary_' + tag] = huxt['smooth_ddt_' + tag + '_zscore'].where(huxt['smooth_ddt_' + tag + '_zscore'] > 4, 0)
    huxt['binary_' + tag] = huxt['binary_' + tag].where(huxt['binary_' + tag] == 0, 1)
    
    fig, axs = plt.subplots(nrows=3, figsize=(8,6))
    
    axs[0].plot(juno.data.index, juno.data[tag], label='Juno/JADE')
    axs[0].plot(tao.index, tao[tag], label='Tao')
    axs[0].plot(vogt.index, vogt[tag], label='SWMF-OH')
    axs[0].plot(mich.index, mich[tag], label='MSWIM2D')
    if tag == 'u_mag': axs[0].plot(huxt.index, huxt[tag], label='HUXt')
    
    axs[0].legend()
    axs[0].set_ylabel(r'$U_{SW} [km/s]$')
    
    tao_reindexed = tao.reindex(juno.data.index, method='nearest')
    vogt_reindexed = vogt.reindex(juno.data.index, method='nearest')
    mich_reindexed = mich.reindex(juno.data.index, method='nearest')
    huxt_reindexed = huxt.reindex(juno.data.index, method='nearest')
    
    axs[1].plot(juno.data.index, juno.data[tag])
    axs[1].plot(tao_reindexed.index, tao_reindexed[tag])
    axs[1].plot(vogt_reindexed.index, vogt_reindexed[tag])
    axs[1].plot(mich_reindexed.index, mich_reindexed[tag])
    if tag == 'u_mag': axs[1].plot(huxt_reindexed.index, huxt_reindexed[tag])
    
    axs[1].set_ylabel(r'$U_{SW} [km/s]$')
    
    axs[2].plot(juno.data.index, juno.data['binary_' + tag])
    axs[2].plot(tao_reindexed.index, tao_reindexed['binary_' + tag])
    axs[2].plot(vogt_reindexed.index, vogt_reindexed['binary_' + tag])
    axs[2].plot(mich_reindexed.index, mich_reindexed['binary_' + tag])
    axs[2].plot(huxt_reindexed.index, huxt_reindexed['binary_' + tag])
    
    axs[2].set_ylabel(r'$\Delta4\sigma$')
    
    plt.tight_layout()
    plt.savefig('Binary_FlowSpeed.png', dpi=200)
    plt.show()
    
    
    return(juno.data[tag])
    
def SWData_TaylorDiagram_JunoExample():
    #import datetime as dt
    #import pandas as pd
    #import numpy as np
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    
    #import sys
    #sys.path.append('/Users/mrutala/projects/SolarWindProp/')
    #import read_SWData
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    starttime = dt.datetime(2016, 5, 1)
    stoptime = dt.datetime(2016, 7, 1)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    tag = 'u_mag'  #  solar wind property of interest
    
    juno = spacecraftdata.SpacecraftData('Juno')
    juno.read_processeddata(starttime, stoptime)
    juno.find_state(reference_frame, observer)
    
    shifted_stats = list()
    
    fig, ax = TD.plot_TaylorDiagram(juno.data[tag], juno.data[tag], color='red', marker='X', markersize='0')
    shifts = np.arange(-72, 72+2, 2)  #  -/+ hours of shift for the timeseries

    for shift in shifts:
        shifted_data = juno.data.copy()
        shifted_data.index = shifted_data.index + pd.Timedelta(shift, 'hours')
        shifted_data_reindexed = shifted_data.reindex(juno.data.index, method='nearest')
        stats, rmse = TD.find_TaylorStatistics(shifted_data_reindexed[tag], juno.data[tag])
        shifted_stats.append(stats)
         
    coords_from_stats = [(np.arccos(e1), e2) for e1, e2 in shifted_stats]
    focus_model_plot = ax.scatter(*zip(*coords_from_stats), c=shifts, cmap='plasma', s=10, marker='*')
             
    ax.set_ylim((0,60))
    plt.margins(0)
    plt.colorbar(focus_model_plot, location='bottom', orientation='horizontal',
                 label=r'$\Delta t$ [hours]', fraction=0.05, pad=-0.1,
                 ticks=np.arange(-72, 72+12, 12))
    
    plt.tight_layout()
    plt.savefig('Juno_shifteddata_TaylorDiagram.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    
    
    return(shifted_stats)

def SWData_MI():
    #import datetime as dt
    #import pandas as pd
    #import numpy as np
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    import scipy.signal as signal
    
    #import read_SWData
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    import sys
    sys.path.append('/Users/mrutala/code/python/libraries/aaft/')
    sys.path.append('/Users/mrutala/code/python/libraries/generic_MI_lag_finder/')
    import generic_mutual_information_routines as mi_lib
    
    starttime = dt.datetime(2016, 5, 15)
    stoptime = dt.datetime(2016, 6, 22)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    tag = 'u_mag'  #  solar wind property of interest
    
    juno = spacecraftdata.SpacecraftData('Juno')
    juno.read_processeddata(starttime, stoptime)
    juno.find_state(reference_frame, observer)
    juno.data = juno.data.resample("15Min").mean()
    
    tao = read_SWModel.Tao('Juno', starttime, stoptime)
    #tao = tao.resample("15Min").mean()
    tao = tao.reindex(juno.data.index, method='nearest')  #!!! is this better, or is lining both up from the start better?
    #vogt = read_SWModel.VogtSW('Juno', starttime, stoptime)
    #mich = read_SWModel.MSWIM2DSW('Juno', starttime, stoptime)
    #huxt = read_SWModel.HUXt('Jupiter', starttime, stoptime)
    
    #minutely_index = pd.date_range(juno.data.index[0], juno.data.index[-1], freq='T')
    
    #juno_reindexed = juno.data.reindex(minutely_index, method='nearest')
    #tao_reindexed = tao.reindex(minutely_index, method='nearest')
    #vogt_reindexed = vogt.reindex(minutely_index, method='nearest')
    #mich_reindexed = mich.reindex(minutely_index, method='nearest')
    #huxt_reindexed = huxt.reindex(minutely_index, method='nearest')
    
    jr = find_RollingDerivativeZScore(juno.data, tag, 2) # !!! vvv
    tr = find_RollingDerivativeZScore(tao, tag, 2) # !!! Rolling derivative of the 15 min resample?
    print(len(juno.data))
    print(len(tao))
    print(len(jr))
    print(len(tr))
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(juno.data.index[0:1296], juno.data['u_mag'][0:1296], color='gray')
        axs[0].plot(tao.index[0:1296], tao['u_mag'][0:1296])
        
        axs[1].plot(jr.index[0:1296], jr['smooth_ddt_'+tag+'_zscore'][0:1296], color='gray')
        axs[1].plot(tr.index[0:1296], tr['smooth_ddt_'+tag+'_zscore'][0:1296])
        
        plt.show()
        
        ax, lags, mi, RPS_mi, x_sq, x_pw = mi_lib.mi_lag_finder(juno.data['u_mag'][0:1296].to_numpy(dtype='float64'), tao['u_mag'][0:1296].to_numpy(dtype='float64'), temporal_resolution=15, max_lag=4320, min_lag=-4320, remove_nan_rows=True)
        #ax, lags, mi, RPS_mi, x_sq, x_pw = mi_lib.mi_lag_finder(np.array(jr['smooth_ddt_'+tag+'_zscore'][0:1296]), np.array(tr['smooth_ddt_'+tag+'_zscore'][0:1296]), temporal_resolution=15, max_lag=288, min_lag=-288, remove_nan_rows=True)
    
    #plt.show()
    return()

    #ax, lags, mutual_information, RPS_mutual_information, x_squared_df, x_piecewise_df = generic_mutual_information_routines.mi_lag_finder(juno_reindexed['u_mag'][0:38880],tao_reindexed['u_mag'][0:38880],temporal_resolution=1,max_lag=4320,min_lag=-4320)


def QQplot_datacomparison():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    import spacecraftdata as SpacecraftData
    
    # =========================================================================
    #   Tag to focus on in the spacecraft data structure
    #   Can be: (u_mag, p_dyn, n_proton, B_mag)
    # =========================================================================
    tag = 'u_mag'
    
    #  Parameters to get the spacecraft positions from SPICE
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'
    
    #  Read spacecraft data: one abcissa, n ordinates
    spacecraft_x = SpacecraftData.SpacecraftData('Juno')
    spacecraft_x.read_processeddata(everything=True)
    spacecraft_x.find_state(reference_frame, observer) 
    spacecraft_x.find_subset(coord1_range=np.array((4.4, 6.0))*spacecraft_x.au_to_km, 
                       coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                       transform='reclat')
    
    spacecraft_y = []
    spacecraft_y.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_y.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_y.append(SpacecraftData.SpacecraftData('Ulysses'))
    
    for sc in spacecraft_y:
        sc.read_processeddata(everything=True)
        sc.find_state(reference_frame, observer) 
        sc.find_subset(coord1_range=np.array((4.4, 6.0))*sc.au_to_km, 
                       coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                       transform='reclat')
    
    #  Calculate percentiles (i.e., 100 quantiles)
    x_quantiles = np.nanquantile(spacecraft_x.data[tag], np.linspace(0, 1, 100))
    y_quantiles = [np.nanquantile(sc.data[tag], np.linspace(0, 1, 100)) for sc in spacecraft_y]
    
    #  Dictionaries of plotting and labelling parametes
    label_dict = {'u_mag': r'$U_{mag,SW}$ [km/s]', 
                  'p_dyn_proton': r'$P_{dyn,SW,p}$ [nPa]'}
    plot_params = {'xlabel':label_dict[tag], 
                   'xlim': (300, 700), 
                   'ylabel':label_dict[tag], 
                   'ylim': (300, 700)}
    
    #  Plot the figure
    try: plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
    except: pass

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_gridspec(left=0.05, bottom=0.05, top=0.75, right=0.75).subplots()
    
    for sc, y in zip(spacecraft_y, y_quantiles):
        ax.plot(x_quantiles, y, marker='o', markersize=4, linestyle='None', label=sc.name)
    
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,
             linestyle=':', color='black')
    ax.set(xlabel=plot_params['xlabel'], xlim=plot_params['xlim'], 
           ylabel=plot_params['ylabel'], ylim=plot_params['ylim'],
           aspect=1.0)
    ax.legend()
    
    ax_top = ax.inset_axes([0, 1.05, 1.0, 0.2], sharex=ax)
    x_hist, x_bins = np.histogram(spacecraft_x.data[tag], bins=50, range=plot_params['xlim'])
    #juno_kde = sns.kdeplot(data=juno.data, x=tag, bw_adjust=0.5, ax=ax_top)
    ax_top.stairs(x_hist, edges=x_bins, orientation='vertical')
    ax_top.set(ylabel='Number')
    ax_top.tick_params('x', labelbottom=False)
    
    ax_right = ax.inset_axes([1.05, 0, 0.2, 1.0], sharey=ax)
    for sc in spacecraft_y:
        y_hist, y_bins = np.histogram(sc.data[tag], bins=50, range=plot_params['ylim'])
        #ulys_kde = sns.kdeplot(data=ulys.data, y=tag, bw_adjust=0.5, ax=ax_right)
        ax_right.stairs(y_hist, edges=y_bins, orientation='horizontal')
        #ax_right.scatter(np.zeros(len(ulys.data[tag])), ulys.data[tag], marker='+', s=2)
    ax_right.set(xlabel='Number')
    ax_right.tick_params('y', labelleft=False)
    
    fig.align_labels()
    plt.tight_layout()
    plt.savefig('figures/QQplot_datacomparison_'+tag+'.png')
    plt.show()
    
def QQplot_modelcomparison():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    import spiceypy as spice
    
    import spacecraftdata as SpacecraftData
    import read_SWModel
    
    starttime = dt.datetime(2016, 5, 1)
    stoptime = dt.datetime(2016, 7, 1)
    #reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    #observer = 'SUN'
    
    tag = 'u_mag'
    label_dict = {'u_mag': r'$u_{mag,SW}$ [km/s]', 'p_dyn': r'$p_{dyn,SW}$ [nPa]'}
    
    juno = SpacecraftData.SpacecraftData('Juno')
    juno.read_processeddata(everything=True)
    
    tao = read_SWModel.TaoSW('Juno', starttime, stoptime)
    vogt = read_SWModel.VogtSW('Juno', starttime, stoptime)
    mich = read_SWModel.MSWIM2DSW('Juno', starttime, stoptime)
    huxt = read_SWModel.HUXt('Jupiter', starttime, stoptime)
    
    tao = tao[(tao.index >= juno.data.index[0]) & (tao.index <= juno.data.index[-1])]
    vogt = vogt[(vogt.index >= juno.data.index[0]) & (vogt.index <= juno.data.index[-1])]
    mich = mich[(mich.index >= juno.data.index[0]) & (mich.index <= juno.data.index[-1])]
    huxt = huxt[(huxt.index >= juno.data.index[0]) & (huxt.index <= juno.data.index[-1])]
    
    #juno_u_mag_zscore = (juno.data.u_mag - np.mean(juno.data.u_mag))/np.std(juno.data.u_mag)
    
    juno_quantiles = np.nanquantile(juno.data[tag], np.linspace(0, 1, 100))
    tao_quantiles  = np.nanquantile(tao[tag], np.linspace(0, 1, 100))
    vogt_quantiles = np.nanquantile(vogt[tag], np.linspace(0, 1, 100))
    mich_quantiles = np.nanquantile(mich[tag], np.linspace(0, 1, 100))
    if tag == 'u_mag': huxt_quantiles = np.nanquantile(huxt[tag], np.linspace(0, 1, 100))
    
    fig, axs = plt.subplots(1, 1, figsize=(8,6))
    axs.plot(juno_quantiles, tao_quantiles, marker='o', color='teal', markersize=4, linestyle='None', label='Tao')
    axs.plot(juno_quantiles, vogt_quantiles, marker='o', color='magenta', markersize=4, linestyle='None', label='SWMF-OH (BU)')
    axs.plot(juno_quantiles, mich_quantiles, marker='o', color='red', markersize=4, linestyle='None', label='MSWIM2D')
    if tag == 'u_mag': axs.plot(juno_quantiles, huxt_quantiles, marker='o', color='gold', markersize=4, linestyle='None', label='HUXt')
    
    axs.plot([0, 1], [0, 1], transform=axs.transAxes,
             linestyle=':', color='black')
    
    #lim = np.min([juno_quantiles, ulys_quantiles])
    # axs.set_xscale('log')
    # axs.set_yscale('log')
    # axs.set_xlim(1e-3,1.5)
    # axs.set_ylim(1e-3,1.5)
    
    axs.set_xlim(350,600)
    axs.set_ylim(350,600)
    
    axs.set_xlabel(r'Juno JADE ' + label_dict[tag])
    axs.set_ylabel(r'Model ' + label_dict[tag])
    axs.set_aspect(1)
    axs.legend()

    plt.tight_layout()
    plt.savefig('QQplot_modelcomparison_u_mag.png', dpi=200)
    plt.show()
    
def plot_spacecraftcoverage_solarcycle():
    import matplotlib.pyplot as plt
    #import datetime as dt
    #import numpy as np
    
    #import pandas as pd
    import spiceypy as spice
    
    #import sys
    #sys.path.append('/Users/mrutala/code/python/')
    import spiceypy_metakernelcontext as spiceypy_mkc
    #sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    #import read_SWData
    import spacecraftdata as SpacecraftData
    
    # =============================================================================
    # Find when Ulysses was within +/- 30 deg. of the ecliptic, and near 5 AU
    # =============================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    spacecraft_list = []
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Ulysses'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Juno'))
    
    for sc in spacecraft_list:
        sc.read_processeddata(everything=True)
        sc.find_state(reference_frame, observer)
        sc.find_subset(coord1_range=np.array((4.0, 6.4))*sc.au_to_km, # 4.4 - 6
                       coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                       transform='reclat')
    spacecraft_labels = {'Voyager 1': 1, 
                         'Voyager 2': 2, 
                         'Ulysses'  : 3, 
                         'Juno'     : 4}
    
    # =============================================================================
    # F10.4 Radio flux from the sun
    # =============================================================================
    column_headers = ('date', 'observed_flux', 'adjusted_flux',)
    solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                   header = 0, names = column_headers)
    solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                for d in solar_radio_flux['date']]

    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, ax0 = plt.subplots()
        ax0.plot(solar_radio_flux['date'], solar_radio_flux['observed_flux'], 
                 marker='.', color='gray', markersize=2, linestyle='None')
        
        # import matplotlib.patches as patch
        # from matplotlib.collections import PatchCollection
        ax0twin = ax0.twinx()
        for sc in spacecraft_list:
            
            #  This bit splits the spacecraft data up into chunks to plot
            sc_deltas = sc.data.index.to_series().diff()
            gaps_indx = np.where(sc_deltas > dt.timedelta(days=30))[0]
            start_indx = np.insert(gaps_indx, 0, 0)
            stop_indx = np.append(gaps_indx, len(sc.data.index))-1
            
            print('Start and stop dates for ' + sc.name + ', 4.4-6.0 AU and +/-8.5 deg. lat:')
            for first, final in zip(start_indx, stop_indx):
                print(sc.data.index[first].strftime('%Y-%m-%d') + 
                      ' -- ' + 
                      sc.data.index[final].strftime('%Y-%m-%d'))
            print('------------------------------------------')
            # rectangle_list = []
            # for x0, xf in zip(np.insert(start_indx, 0, 0), stop_indx):
            #     ll_corner = (sc.data.index[x0].timestamp(), spacecraft_labels[sc.name]-0.333)
            #     width = sc.data.index[xf-1].timestamp() - sc.data.index[x0].timestamp()
            #     height = 0.666
                
            #     rectangle_list.append(patch.Rectangle(ll_corner, width, height))
            # ax0twin.add_collection(PatchCollection(rectangle_list))
            
            ax0twin.plot(sc.data.index, np.zeros(len(sc.data.index))+spacecraft_labels[sc.name], linestyle='None', marker='|', markersize=36)

        ax0.set_xlim((dt.datetime(1975, 1, 1), dt.datetime(2020, 1, 1)))
        ax0.set_ylabel('Observed Radio Flux @ 10.7 cm [SFU]', fontsize=16)
        
        #ax0.set_title(r'Spacecraft data availability relative to the Solar Cycle between 4.5-5.5 $R_J$ and -10-10 deg. lat.', wrap=True, fontsize=18)
        
        ax0.set_ylim((50, 400))
        
        ax0twin.set_ylim((0,5))
        ax0twin.set_yticks(list(spacecraft_labels.values()))
        ax0twin.set_yticklabels(list(spacecraft_labels.keys()))
        ax0twin.set_ylabel('Spacecraft', fontsize=18)
        ax0twin.tick_params(labelsize=18)
        
        ax0.tick_params(labelsize=18)
        
        plt.tight_layout()
        plt.savefig('figures/SolarCycleComparison.png')
        plt.show()
        
        fig, axs = plt.subplots(nrows=2, ncols=2)
        
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
        spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
        for ax, sc in zip(axs.reshape(-1),spacecraft_list):
            
            observer='Jupiter Barycenter'
            reference_frame = 'ECLIPJ2000'
            sc.find_state(reference_frame, observer, keep_kernels=True)
            
            sc_full = SpacecraftData.SpacecraftData(sc.name)
            sc_full.find_lifetime(keep_kernels=True)
            sc_full.make_timeseries(timedelta=dt.timedelta(days=10))
            sc_full.find_state(reference_frame, observer, keep_kernels=True)
            
            xyz_in_AU = np.array([row[['x_pos', 'y_pos', 'z_pos']]/sc.au_to_km
                                  for indx, row in sc.data.iterrows()])
            
            xyz_full_in_AU = np.array([row[['x_pos', 'y_pos', 'z_pos']]/sc.au_to_km
                                  for indx, row in sc_full.data.iterrows()])
            
            ax.plot(xyz_full_in_AU[:,0], xyz_full_in_AU[:,1],
                    color='gray', linewidth=1)
            ax.plot(xyz_in_AU[:,0], xyz_in_AU[:,1], 
                     label=sc.name, marker='o', markersize=1, linestyle='None')
            
            ax.set_aspect(1)

            times = [spice.str2et(t.strftime('%b %d, %Y %H:%M:%S.%f')) for t in sc.data.index]
            ear_pos, ear_lt = spice.spkpos('EARTH BARYCENTER', times, reference_frame, 'NONE', observer)
            ear_pos = np.array(ear_pos) / sc.au_to_km
            jup_pos, jup_lt = spice.spkpos('JUPITER BARYCENTER', times, reference_frame, 'NONE', observer)
            jup_pos = np.array(jup_pos) / sc.au_to_km
            
            ax.plot(ear_pos[:,0], ear_pos[:,1], 
                    label='Earth', color='xkcd:kelly green', marker='o', markersize=1, linestyle='None')
            ax.plot(ear_pos[-1,0], ear_pos[-1,1],  
                    color='xkcd:kelly green', marker='o', markersize=4)
            
            ax.plot(jup_pos[:,0], jup_pos[:,1], 
                    label='Jupiter', color='xkcd:peach', marker='o', markersize=1, linestyle='None')
            ax.plot(jup_pos[-1,0], jup_pos[-1,1],
                    color='xkcd:peach', marker='o', markersize=4)
            
            ax.set(xlim=[-6,6], xlabel='J2000 Ecliptic X [AU]', 
                   ylim=[-6,6], ylabel='J2000 Ecliptic Y [AU]')
        spice.kclear()
        
        #fig.legend()
        plt.show()
    
    #return(sc_data) 
    
def find_RollingDerivativeZScore(dataframe, tag, window):
    
    #  Test smoothing
    halftimedelta = dt.timedelta(hours=0.5*window)
    
    derivative_tag = 'ddt_' + tag
    smooth_tag = 'smooth_' + tag
    smooth_derivative_tag = 'smooth_ddt_' + tag
    smooth_derivative_zscore_tag = 'smooth_ddt_' + tag + '_zscore'
    
    #  
    dataframe[derivative_tag] = np.gradient(dataframe[tag], 
                                              dataframe.index.values.astype(np.int64))
    
    rolling_dataframe = dataframe.rolling(2*halftimedelta, min_periods=1, 
                                          center=True, closed='left')
    
    dataframe[smooth_tag] = rolling_dataframe[tag].mean()
    
    dataframe[smooth_derivative_tag] = np.gradient(dataframe[smooth_tag], 
                                                     dataframe.index.values.astype(np.int64))
    
    dataframe[smooth_derivative_zscore_tag] = dataframe[smooth_derivative_tag] / np.std(dataframe[smooth_derivative_tag])
    
    return(dataframe)
    
def SWData_derivativesplots_old():
    
    


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from astropy.io import ascii
    import datetime as dt
    import matplotlib.dates as mdates
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    #from read_juno_mag_from_amda import juno_b_mag_from_amda
    import read_SWModel as read_SWModel
    import read_SWData as read_SWData
    
    from matplotlib import animation
    
    #timerange = (Time('2001-01-01T00:00:00', format='isot').datetime, Time('2001-03-01T00:00:00', format='isot').datetime)
    target = 'juno'
    timerange = (dt.datetime(2016, 5, 20, 0), dt.datetime(2016, 6, 15, 0))
     
    data_path = ''
    output_figure_name = 'test.png'
    
# =============================================================================
#     Hard-coded results from polling DIAS Magnetospheres group
# =============================================================================
    #  Alexandra, Charlie, Caitriona, Sean, Dale
    results_dict = {'2016-141 12:00:00': 5,
                    '2016-142 12:00:00': 2,
                    '2016-143 03:00:00': 2,
                    '2016-144 18:00:00': 1,
                    '2016-149 12:00:00': 5,
                    '2016-150 03:00:00': 1,
                    '2016-150 18:00:00': 2,
                    '2016-151 12:00:00': 1,
                    '2016-152 12:00:00': 1,
                    '2016-154 21:00:00': 4,
                    '2016-156 00:00:00': 1,
                    '2016-158 06:00:00': 2,
                    '2016-159 21:00:00': 1,
                    '2016-162 12:00:00': 1,
                    '2016-165 18:00:00': 2,
                    }
    results = pd.DataFrame(data=results_dict.values(), 
                           index=[dt.datetime.strptime(t, '%Y-%j %H:%M:%S') for t in results_dict.keys()], 
                           columns=['counts'])


    # =============================================================================
    # Adding Wilson Juno/JADE data
    # =============================================================================
    juno_data = read_SWData.Juno_Wilson2018(timerange[0], timerange[1])
    
    #  Test derivatives
    #d_Umag_dt = np.gradient(juno_data['Umag'], juno_data.index.values.astype(np.int64))
    
    #  Test smoothing
    halftimedelta = dt.timedelta(hours=2.0)
    
    #  This took 18s to run, 100,000x slower than the pandas rolling below
    #timer1 = dt.datetime.now()
    #smooth_Umag = np.empty(len(juno_data))
    #for i, t in enumerate(juno_data.index):
    #    temp = juno_data['Umag'].loc[(juno_data.index >= t - halftimedelta)
    #                                 & (juno_data.index <  t + halftimedelta)]
    #    smooth_Umag[i] = np.mean(temp)
    #print('Time elapsed in smoothing loop: ' + str(dt.datetime.now()-timer1))
    #d_smooth_Umag_dt = np.gradient(smooth_Umag, juno_data.index.values.astype(np.int64))
    #norm_dsU_dt = d_smooth_Umag_dt / np.std(d_smooth_Umag_dt)
    
    #  Pandas rolling test
    juno_data['ddt_pdynproton'] = np.gradient(np.log10(juno_data['pdynproton']), 
                                              juno_data.index.values.astype(np.int64))
    rolling_juno_data = juno_data.rolling(2*halftimedelta, min_periods=1, 
                                          center=True, closed='left')
    juno_data['smooth_pdynproton'] = rolling_juno_data['pdynproton'].mean()
    
    juno_data['smooth_ddt_pdynproton'] = np.gradient(np.log10(juno_data['smooth_pdynproton']), 
                                                     juno_data.index.values.astype(np.int64))
    juno_data['norm_smooth_ddt_pdynproton'] = juno_data['smooth_ddt_pdynproton'] / np.std(juno_data['smooth_ddt_pdynproton'])
    
    #d_smooth_Umag_dt = np.gradient(rolling_juno_data['Umag'].mean(), juno_data.index.values.astype(np.int64))
    #norm_dsU_dt = d_smooth_Umag_dt / np.std(d_smooth_Umag_dt)
    #d_smooth_pdynproton_dt = np.gradient(rolling_juno_data['pdynproton'].mean(), juno_data.index.values.astype(np.int64))
    #norm_dsp_dt = d_smooth_pdynproton_dt / np.std(d_smooth_pdynproton_dt)
    
    # %%===========================================================================
    #  Plot the figure   
    # =============================================================================
    fig = plt.figure(layout='constrained', figsize=(8,11))
    subfigs = fig.subfigures(4, 1, hspace=0.02, height_ratios=[1,1,1,0.2])
    
    axs0 = subfigs[0].subplots(2, 1, sharex=True)
    axs1 = subfigs[1].subplots(2, 1, sharex=True)
    axs2 = subfigs[2].subplots(2, 1, sharex=True)
    cax = subfigs[3].subplots(1, 1)
    
    
    
    # Pressures
    axs0[0].plot(juno_data.index, juno_data.pdynproton, 
                 color='k', linewidth=2, 
                 label='JADE (Wilson+ 2018)')
    axs0[0].fill_between(juno_data.index, 
                         juno_data.pdynproton-juno_data.pdynproton_err, 
                         juno_data.pdynproton+juno_data.pdynproton_err, 
                         step='mid', color='black', alpha=0.5)
    axs0[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                 color='orange', linewidth=3,
                 label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
    
    axs1[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
    axs1[0].fill_between(juno_data.index, 
                         juno_data.pdynproton-juno_data.pdynproton_err, 
                         juno_data.pdynproton+juno_data.pdynproton_err, 
                         step='mid', color='black', alpha=0.5)
    axs1[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                 color='orange', linewidth=3,
                 label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
    
    axs2[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
    axs2[0].fill_between(juno_data.index, 
                         juno_data.pdynproton-juno_data.pdynproton_err, 
                         juno_data.pdynproton+juno_data.pdynproton_err, 
                         step='mid', color='black', alpha=0.5)
    axs2[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                 color='orange', linewidth=3,
                 label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))

    # Derivatives
    #axs0[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
    #axs0[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
    axs0[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                 label=r'Smoothed + Differentiated + Normalized')
    #axs1[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
    #axs1[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
    axs1[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                 label=r'Smoothed + Differentiated + Normalized')
    #axs2[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
    #axs2[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
    axs2[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                 label=r'Smoothed + Differentiated + Normalized')
    
    #  Assign colors from colormap to selection results
    cmap = plt.get_cmap('plasma')
    bounds = np.arange(0.5, results.counts.max()+1, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    
    for t, r in results.iterrows():
        for axs in (axs0, axs1, axs2,):
            color = cmap((r['counts']-0.5)/results.counts.max())
            alpha = 0.6 # (r['counts']-0.5)/results.counts.max()
            axs[0].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
            axs[1].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
    
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label='# of selections', 
             ticks=np.arange(1, results.counts.max()+1, 1))
    
    axs0_range = (dt.datetime(2016, 5, 20, 0), dt.datetime(2016, 6, 2, 0),)
    axs0[1].set_xlim(axs0_range)
    axs0[1].set_xticks(np.arange(axs0_range[0], axs0_range[1], dt.timedelta(days=2)).astype(dt.datetime))
    
    axs1_range = (dt.datetime(2016, 5, 26, 12), dt.datetime(2016, 6, 8, 12),)
    axs1[1].set_xlim(axs1_range)
    test_arr = np.arange(axs1_range[0]+dt.timedelta(hours=12), axs1_range[1]+dt.timedelta(hours=12), dt.timedelta(days=2)).astype(dt.datetime)
    axs1[1].set_xticks(test_arr)
    
    axs2_range = (dt.datetime(2016, 6, 2, 0), dt.datetime(2016, 6, 15, 0))
    axs2[1].set_xlim(axs2_range)   
    axs2[1].set_xticks(np.arange(axs2_range[0], axs2_range[1], dt.timedelta(days=2)).astype(dt.datetime))
    
    for ax in (axs0, axs1, axs2,):
        ax[0].set_ylim(5e-4, 5e0)
        ax[0].set_yscale('log')
        ax[0].set_ylabel(r'$P_{dyn,H^+}$ [nPa]', fontsize=12)
        ax[0].tick_params(labelsize=14)
        
        ax[1].set_ylim(-6, 6)
        ax[1].set_ylabel(r'$\frac{d}{dt}log_{10}(P_{dyn,H^+})$ [$\sigma$]', fontsize=12)
        ax[1].tick_params(labelsize=14)
        #xticks = ax[1].get_xticks()
        #xlabels = ax[1].get_xticklabels()
        #ax[1].set_xticks(xticks)
        #ax[1].set_xticklabels(xlabels, rotation=30, ha='right', rotation_mode='anchor')
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%j"))
        
    axs0[1].set_xlabel(' ', fontsize=14)
    axs1[1].set_xlabel(' ', fontsize=14)
    axs2[1].set_xlabel('Day of Year 2016', fontsize=14)
    plt.suptitle('Cruise solar wind parameters, Juno JADE (Wilson+ 2018) \n ' + 
                 r'Smoothed by $\pm$'+str(halftimedelta), wrap=True, fontsize=14)
    #subfigs[0].align_labels()
    for subfig in subfigs: subfig.align_labels()
    
    plt.savefig('SWFeatureSelection.png', format='png', dpi=500)
    plt.show()
    
# =============================================================================
#     Attempt an animation
# =============================================================================
    #  Initialize figure and axes
    fig = plt.figure(layout='constrained', figsize=(8,11))
    subfigs = fig.subfigures(4, 1, hspace=0.02, height_ratios=[1,1,1,0.2])
    
    axs0 = subfigs[0].subplots(2, 1, sharex=True)
    axs1 = subfigs[1].subplots(2, 1, sharex=True)
    axs2 = subfigs[2].subplots(2, 1, sharex=True)
    cax = subfigs[3].subplots(1, 1)    
    
    #  timedeltas
    halftimedeltas_hours = np.arange(0.05, 8.05, 0.05)
    
    variance = []

    def animate_func(f):
        
        halftimedelta1 = dt.timedelta(hours=halftimedeltas_hours[f])
        
        #  Pandas rolling test
        juno_data['ddt_pdynproton'] = np.gradient(np.log10(juno_data['pdynproton']), 
                                                  juno_data.index.values.astype(np.int64))
        rolling_juno_data = juno_data.rolling(2*halftimedelta1, min_periods=1, 
                                              center=True, closed='left')
        juno_data['smooth_pdynproton'] = rolling_juno_data['pdynproton'].mean()
        
        juno_data['smooth_ddt_pdynproton'] = np.gradient(np.log10(juno_data['smooth_pdynproton']), 
                                                         juno_data.index.values.astype(np.int64))
        juno_data['norm_smooth_ddt_pdynproton'] = juno_data['smooth_ddt_pdynproton'] / np.std(juno_data['smooth_ddt_pdynproton'])
        
        variance.append(np.var(juno_data['smooth_ddt_pdynproton']))
        
        # Clears the figure to update the line, point,   
        for axs in (axs0, axs1, axs2,):
            axs[0].clear()
            axs[1].clear()
        cax.clear()
        
        # Pressures
        axs0[0].plot(juno_data.index, juno_data.pdynproton, 
                     color='k', linewidth=2, 
                     label='JADE (Wilson+ 2018)')
        axs0[0].fill_between(juno_data.index, 
                             juno_data.pdynproton-juno_data.pdynproton_err, 
                             juno_data.pdynproton+juno_data.pdynproton_err, 
                             step='mid', color='black', alpha=0.5)
        axs0[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                     color='orange', linewidth=3,
                     label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
        
        axs1[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
        axs1[0].fill_between(juno_data.index, 
                             juno_data.pdynproton-juno_data.pdynproton_err, 
                             juno_data.pdynproton+juno_data.pdynproton_err, 
                             step='mid', color='black', alpha=0.5)
        axs1[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                     color='orange', linewidth=3,
                     label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
        
        axs2[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
        axs2[0].fill_between(juno_data.index, 
                             juno_data.pdynproton-juno_data.pdynproton_err, 
                             juno_data.pdynproton+juno_data.pdynproton_err, 
                             step='mid', color='black', alpha=0.5)
        axs2[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                     color='orange', linewidth=3,
                     label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))

        # Derivatives
        #axs0[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
        #axs0[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
        axs0[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                     label=r'Smoothed + Differentiated + Normalized')
        #axs1[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
        #axs1[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
        axs1[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                     label=r'Smoothed + Differentiated + Normalized')
        #axs2[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
        #axs2[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
        axs2[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                     label=r'Smoothed + Differentiated + Normalized')
        
        #  Assign colors from colormap to selection results
        cmap = plt.get_cmap('plasma')
        bounds = np.arange(0.5, results.counts.max()+1, 1)
        norm = colors.BoundaryNorm(bounds, cmap.N, extend='neither')
        
        for t, r in results.iterrows():
            for axs in (axs0, axs1, axs2,):
                color = cmap((r['counts']-0.5)/results.counts.max())
                alpha = 0.6 # (r['counts']-0.5)/results.counts.max()
                axs[0].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
                axs[1].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
        
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation='horizontal', label='# of selections', 
                 ticks=np.arange(1, results.counts.max()+1, 1))
        
        axs0_range = (dt.datetime(2016, 5, 20, 0), dt.datetime(2016, 6, 2, 0),)
        axs0[1].set_xlim(axs0_range)
        axs0[1].set_xticks(np.arange(axs0_range[0], axs0_range[1], dt.timedelta(days=2)).astype(dt.datetime))
        
        axs1_range = (dt.datetime(2016, 5, 26, 12), dt.datetime(2016, 6, 8, 12),)
        axs1[1].set_xlim(axs1_range)
        test_arr = np.arange(axs1_range[0]+dt.timedelta(hours=12), axs1_range[1]+dt.timedelta(hours=12), dt.timedelta(days=2)).astype(dt.datetime)
        axs1[1].set_xticks(test_arr)
        
        axs2_range = (dt.datetime(2016, 6, 2, 0), dt.datetime(2016, 6, 15, 0))
        axs2[1].set_xlim(axs2_range)   
        axs2[1].set_xticks(np.arange(axs2_range[0], axs2_range[1], dt.timedelta(days=2)).astype(dt.datetime))
        
        for ax in (axs0, axs1, axs2,):
            ax[0].set_ylim(5e-4, 5e0)
            ax[0].set_yscale('log')
            ax[0].set_ylabel(r'$P_{dyn,H^+}$ [nPa]', fontsize=12)
            ax[0].tick_params(labelsize=14)
            
            ax[1].set_ylim(-6, 6)
            ax[1].set_ylabel(r'$\frac{d}{dt}log_{10}(P_{dyn,H^+})$ [$\sigma$]', fontsize=12)
            ax[1].tick_params(labelsize=14)
            #xticks = ax[1].get_xticks()
            #xlabels = ax[1].get_xticklabels()
            #ax[1].set_xticks(xticks)
            #ax[1].set_xticklabels(xlabels, rotation=30, ha='right', rotation_mode='anchor')
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%j"))
            
        axs0[1].set_xlabel(' ', fontsize=14)
        axs1[1].set_xlabel(' ', fontsize=14)
        axs2[1].set_xlabel('Day of Year 2016', fontsize=14)
        plt.suptitle('Cruise solar wind parameters, Juno JADE (Wilson+ 2018) \n ' + 
                     r'Smoothed by $\pm$'+str(halftimedelta1), wrap=True, fontsize=14)
        #subfigs[0].align_labels()
        for subfig in subfigs: subfig.align_labels()
        
    #anim = animation.FuncAnimation(fig, animate_func, interval=1000,   
    #                               frames=len(halftimedeltas_hours))
    #FFwriter = animation.FFMpegWriter(fps=6)
    #anim.save('DifferingSmoothingWindows.mp4', writer=FFwriter)
    
    fig, ax = plt.subplots()
    
    ax.plot(halftimedeltas_hours, variance[1:])
    ax.set_xlabel('Smoothing half width [hours]')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_ylabel(r'$\sigma^2$ of the smoothed and differentiated $log_{10}(P_{dyn})$')
    
    plt.savefig('SWFeatureVariance_withsmoothing.png', format='png', dpi=500)
    plt.show()
    
    return(variance)