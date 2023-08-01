#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:23:16 2023

@author: mrutala
"""
import datetime as dt
import pandas as pd
import numpy as np

import spacecraftdata
import read_SWModel

spacecraft_colors = {'Pioneer 10': '#F6E680',
                     'Pioneer 11': '#FFD485',
                     'Voyager 1' : '#FEC286',
                     'Voyager 2' : '#FEAC86',
                     'Ulysses'   : '#E6877E',
                     'Juno'      : '#D8696D'}
model_colors = {'Tao'    : '#C59FE5',
                'HUXt'   : '#A0A5E4',
                'SWMF-OH': '#98DBEB',
                'MSWIM2D': '#A9DBCA',
                'ENLIL'  : '#DCED96'}
model_symbols = {'Tao'    : 'v',
                'HUXt'   : '*',
                'SWMF-OH': '^',
                'MSWIM2D': 'd',
                'ENLIL'  : 'h'}

r_range = np.array((4.9, 5.5))  #  [4.0, 6.4]
lat_range = np.array((-6.1, 6.1))  #  [-10, 10]

# =============================================================================
# Non-plotting analysis routines
# =============================================================================
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

# =============================================================================
# Time Series plotting stuff
# =============================================================================
def plot_SingleTimeseries(parameter, spacecraft_name, model_names, starttime, stoptime):
    """
    Plot the time series of a single parameter from a single spacecraft,
    separated into 4/5 panels to show the comparison to each model.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    #  Check which parameter was specified and set up some plotting keywords
    match parameter:
        case ('u_mag' | 'flow speed'):
            tag = 'u_mag'
            ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
            plot_kw = {'yscale': 'linear', 'ylim': (350, 550),
                       'yticks': np.arange(350,550+50,100)}
        case ('p_dyn' | 'pressure'):
            tag = 'p_dyn'
            ylabel = r'Solar Wind Dynamic Pressure $p_{dyn}$ [nPa]'
            plot_kw = {'yscale': 'log', 'ylim': (1e-3, 2e0),
                       'yticks': 10.**np.arange(-3,0+1,1)}
        case ('n_tot' | 'density'):
            tag = 'n_tot'
            ylabel = r'Solar Wind Ion Density $n_{tot}$ [cm$^{-3}$]'
            plot_kw = {'yscale': 'log', 'ylim': (5e-3, 5e0),
                       'yticks': 10.**np.arange(-2, 0+1, 1)}
        case ('B_mag' | 'magnetic field'):
            tag = 'B_mag'
            ylabel = r'Solar Wind Magnetic Field Magnitude $B_{mag}$ [nT]'
            plot_kw = {'yscale': 'linear', 'ylim': (0, 5),
                       'yticks': np.arange(0, 5+1, 2)}
    #
    save_filestem = 'Timeseries_{}_{}_{}-{}'.format(spacecraft_name.replace(' ', ''),
                                                    tag,
                                                    starttime.strftime('%Y%m%d'),
                                                    stoptime.strftime('%Y%m%d'))
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime, resolution=99)
    spacecraft.find_state(reference_frame, observer)
    spacecraft.find_subset(coord1_range=np.array(r_range)*spacecraft.au_to_km, 
                           coord3_range=np.array(lat_range)*np.pi/180., 
                           transform='reclat')

    #  Read models
    models = dict.fromkeys(model_names, None)
    for model in models.keys():
        models[model] = read_SWModel.choose(model, spacecraft_name, 
                                       starttime, stoptime)
    #!!! Only because HUXt doesn't have Juno yet
    #models['HUXt'] = read_SWModel.choose('HUXt', 'Jupiter', starttime, stoptime)
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(figsize=(8,6), nrows=len(models), sharex=True, squeeze=False)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.0)
        
        for indx, ax in enumerate(axs.flatten()):
            
            ax.plot(spacecraft.data.index, 
                    spacecraft.data[tag],
                    color='xkcd:blue grey', alpha=0.6, linewidth=1.5,
                    label='Juno/JADE (Wilson+ 2018)')
            
            ax.set(**plot_kw)
            #  get_yticks() for log scale doesn't work as expected; workaround:
            yticks = ax.get_yticks()
            yticks = [yt for yt in yticks if ax.get_ylim()[0] <= yt <= ax.get_ylim()[1]]
            if indx != len(axs)-1: 
                if (yticks[0] in ax.get_ylim()) and yticks[-1] in ax.get_ylim():
                    ax.set(yticks=yticks[1:])
            #ax.grid(linestyle='-', linewidth=2, alpha=0.1, color='xkcd:black')
            
        for ax, (model, model_info) in zip(axs.flatten(), models.items()):
            if tag in model_info.columns: 
                ax.plot(model_info.index, model_info[tag], 
                        color=model_colors[model], label=model, linewidth=2)
            
            model_label_box = dict(facecolor=model_colors[model], 
                                   edgecolor=model_colors[model], 
                                   pad=0.1, boxstyle='round')
            # !!!! Again, special for HUXt
            if model == 'HUXt': model += '*'
            ax.text(0.01, 0.95, model, color='black',
                    horizontalalignment='left', verticalalignment='top',
                    bbox = model_label_box,
                    transform = ax.transAxes)
        
        fig.text(0.5, 0.025, 'Day of Year {:.0f}'.format(starttime.year), ha='center', va='center')
        fig.text(0.025, 0.5, ylabel, ha='center', va='center', rotation='vertical')
        
        tick_interval = int((stoptime - starttime).days/10.)
        subtick_interval = int(tick_interval/5.)
        
        axs[0,0].set_xlim((starttime, stoptime))
        axs[0,0].xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
        if subtick_interval > 0:
            axs[0,0].xaxis.set_minor_locator(mdates.DayLocator(interval=subtick_interval))
        axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
        
        # #  Trying to set up year labels, showing the year once at the start
        # #  and again every time it changes
        # ax_year = axs[-1].secondary_xaxis(-0.15)
        # ax_year.set_xticks([axs[-1].get_xticks()[0]])
        # ax_year.visible=False
        # ax_year.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.align_ylabels()
        for suffix in ['.png', '.pdf']:
            plt.savefig('figures/' + save_filestem, dpi=300, bbox_inches='tight')
        plt.show()
        return(models)

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

def plot_histograms():
    
    return()

# =============================================================================
# Taylor Diagrams for model baselining
# =============================================================================
def plot_TaylorDiagrams():
    
    for parameter in ['u_mag', 'p_dyn', 'n_tot', 'B_mag']:
        
        _ = plot_TaylorDiagram_Baseline(parameter, 'Juno', 
                                        ['Tao', 'SWMF-OH', 'MSWIM2D', 'ENLIL'], 
                                        dt.datetime(2016, 5, 15), dt.datetime(2016, 6, 25))
        _ = plot_TaylorDiagram_ConstantTimeWarping(parameter, 'Juno', 
                                                   ['Tao', 'SWMF-OH', 'MSWIM2D', 'ENLIL'], 
                                                   dt.datetime(2016, 5, 15), dt.datetime(2016, 6, 25))

def plot_TaylorDiagram_Baseline(parameter, spacecraft_name, model_names, starttime, stoptime):
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    
    import sys
    #sys.path.append('/Users/mrutala/projects/SolarWindProp/')
    #import read_SWData
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    match parameter:
        case ('u_mag' | 'flow speed'):
            tag = 'u_mag'
            ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
            plot_kw = {'yscale': 'linear', 'ylim': (0, 60),
                       'yticks': np.arange(350,550+50,100)}
        case ('p_dyn' | 'pressure'):
            tag = 'p_dyn'
            ylabel = r'log Solar Wind Dynamic Pressure $log_{10}(p_{dyn})$ [$log_{10}(nPa)$]'
            plot_kw = {'yscale': 'log', 'ylim': (0, 0.6),
                       'yticks': 10.**np.arange(-3,0+1,1)}
        case ('n_tot' | 'density'):
            tag = 'n_tot'
            ylabel = r'log Solar Wind Ion Density $log_{10}(n_{tot})$ [$log_{10}(cm^{-3})$]'
            plot_kw = {'yscale': 'log', 'ylim': (0, 0.6),
                       'yticks': 10.**np.arange(-2, 0+1, 1)}
        case ('B_mag' | 'magnetic field'):
            tag = 'B_mag'
            ylabel = r'Solar Wind Magnetic Field Magnitude $B_{mag}$ [nT]'
            plot_kw = {'yscale': 'linear', 'ylim': (0, 1),
                       'yticks': np.arange(0, 5+1, 2)}
    #
    save_filestem = 'TaylorDiagram_{}_{}_{}-{}'.format(spacecraft_name.replace(' ', ''),
                                                       tag,
                                                       starttime.strftime('%Y%m%d'),
                                                       stoptime.strftime('%Y%m%d'))
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime, resolution=99)
    spacecraft.find_state(reference_frame, observer)
    
    #  Read models
    models = dict.fromkeys(model_names, None)
    for model in models.keys():
        models[model] = read_SWModel.choose(model, spacecraft_name, 
                                       starttime, stoptime)
        
        models[model] = models[model].reindex(spacecraft.data.index, 
                                              axis='index', method='nearest')
        
    #!!! Only because HUXt doesn't have Juno yet
    models['HUXt'] = read_SWModel.choose('HUXt', 'Jupiter', starttime, stoptime)
    models['HUXt'] = models['HUXt'].reindex(spacecraft.data.index, axis='index', method='nearest')
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        """
        It would be really ideal to write plot_TaylorDiagram such that there's a 
        function I could call here which "initializes" ax1 (axs[0]) to be a 
        Taylor Diagram-- i.e., I set aside the space in the figure, and then give 
        that axis to a program which adds the axes, labels, names, etc.
        Everything except the data, basically
        
        I guess it would make sense to additionally add a function to plot the 
        RMS difference, rather than doing it with everything else
        """
        #  "Initialize" the plot by plotting a model, but with zero size
        model_timeseries = np.array(models[model_names[0]][tag], dtype='float64')
        sc_timeseries = np.array(spacecraft.data[tag], dtype='float64')
        if plot_kw['yscale'] == 'log':
            model_timeseries = np.log10(model_timeseries)
            sc_timeseries = np.log10(sc_timeseries)
        fig, ax = TD.plot_TaylorDiagram(model_timeseries, 
                                        sc_timeseries,
                                        color='red', marker='X', markersize='0')
        
        for model, model_info in models.items():
            if tag in model_info.columns:
                model_timeseries = np.array(model_info[tag], dtype='float64')
                sc_timeseries = np.array(spacecraft.data[tag], dtype='float64')
                
                if plot_kw['yscale'] == 'log':
                    model_timeseries = np.log10(model_timeseries)
                    sc_timeseries = np.log10(sc_timeseries)
                
                stats, rmse = TD.find_TaylorStatistics(model_timeseries, sc_timeseries)
                
                ax.scatter(np.arccos(stats[0]), stats[1], 
                        marker=model_symbols[model], s=48, c=model_colors[model], edgecolors='black',
                        label=model)
                #print(stats)
                
        ax.set_ylim([0, 1.5*np.nanstd(sc_timeseries)])
        ax.text(0.5, 0.9, ylabel,
                horizontalalignment='center', verticalalignment='top',
                transform = ax.transAxes)
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand')
        ax.set_axisbelow(True)
    
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    for suffix in ['.png', '.pdf']:
        plt.savefig('figures/' + save_filestem, dpi=300, bbox_inches=extent)
    plt.show()
    
    return()

def plot_TaylorDiagram_ConstantTimeWarping(parameter, spacecraft_name, model_names, starttime, stoptime):
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    from matplotlib.transforms import Bbox
    
    import sys
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    match parameter:
        case ('u_mag' | 'flow speed'):
            tag = 'u_mag'
            ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
            plot_kw = {'yscale': 'linear', 'ylim': (0, 60),
                       'yticks': np.arange(350,550+50,100)}
        case ('p_dyn' | 'pressure'):
            tag = 'p_dyn'
            ylabel = r'log Solar Wind Dynamic Pressure $log_{10}(p_{dyn})$ [$log_{10}(nPa)$]'
            plot_kw = {'yscale': 'log', 'ylim': (0, 0.6),
                       'yticks': 10.**np.arange(-3,0+1,1)}
        case ('n_tot' | 'density'):
            tag = 'n_tot'
            ylabel = r'log Solar Wind Ion Density $log_{10}(n_{tot})$ [$log_{10}(cm^{-3})$]'
            plot_kw = {'yscale': 'log', 'ylim': (0, 0.6),
                       'yticks': 10.**np.arange(-2, 0+1, 1)}
        case ('B_mag' | 'magnetic field'):
            tag = 'B_mag'
            ylabel = r'Solar Wind Magnetic Field Magnitude $B_{mag}$ [nT]'
            plot_kw = {'yscale': 'linear', 'ylim': (0, 1),
                       'yticks': np.arange(0, 5+1, 2)}
    #
    save_filestem = 'TaylorDiagram_ConstantTW_{}_{}_{}-{}'.format(spacecraft_name.replace(' ', ''),
                                                       tag,
                                                       starttime.strftime('%Y%m%d'),
                                                       stoptime.strftime('%Y%m%d'))
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime, resolution=99)
    spacecraft.find_state(reference_frame, observer)
    
    shifts = np.arange(-72, 72+6, 6)  #  -/+ hours of shift for the timeseries
    shifted_stats = {name: [] for name in model_names}
    shifted_stats['HUXt'] = []
    for model, shifted_stat in shifted_stats.items():
        for shift in shifts:
            #  Shifting the start and stop times *before* reading the model
            #  means that we don't need to cut the model/data later ->
            #  means more information retained
            shifted_starttime = starttime + dt.timedelta(hours = int(shift))
            shifted_stoptime = stoptime + dt.timedelta(hours = int(shift))
            
            model_info = read_SWModel.choose(model, spacecraft_name, 
                                             shifted_starttime, shifted_stoptime)
            if model == 'HUXt':
                model_info = read_SWModel.choose(model, 'Jupiter', 
                                                 shifted_starttime, shifted_stoptime)
            if len(model_info) > 0:
                model_info.index = model_info.index - dt.timedelta(hours = int(shift))
                model_info = model_info.reindex(spacecraft.data.index, axis='index', method='nearest')

            #if tag in model_info.columns:
                model_timeseries = np.array(model_info[tag], dtype='float64')
                sc_timeseries = np.array(spacecraft.data[tag], dtype='float64')
                
                if plot_kw['yscale'] == 'log':
                    model_timeseries = np.log10(model_timeseries)
                    sc_timeseries = np.log10(sc_timeseries)
                
                stats, rmse = TD.find_TaylorStatistics(model_timeseries, sc_timeseries)
                shifted_stat.append(stats)
            
            else:
                shifted_stat.append((np.nan, np.nan))
            
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        """
        It would be really ideal to write plot_TaylorDiagram such that there's a 
        function I could call here which "initializes" ax1 (axs[0]) to be a 
        Taylor Diagram-- i.e., I set aside the space in the figure, and then give 
        that axis to a program which adds the axes, labels, names, etc.
        Everything except the data, basically
        
        I guess it would make sense to additionally add a function to plot the 
        RMS difference, rather than doing it with everything else
        """
        fig, ax = TD.plot_TaylorDiagram(model_timeseries, 
                                        sc_timeseries,
                                        color='red', marker='X', markersize='0')
        
        for model, shifted_stat in shifted_stats.items():
            if not np.isnan(shifted_stat).all():
                
                plot_thetar = [(np.arccos(s1), s2) for s1, s2 in shifted_stat]
                
                shifted_stat_plot = ax.scatter(*zip(*plot_thetar), 
                                               marker=model_symbols[model], s=24, c=shifts,
                                               zorder=1)
                center_indx = np.where(shifts == 0)[0][0]
                ax.scatter(*plot_thetar[center_indx],
                           marker=model_symbols[model], s=48, c=model_colors[model],
                           edgecolors='black', zorder=2, label=model)
            
        ax.set_ylim([0, 1.5*np.nanstd(sc_timeseries)])
        ax.text(0.5, 0.9, ylabel,
                horizontalalignment='center', verticalalignment='top',
                transform = ax.transAxes)
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand')
        ax.set_axisbelow(True)
        
        plt.colorbar(shifted_stat_plot, location='right', orientation='vertical', 
                     label='Model Shift [hr]',
                     fraction=0.2, pad=0.08, shrink=0.6, aspect=15,
                     ticks=[-72, -48, -24, 0, 24, 48, 72])
        
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1.5, 1.0)
    extent.intervalx = extent.intervalx + 0.5

    for suffix in ['.png', '.pdf']:
        plt.savefig('figures/' + save_filestem, dpi=300, bbox_inches=extent)
    plt.show()
    
    return()
    
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

def MI_ProofOfConcept():
    import matplotlib.pyplot as plt   
    import read_SWModel
    import spacecraftdata    
    import sys
    sys.path.append('/Users/mrutala/code/python/libraries/aaft/')
    sys.path.append('/Users/mrutala/code/python/libraries/generic_MI_lag_finder/')
    import generic_mutual_information_routines as mi_lib
    
    
    n = 1e3
    faketime = np.linspace(0, n, int(n))
    
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(8,6), sharex='col')
    plt.subplots_adjust(left=0.15, bottom=0.05, right=0.95, top=0.95, wspace=0.25, hspace=0.05)
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        #  CASE 1: What does MI give is everything is flat (zero)?
        fakedata_series = np.zeros(int(n))
        fakemodel_series = np.zeros(int(n))
        mi_case1 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[0,0].plot(faketime, fakedata_series, label='data')
        axs[0,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case1
        axs[0,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[0,1].plot(lags, RPS_mi, color='grey')
        
        #  CASE 2: What does MI give is everything is flat (constant)?
        fakedata_series = np.zeros(int(n)) + 10.
        fakemodel_series = np.zeros(int(n)) + 20.
        mi_case2 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[1,0].plot(faketime, fakedata_series, label='data')
        axs[1,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case2
        axs[1,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[1,1].plot(lags, RPS_mi, color='grey')
        
        #  CASE 3: What does MI give with a single defined peak in each timeseries?
        fakedata_series = np.zeros(int(n))
        fakemodel_series = np.zeros(int(n))
        fakedata_series[100:120] = 10.
        fakemodel_series[150:170] = 20.
        mi_case3 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[2,0].plot(faketime, fakedata_series, label='data')
        axs[2,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case3
        axs[2,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[2,1].plot(lags, RPS_mi, color='grey')
        
        #  CASE 4: What does MI give with multiple, regular defined peaks in each timeseries?
        fakedata_series = np.zeros(int(n))
        fakemodel_series = np.zeros(int(n))
        for start in np.arange(0, 1000, 400): fakedata_series[start:start+20] = 10.
        for start in np.arange(0, 1000, 400): fakemodel_series[start+50:start+70] = 20.
        mi_case4 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[3,0].plot(faketime, fakedata_series, label='data')
        axs[3,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case4
        axs[3,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[3,1].plot(lags, RPS_mi, color='grey')
        
        #  CASE 5: What does MI give with multiple, irregular defined peaks in each timeseries?
        fakedata_series = np.zeros(int(n))
        fakemodel_series = np.zeros(int(n))
        for start in np.arange(0, 1000, 400): fakedata_series[start:start+20] = 10.
        for start in np.arange(0, 1000, 400): fakemodel_series[int(start*1.05):int(start*1.05)+20] = 20.
        mi_case5 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[4,0].plot(faketime, fakedata_series, label='data')
        axs[4,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case5
        axs[4,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[4,1].plot(lags, RPS_mi, color='grey')
        
        #  CASE 6: What does MI give with multiple, irregular defined peaks in each timeseries, but with some added variation?
        fakedata_series = np.zeros(int(n))
        fakemodel_series = np.zeros(int(n)) + np.random.normal(0, 0.5, int(n))
        for start in np.arange(0, 1000, 400): fakedata_series[start:start+20] = 10.
        for start in np.arange(0, 1000, 400): fakemodel_series[int(start*1.05):int(start*1.05)+20] = 20.
        mi_case5 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[5,0].plot(faketime, fakedata_series, label='data')
        axs[5,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case5
        axs[5,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[5,1].plot(lags, RPS_mi, color='grey')
        
        #  CASE 6: What does MI give with multiple, irregular defined peaks in each timeseries, but with some added variation?
        fakedata_series = np.zeros(int(n)) + np.random.normal(0, 1.5, int(n))
        fakemodel_series = np.zeros(int(n)) + np.random.normal(0, 0.5, int(n))
        for start in np.arange(0, 1000, 400): fakedata_series[start:start+20] = 10.
        for start in np.arange(0, 1000, 400): fakemodel_series[int(start*1.05):int(start*1.05)+20] = 20.
        mi_case5 = mi_lib.mi_lag_finder(fakedata_series, fakemodel_series, 
                                        temporal_resolution=1, 
                                        max_lag=100, min_lag=-100, 
                                        remove_nan_rows=True, no_plot=True)
        axs[6,0].plot(faketime, fakedata_series, label='data')
        axs[6,0].plot(faketime, fakemodel_series, color='xkcd:gold', label='model')
        lags, mi, RPS_mi, x_sq, x_pw = mi_case5
        axs[6,1].plot(lags, mi, color='indianred', marker='x', linestyle='None')
        axs[6,1].plot(lags, RPS_mi, color='grey')
        
        axs[6,0].set(xlabel='Elapsed time [min.]')
        axs[6,1].set(xlabel='Time lags [min.]')
        
        axs[3,0].text(-0.2, 0.5, 'Signal [arb.]', 
                      horizontalalignment='center', verticalalignment='center', rotation='vertical',
                      transform = axs[3,0].transAxes)
        axs[3,1].text(-0.2, 0.5, 'MI [nats]', 
                      horizontalalignment='center', verticalalignment='center', rotation='vertical',
                      transform = axs[3,1].transAxes)
        
    figurename = 'MI_DeltaFunctions_ProofOfConcept'
    for suffix in ['.png', '.pdf']:
        plt.savefig('figures/' + figurename + suffix, bbox_inches='tight')
    plt.show()

def SWData_MI():
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    import scipy
    import sys
    
    #import read_SWData
    import read_SWModel
    import spacecraftdata
    import plot_TaylorDiagram as TD
    
    import sys
    sys.path.append('/Users/mrutala/code/python/libraries/aaft/')
    sys.path.append('/Users/mrutala/code/python/libraries/generic_MI_lag_finder/')
    import generic_mutual_information_routines as mi_lib
    
    starttime = dt.datetime(2016, 5, 15)
    stoptime = dt.datetime(2016, 6, 20)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    tag = 'u_mag'  #  solar wind property of interest
    spacecraft_names = ['Juno']#, 'Ulysses']
    model_name = 'Tao'
    
    spacecraft_dict = dict.fromkeys(spacecraft_names, None)
    for sc_name in spacecraft_dict.keys():
        sc_info = spacecraftdata.SpacecraftData(sc_name)
        sc_info.read_processeddata(starttime, stoptime, resolution="60Min")
        sc_info.find_state(reference_frame, observer)
        sc_info.find_subset(coord1_range=r_range*sc_info.au_to_km,
                            coord3_range=lat_range*np.pi/180.,
                            transform='reclat')
        spacecraft_dict[sc_name] = sc_info
        sc_columns = sc_info.data.columns
    
    model_dict = dict.fromkeys(spacecraft_names, None)
    for sc_name in model_dict.keys():
        model_info = read_SWModel.Tao(sc_name, starttime, stoptime)
        model_dict[sc_name] = model_info
        model_columns = model_info.columns
      
    spacecraft_data = pd.DataFrame(columns = sc_columns)
    model_data = pd.DataFrame(columns = model_columns)   
    for sc_name in spacecraft_dict.keys():
        spacecraft_data = pd.concat([spacecraft_data, spacecraft_dict[sc_name].data])
        model_data = pd.concat([model_data, model_dict[sc_name]])
    
    model_data = model_data.reindex(spacecraft_data.index, method='nearest')
    
    #return(spacecraft_data, model_data)
    
    #!!! Two options here: align then resample, or resample then align
    #  Resampling first allows the mean to occur first, and then they should
    #  be resampled to the same indices, and align shouldn't change much...
    #juno.data = juno.data.resample("5Min").mean()
    #tao_data = tao_data.resample("5Min").mean()
    
    spacecraft_data = find_RollingDerivativeZScore(spacecraft_data, tag, 2) # !!! vvv
    spacecraft_data['smooth_ddt_'+tag+'_zscore'] = spacecraft_data['smooth_ddt_'+tag+'_zscore'].where(spacecraft_data['smooth_ddt_'+tag+'_zscore'] > 3, 0)
    model_data = find_RollingDerivativeZScore(model_data, tag, 2) # !!! Rolling derivative of the 15 min resample?
    #tr['smooth_ddt_'+tag+'_zscore'] = tr['smooth_ddt_'+tag+'_zscore'].where(tr['smooth_ddt_'+tag+'_zscore'] > 3, 0)
    
    #timespan = dt.timedelta(days=13.5)
    #  Dataframes are indexed so that there's data every 15 minutes
    #  So timedelta can be replaced with the equivalent integer
    #  Which allows step size larger than 1
    timespan_int = int(13.5*24*60 / 60.)
    timestep_int = int(0.5*24*60 / 60.)
    
    sc_windows = spacecraft_data.rolling(timespan_int, min_periods=timespan_int, center=True, step=timestep_int)
    model_windows = model_data.rolling(timespan_int, min_periods=timespan_int, center=True, step=timestep_int)
    
    window_centers = list()
    #rel_mi_lists =  list()
    maxima_list = list()
    for counter, (sc_window, model_window) in enumerate(zip(sc_windows, model_windows)):
        if len(sc_window) == timespan_int:
        
            #with open('MI_lag_finder_output.txt', 'a') as sys.stdout:
            sc_series = np.array(sc_window[tag]).astype(np.float64)
            model_series = np.array(model_window[tag]).astype(np.float64)
            
            mi_out = mi_lib.mi_lag_finder(sc_series, model_series, 
                                          temporal_resolution=60, 
                                          max_lag=4320, min_lag=-4320, 
                                          remove_nan_rows=True, no_plot=True)
            
            sc_series = np.array(sc_window['smooth_ddt_'+tag+'_zscore']).astype(np.float64)
            model_series = np.array(model_window['smooth_ddt_'+tag+'_zscore']).astype(np.float64)
            mi_out_zscore = mi_lib.mi_lag_finder(sc_series, model_series, 
                                                 temporal_resolution=60, 
                                                 max_lag=4320, min_lag=-4320, 
                                                 remove_nan_rows=True, no_plot=True)
            
            with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
                plt.rcParams.update({'font.size': 10})
                fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', width_ratios=(2,1,1), figsize=(8,6))
                plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.0, wspace=0.25)
                
                #  Unmodified timeseries
                axs[0,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                              sc_window[tag], color='gray')
                axs[0,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                              model_window[tag], color=model_colors[model_name])
                axs[0,0].set(ylim=(350,550), ylabel=r'u$_{sw}$ [km/s]')
                
                #  Modified timeseries to highlight jumps
                axs[1,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                              sc_window['smooth_ddt_'+tag+'_zscore'], color='gray')
                axs[1,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                              model_window['smooth_ddt_'+tag+'_zscore'], color=model_colors[model_name])
                axs[1,0].set(xlabel='Elapsed time [min.]',
                             ylim=(-1,12), ylabel=r'$Z(\frac{d u_{sw}}{dt})$')
                
                #  Unmodified timeseries MI
                lags, mi, RPS_mi, x_sq, x_pw = mi_out
                axs[0,1].plot(lags, mi, marker='x', linestyle='None', color='black')
                axs[0,1].plot(lags, RPS_mi, color='xkcd:light gray')
                axs[0,1].set(ylabel='MI (nats)')
                
                #  Modified timeseries MI
                lags, mi, RPS_mi, x_sq, x_pw = mi_out_zscore
                axs[1,1].plot(lags, mi, marker='x', linestyle='None', color='black')
                axs[1,1].plot(lags, RPS_mi, color='xkcd:light gray')
                axs[1,1].set(xlabel='Lags [min.]',
                             ylabel='MI (nats)')
                
                #  Old-school cross-correlation
                lags_cc = scipy.signal.correlation_lags(len(model_series[~np.isnan(model_series)]),
                                                     len(sc_series[~np.isnan(sc_series)]))
                cc = scipy.signal.correlate(model_series[~np.isnan(model_series)],
                                            sc_series[~np.isnan(sc_series)])
                cc /= np.max(cc)
                axs[1,2].scatter(lags_cc, cc)
                axs[1,2].set_xlim(-72, 72)
                
            
            window_centers.append(sc_window.index[int(len(sc_window)*0.5-1)])
            norm_mi = mi / RPS_mi
            norm_mi = np.nan_to_num(norm_mi, nan=0., posinf=0., neginf=0.)
            
            #maxima = scipy.signal.find_peaks(norm_mi, height=10.)
            maxima = scipy.signal.find_peaks(cc, height=0.6)
            
            for m, h in zip(maxima[0], maxima[1]['peak_heights']):
                maxima_list.append((sc_window.index[int(len(sc_window)*0.5-1)], lags_cc[m], h))
            
            #plt.tight_layout()
            #figurename = 'garbage'
            #for suffix in ['.png', '.pdf']:
            #    plt.savefig('figures/' + figurename + suffix)
            # plt.savefig('figures/Juno_MI_frames/frame_' + f'{counter:03}' + '.png')
            plt.show()
                                     
            
            print(counter)
            #if counter == 1000:
                #break
    #indx = np.where(np.array(max_relative_mi) > 0)[0]
    #window_centers = np.array(window_centers)[indx]
    #max_relative_mi = np.array(max_relative_mi)[indx]
    #max_lags = np.array(max_lags)[indx]
    
    #return(window_centers, rel_mi_lists, lag_lists)
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        
        #window_centers_min = [(wc - jr.index[0]).total_seconds()/60. for wc in window_centers]
        fig, ax = plt.subplots(figsize=(8,6))
        # for indx, (rel_mi, lag) in enumerate(zip(rel_mi_lists, lag_lists)):
        #     ax.plot(window_centers, [l/60. for l in lag], marker='o', markersize=6, linestyle='None')
        x, y, c = zip(*maxima_list)
        scatterplot = ax.scatter(x, y, s=36, marker='.', c=c, cmap='viridis', vmax=1)
        ax.set_ylim(-72, 72)
        
        # #window_centers_flattened = window_centers * 10
        # #lag_flattened = [item/60. for sublist in lag_lists for item in sublist]
        # #color_flattened = []
        # #for i in range(10):
        # #    color_flattened.extend([i] * len(window_centers))
        # scatterplot = ax.scatter(window_centers_flattened, lag_flattened, s=36, marker='_', c=color_flattened, cmap='viridis')
        # ax.set(xlabel='Date', ylabel='Time Lags [hours]', ylim=[-72, 72])
        plt.colorbar(scatterplot, label='MI relative to RPS')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.tight_layout()
        plt.show()

    # plt.savefig('figures/Juno_MI_runningoffset_max_only.png', dpi=300)    
    
    return()

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
    import copy
    import matplotlib.pyplot as plt
    #import datetime as dt
    #import numpy as np
    
    #import pandas as pd
    import spiceypy as spice
    
    #import sys
    #sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    #import read_SWData
    import spacecraftdata as SpacecraftData
    
    r_range = [4.9, 5.5]  #  [4.0, 6.4]
    lat_range = [-6.1, 6.1]  #  [-10, 10]
    
    # =============================================================================
    # Find when Ulysses was within +/- 30 deg. of the ecliptic, and near 5 AU
    # =============================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    spacecraft_list = []
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 10'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 11'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Ulysses'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Juno'))
    
    for sc in spacecraft_list:
        # temp_sc = copy.deepcopy(sc)
        # temp_sc.find_lifetime()
        # temp_sc.make_timeseries()
        # temp_sc.find_state(reference_frame, observer)
        # temp_sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, # 4.4 - 6
        #                     coord3_range=np.array(lat_range)*np.pi/180., 
        #                     transform='reclat')
        # startdate = temp_sc.data.index[0]
        # stopdate = temp_sc.data.index[-1]
        sc.read_processeddata(everything=True)
        sc.find_state(reference_frame, observer)
        sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, # 4.4 - 6
                       coord3_range=np.array(lat_range)*np.pi/180., 
                       transform='reclat')
    
    spacecraft_labels = {'Pioneer 10': 1,
                         'Pioneer 11': 2,
                         'Voyager 1': 3, 
                         'Voyager 2': 4, 
                         'Ulysses'  : 5, 
                         'Juno'     : 6}
    
    # =============================================================================
    # F10.4 Radio flux from the sun
    # =============================================================================
    column_headers = ('date', 'observed_flux', 'adjusted_flux',)
    solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                   header = 0, names = column_headers)
    solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                for d in solar_radio_flux['date']]

    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(nrows=2, height_ratios=(1,2), figsize=(6,4.5), sharex='col')
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        axs[1].plot(solar_radio_flux['date'], solar_radio_flux['observed_flux'], 
                    marker='.', color='gray', markersize=2, linestyle='None')
        
        for sc in spacecraft_list:
            
            #  This bit splits the spacecraft data up into chunks to plot
            sc_deltas = sc.data.index.to_series().diff()
            gaps_indx = np.where(sc_deltas > dt.timedelta(days=30))[0]
            start_indx = np.insert(gaps_indx, 0, 0)
            stop_indx = np.append(gaps_indx, len(sc.data.index))-1
            
            print('Start and stop dates for ' + sc.name)
            for first, final in zip(start_indx, stop_indx):
                print(sc.data.index[first].strftime('%Y-%m-%d') + 
                      ' -- ' + 
                      sc.data.index[final].strftime('%Y-%m-%d'))
                print(str((sc.data.index[final] - sc.data.index[first]).total_seconds()/3600.) + ' total hours')
                rlonlat = [spice.reclat(np.array(row[['x_pos', 'y_pos', 'z_pos']], dtype='float64')) for indx, row in sc.data.iterrows()]
                rlonlat = np.array(rlonlat).T
                print('Radial range of: ' + str(np.min(rlonlat[0,:])/sc.au_to_km) +
                      ' - ' + str(np.max(rlonlat[0,:])/sc.au_to_km) + ' AU')
                print('Heliolatitude range of: ' + str(np.min(rlonlat[2,:])*180/np.pi) +
                      ' - ' + str(np.max(rlonlat[2,:])*180/np.pi) + 'deg.')
            print('------------------------------------------')
            # rectangle_list = []
            # for x0, xf in zip(np.insert(start_indx, 0, 0), stop_indx):
            #     ll_corner = (sc.data.index[x0].timestamp(), spacecraft_labels[sc.name]-0.333)
            #     width = sc.data.index[xf-1].timestamp() - sc.data.index[x0].timestamp()
            #     height = 0.666
                
            #     rectangle_list.append(patch.Rectangle(ll_corner, width, height))
            # ax0twin.add_collection(PatchCollection(rectangle_list))
            
            axs[0].plot(sc.data.index, np.zeros(len(sc.data.index))+spacecraft_labels[sc.name], 
                        linestyle='None', marker='|', markersize=12, color=spacecraft_colors[sc.name])
            
        axs[0].set_xlim((dt.datetime(1970, 1, 1), dt.datetime(2020, 1, 1)))
        
        axs[1].set_ylabel('Observed Radio Flux @ 10.7 cm [SFU]')
        axs[1].set_xlabel('Year')
        #ax0.set_title(r'Spacecraft data availability relative to the Solar Cycle between 4.5-5.5 $R_J$ and -10-10 deg. lat.', wrap=True, fontsize=18)
        
        axs[1].set_ylim((50, 300))
        
        axs[0].set_ylim((0,7))
        axs[0].set_yticks(list(spacecraft_labels.values()))
        axs[0].set_yticklabels(list(spacecraft_labels.keys()))
        axs[0].set_ylabel('Spacecraft')
        
        #plt.tight_layout()
        figurename = 'Coverage_vs_SolarCycle'
        for suffix in ['.png', '.pdf']:
            plt.savefig('figures/' + figurename + suffix, bbox_inches='tight')
        plt.show()
        spice.kclear()
        
def plot_SpacecraftSpatialCoverage():
    import matplotlib.pyplot as plt
    import copy
    import spiceypy as spice
    import spacecraftdata as SpacecraftData
    
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    observer='Solar System Barycenter'
    reference_frame = 'ECLIPJ2000'

    spacecraft_list = []
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 10'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 11'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Ulysses'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Juno'))
    
    for sc in spacecraft_list:
        #sc.find_lifetime()
        #sc.make_timeseries()
        sc.read_processeddata(everything=True)
        sc.find_state(reference_frame, observer, keep_kernels=True)
        
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        

    
    for sc in spacecraft_list:
        #  For each spacecraft, get the positions of Earth and Jupiter
        times = [spice.datetime2et(t) for t in sc.data.index]
        print(sc.name)
        print(sc.data.index[0])
        print(sc.data.index[-1])
        
        ear_pos, ear_lt = spice.spkpos('EARTH BARYCENTER', times, reference_frame, 'NONE', observer)
        ear_pos = np.array([spice.recsph(row / sc.au_to_km) for row in ear_pos])
        
        jup_pos, jup_lt = spice.spkpos('JUPITER BARYCENTER', times, reference_frame, 'NONE', observer)
        jup_pos = np.array([spice.recsph(row / sc.au_to_km) for row in jup_pos])
        
        xyz_in_AU = [np.array(row[['x_pos', 'y_pos', 'z_pos']]).astype('float64')
                              for indx, row in sc.data.iterrows()]
        sph_in_AU = np.array([spice.recsph(row / sc.au_to_km) for row in xyz_in_AU])
        #print(sph_in_AU)
        #xyz_full_in_AU = np.array([row[['x_pos', 'y_pos', 'z_pos']]/sc.au_to_km
        #                      for indx, row in sc_full.data.iterrows()])
        
        #ax.plot(xyz_full_in_AU[:,0], xyz_full_in_AU[:,1],
        #        color='gray', linewidth=1)
        ax.plot((sph_in_AU[:,2]-jup_pos[:,2]), sph_in_AU[:,0], 
                  label=sc.name, linewidth=1, color=spacecraft_colors[sc.name])
        
        ax.plot((ear_pos[:,2]-jup_pos[:,2]), ear_pos[:,0], 
                label='Earth', color='xkcd:kelly green', marker='o', markersize=1, linestyle='None')
        #ax.plot(ear_pos[-1,0], ear_pos[-1,1],  
        #        color='xkcd:kelly green', marker='o', markersize=4)
        
        ax.plot((jup_pos[:,2]-jup_pos[:,2]), jup_pos[:,0], 
                label='Jupiter', color='xkcd:peach', marker='o', markersize=1, linestyle='None')
        
    
    ax.set_aspect(1)
    ax.set_rlim((0, 6))

    return(jup_pos)
    
    
    #ax.plot(jup_pos[-1,0], jup_pos[-1,1],
    #        color='xkcd:peach', marker='o', markersize=4)
    
    #ax.set(xlim=[-6,6], 
    #        ylim=[-6,6])
    
    plt.show()
    spice.kclear()
    
def plot_spacecraft_spatial_coverage():
    import matplotlib.pyplot as plt
    import copy
    import spiceypy as spice
    import spacecraftdata as SpacecraftData
    
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    spacecraft_list = []
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 10'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 11'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Ulysses'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Juno'))
    
    for sc in spacecraft_list:
        temp_sc = copy.deepcopy(sc)
        temp_sc.find_lifetime()
        temp_sc.make_timeseries()
        temp_sc.find_state(reference_frame, observer)
        temp_sc.find_subset(coord1_range=np.array((4.0, 6.4))*sc.au_to_km, # 4.4 - 6
                            coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                            transform='reclat')
        startdate = temp_sc.data.index[0]
        stopdate = temp_sc.data.index[-1]
        sc.read_processeddata(startdate, stopdate)
        sc.find_state(reference_frame, observer)
        sc.find_subset(coord1_range=np.array((4.0, 6.4))*sc.au_to_km, # 4.4 - 6
                       coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                       transform='reclat')
        
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    #fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #plt.grid(False)
    #plt.xlabel("Ecliptic X")
    #plt.ylabel("Ecliptic Y")
    for ax, sc in zip(axs.reshape(-1),spacecraft_list):
        
        observer='Solar System Barycenter'
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
                  label=sc.name, marker='o', markersize=1, linestyle='None', color=spacecraft_colors[sc.name])
        
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
        
        ax.set(xlim=[-6,6], 
                ylim=[-6,6])
    



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

def find_BestTemporalShifts():
    """
    Use both mutual information (MI) and cross correlation (CC) to find the 
    optimal temporal shifts of a model relative to data.

    Returns
    -------
    The centers of the temporal window used to find optimal shifts
    The time lags of those optimal shifts
    A relative estimate of the significance of each suggested shift (there may
    be multiple optimal shifts per window)

    """
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    import scipy
    
    import read_SWModel
    import spacecraftdata
    
    import sys
    sys.path.append('/Users/mrutala/code/python/libraries/aaft/')
    sys.path.append('/Users/mrutala/code/python/libraries/generic_MI_lag_finder/')
    import generic_mutual_information_routines as mi_lib
    
    # =============================================================================
    #    Would-be inputs go here 
    # =============================================================================
    tag = 'u_mag'  #  solar wind property of interest
    spacecraft_name = 'Juno' # ['Ulysses', 'Juno'] # 'Juno'
    model_name = 'Tao'
    starttime = dt.datetime(1990, 1, 1)
    stoptime = dt.datetime(2016, 7, 1)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    # =============================================================================
    #  Would-be keywords go here
    # =============================================================================
    window_in_days = 13.5
    stepsize_in_days = 0.5
    resolution_in_min = 60.
    max_lag_in_days = 6.0
    min_lag_in_days = -6.0
    
    #  Parse inputs and keywords
    resolution_str = '{}Min'.format(int(resolution_in_min))
    
    # =============================================================================
    #  Read in data; for this, we expect n spacecraft and 1 model
    # =============================================================================
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    
    sc_info = spacecraftdata.SpacecraftData(spacecraft_name)
    sc_info.read_processeddata(starttime, stoptime, resolution=resolution_str, combined=True)
    sc_info.find_state(reference_frame, observer, keep_kernels=True)
    sc_info.find_subset(coord1_range=r_range*sc_info.au_to_km,
                        coord3_range=lat_range*np.pi/180.,
                        transform='reclat')
        
    spice.furnsh(sc_info.SPICE_METAKERNEL)
    ang_datetimes = sc_info.data.index  
    #np.arange(starttime, stoptime, dt.timedelta(days=1)).astype(dt.datetime)
    ets = spice.datetime2et(ang_datetimes)
    ang = [np.abs(spice.trgsep(et, sc_info.name, 'POINT', None, 'Earth', 'POINT', None, 'Sun', 'None')*180/np.pi) for et in ets]
    sc_info.data['a_tse'] = ang
    
    sc_columns = sc_info.data.columns
    
    spice.kclear()
    
    model_info = read_SWModel.Tao(spacecraft_name, starttime, stoptime)
    model_columns = model_info.columns
      
    spacecraft_data = sc_info.data
    model_output = model_info 
    
    #  Make model index match spacecraft index, getting rid of excess
    model_output = model_output.reindex(spacecraft_data.index, method='nearest')
    
    # =============================================================================
    #     
    # =============================================================================
    sigma_cutoff = 4
    spacecraft_data = find_RollingDerivativeZScore(spacecraft_data, tag, 1) # !!! vvv
    spacecraft_data['smooth_ddt_'+tag+'_zscore'] = spacecraft_data['smooth_ddt_'+tag+'_zscore'].where(spacecraft_data['smooth_ddt_'+tag+'_zscore'] > sigma_cutoff, 0)
    spacecraft_data['smooth_ddt_'+tag+'_zscore'] = spacecraft_data['smooth_ddt_'+tag+'_zscore'].where(spacecraft_data['smooth_ddt_'+tag+'_zscore'] < sigma_cutoff, sigma_cutoff)
    
    def model_features(model_output):
        model_output = find_RollingDerivativeZScore(model_output, tag, 2) # !!! Rolling derivative of the 15 min resample?
        model_output['smooth_ddt_'+tag+'_zscore'] = model_output['smooth_ddt_'+tag+'_zscore'].where(model_output['smooth_ddt_'+tag+'_zscore'] > sigma_cutoff, 0)
        model_output['smooth_ddt_'+tag+'_zscore'] = model_output['smooth_ddt_'+tag+'_zscore'].where(model_output['smooth_ddt_'+tag+'_zscore'] < sigma_cutoff, sigma_cutoff)
        return(model_output)
    model_output = model_features(model_output)
    
    #  Split the combined spacecraft data set into (mostly) continuous chunks
    spacecraft_deltas = spacecraft_data.index.to_series().diff()
    gaps_indx = np.where(spacecraft_deltas > dt.timedelta(days=30))[0]
    start_indx = np.insert(gaps_indx, 0, 0)
    stop_indx = np.append(gaps_indx, len(spacecraft_data.index))-1
    
    spacecraft_data_list = []
    model_output_list = []
    for i0, i1 in zip(start_indx, stop_indx):
        spacecraft_data_list.append(spacecraft_data[i0:i1])
        model_output_list.append(model_output[i0:i1])
    
    #  Dataframes are indexed so that there's data every 15 minutes
    #  So timedelta can be replaced with the equivalent integer
    #  Which allows step size larger than 1
    timespan_int = int(window_in_days*24*60 / resolution_in_min)
    timestep_int = int(stepsize_in_days*24*60 / resolution_in_min)
    max_lag_in_min = max_lag_in_days*24*60.
    min_lag_in_min = min_lag_in_days*24*60.
    
    window_centers = list()
    maxima_list = list()
    
    output_df = pd.DataFrame(columns=['window_center', 'lag', 'correlation', 'false_positives', 'false_negatives'])
    for sc_data, m_output in zip(spacecraft_data_list, model_output_list):
            
        sc_windows = sc_data.rolling(timespan_int, min_periods=timespan_int, center=True, step=timestep_int)
        m_windows = m_output.rolling(timespan_int, min_periods=timespan_int, center=True, step=timestep_int)
        
        for counter, (sc_window, m_window) in enumerate(zip(sc_windows, m_windows)):
            if len(sc_window) == timespan_int:
                
                sc_series = np.array(sc_window[tag]).astype(np.float64)
                m_series = np.array(m_window[tag]).astype(np.float64)
                
                # mi_out = mi_lib.mi_lag_finder(sc_series, m_series, 
                #                               temporal_resolution=resolution_in_min, 
                #                               max_lag=max_lag_in_min, 
                #                               min_lag=min_lag_in_min, 
                #                               remove_nan_rows=True, no_plot=True)        
                sc_z_series = np.array(sc_window['smooth_ddt_'+tag+'_zscore']).astype(np.float64)
                m_z_series = np.array(m_window['smooth_ddt_'+tag+'_zscore']).astype(np.float64)
                # mi_out_z = mi_lib.mi_lag_finder(sc_z_series, m_z_series, 
                #                                 temporal_resolution=resolution_in_min, 
                #                                 max_lag=max_lag_in_min,
                #                                 min_lag=min_lag_in_min, 
                #                                 remove_nan_rows=True, no_plot=True)
                
                #  Define old-school cross-correlation
                def crosscorrelate_withnan(a, b, temporal_resolution=1, min_lag=1, max_lag=1):
                    #  Assume temporal_resolution, min_lag, max_lag are in min.
                    a, b = np.array(a), np.array(b)
                    a_indx = ~np.isnan(a)
                    a = a[a_indx]
                    b = b[a_indx]
                    b_indx = ~np.isnan(b)
                    a = a[b_indx]
                    b = b[b_indx]
                    
                    na, nb = len(a), len(b)
                    
                    corr = scipy.signal.correlate(a, b)
                    lags = scipy.signal.correlation_lags(na, nb)*temporal_resolution
                    
                    indx = np.where(np.logical_and(lags > min_lag, lags < max_lag))
                    corr, lags = corr[indx], lags[indx]
                    corr /= np.max(corr)
                    
                    return(lags, corr)
                try: 
                    cc_out = crosscorrelate_withnan(sc_series, m_series,
                                                    temporal_resolution = resolution_in_min,
                                                    max_lag=max_lag_in_min,
                                                    min_lag=min_lag_in_min)
                    cc_out_z = crosscorrelate_withnan(sc_z_series, m_z_series,
                                                      temporal_resolution = resolution_in_min,
                                                      max_lag=max_lag_in_min,
                                                      min_lag=min_lag_in_min)
                except:
                    print('sc_series')
                    print(sc_series)
                    print('m_series')
                    print(m_series)
                    fig, ax = plt.subplots()
                    ax.plot(sc_window.index, sc_series)
                    ax.plot(m_window.index, m_series)
                    plt.show()
                
                with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
                    plt.rcParams.update({'font.size': 10})
                    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', width_ratios=(3,1), figsize=(8,6))
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.05, wspace=0.325)
                    
                    #  Unmodified timeseries
                    axs[0,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                                  sc_window[tag], color='gray')
                    axs[0,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                                  m_window[tag], color=model_colors[model_name])
                    axs[0,0].set(ylim=(250,600), ylabel=r'u$_{sw}$ [km/s]')
                    
                    #  Modified timeseries to highlight jumps
                    axs[1,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                                  sc_window['smooth_ddt_'+tag+'_zscore'], color='gray')
                    axs[1,0].plot((sc_window.index-spacecraft_data.index[0]).total_seconds()/60.,
                                  m_window['smooth_ddt_'+tag+'_zscore'], color=model_colors[model_name])
                    axs[1,0].set(xlabel='Elapsed time [min.]',
                                  ylim=(-1,12), ylabel=r'$Z(\frac{d u_{sw}}{dt})$')
                    
                    # #  Unmodified timeseries MI
                    # lags, mi, RPS_mi, x_sq, x_pw = mi_out
                    # axs[0,1].plot(lags, mi, marker='x', linestyle='None', color='black')
                    # axs[0,1].plot(lags, RPS_mi, color='xkcd:light gray')
                    # axs[0,1].set(ylabel='MI (nats)')
                    
                    # #  Modified timeseries MI
                    # lags, mi, RPS_mi, x_sq, x_pw = mi_out_z
                    # axs[1,1].plot(lags, mi, marker='x', linestyle='None', color='black')
                    # axs[1,1].plot(lags, RPS_mi, color='xkcd:light gray')
                    # axs[1,1].set(xlabel='Lags [min.]',
                    #               ylabel='MI [nats]')
                    
                    
                    axs[0,1].scatter(cc_out[0], cc_out[1], s=24)
                    axs[0,1].set(xlabel='Lags [min].', 
                                  ylabel='Cross Correlation')
                    
                    axs[1,1].scatter(cc_out_z[0], cc_out_z[1], s=24)
                    axs[1,1].set(xlabel='Lags [min].', 
                                  ylabel='Cross Correlation')
                    
                plt.show()
                
                maxima = scipy.signal.find_peaks(cc_out_z[1], height=0.5)
                
                # =============================================================================
                #  Lag the time series by the peak cross-correlation to characterize errors         
                # =============================================================================
                #maximal_cc_indx = maxima[0][np.where(maxima[1]['peak_heights'] == 1.0)]
                optimal_shifts = cc_out_z[0][maxima[0]]
                
                n_false_positives = list()
                n_false_negatives = list()
                for shift in optimal_shifts:
                    
                    shift_start = m_window.index[0] - dt.timedelta(minutes=shift)
                    shift_stop = m_window.index[-1] - dt.timedelta(minutes=shift)
                    shift_indx = np.where((m_output.index >= shift_start) 
                                          & (m_output.index < shift_stop))[0]
                    new_m_window = read_SWModel.Tao(spacecraft_name, shift_start, shift_stop) #m_output.iloc[shift_indx] 
                    new_m_window = model_features(new_m_window)
                    new_m_window.index += dt.timedelta(minutes=shift)
                    
                    shifted_sc_series = sc_window['smooth_ddt_'+tag+'_zscore']
                    shifted_m_series = new_m_window['smooth_ddt_'+tag+'_zscore']
                    if len(shifted_m_series) < len(shifted_sc_series):
                        shifted_sc_series = shifted_sc_series.reindex(shifted_m_series.index)
                    shifted_datetime = shifted_sc_series.index
                    shifted_sc_series, shifted_m_series = shifted_sc_series.to_numpy(), shifted_m_series.to_numpy()
                    
                    try:    
                        residuals = shifted_sc_series - shifted_m_series
                    except:
                        print('RESIDUALS COULD NOT BE CALCULATED')
                        return(sc_window, new_m_window)
                    
                    # fig, axs = plt.subplots(nrows = 2, height_ratios=[2,1], sharex='col')
                    # # axs[0].plot((shifted_datetime-spacecraft_data.index[0]).total_seconds()/60.,
                    # #             shifted_sc_series, color='gray')
                    # # axs[0].plot((shifted_datetime-spacecraft_data.index[0]).total_seconds()/60.,
                    # #             shifted_m_series, color=model_colors[model_name])
                    
                    # # axs[1].plot((shifted_datetime-spacecraft_data.index[0]).total_seconds()/60.,
                    # #             residuals)
                    
                    # axs[0].plot(np.arange(0, len(shifted_datetime)),
                    #             shifted_sc_series, color='gray')
                    # axs[0].plot(np.arange(0, len(shifted_datetime)),
                    #             shifted_m_series, color=model_colors[model_name])
                    
                    # axs[1].plot(np.arange(0, len(shifted_datetime)),
                    #             residuals)
                    # plt.show()
                    
                    false_positives = list()
                    #  Check all model peaks against the residuals for false positives
                    m_peak_indx = np.where(shifted_m_series > 0)[0]
                    m_peak_span_indxs = np.split(m_peak_indx, np.where(np.diff(m_peak_indx) > 1)[0] + 1)
                    for span in m_peak_span_indxs:
                        m_sum = np.sum(shifted_m_series[span])
                        r_sum = np.sum(residuals[span])
                        if m_sum == -r_sum:
                            false_positives.append(span)
                    
                    false_negatives = list()
                    #  Check all data peaks against the residuals for false negatives
                    sc_peak_indx = np.where(shifted_sc_series > 0)[0]
                    sc_peak_span_indxs = np.split(sc_peak_indx, np.where(np.diff(sc_peak_indx) > 1)[0] + 1)
                    for span in sc_peak_span_indxs:
                        sc_sum = np.sum(shifted_sc_series[span])
                        r_sum = np.sum(residuals[span])
                        if sc_sum == r_sum:
                            false_negatives.append(span)
                            
                    # false_neg_indx = np.where((residuals == shifted_sc_series) & 
                    #                      (shifted_sc_series != 0))[0]
                    # false_negatives = np.split(false_neg_indx, np.where(np.diff(false_neg_indx) > 1)[0] + 1)
                    #
                    # false_pos_indx = np.where((residuals == -shifted_m_series) &
                    #                           (shifted_m_series != 0))[0]
                    # false_positives = np.split(false_pos_indx, np.where(np.diff(false_pos_indx) >1)[0] + 1)
                    
                    n_false_positives.append(len(false_positives))
                    n_false_negatives.append(len(false_negatives))
                    #print("False Negatives: " + str(len(false_negatives)))
                    #print("False Positives: " + str(len(false_positives)))
                
                
                # =============================================================================
                #  Record important things to return      
                # =============================================================================
                window_center = sc_window.index[int(len(sc_window)*0.5-1)]
                #window_centers.append(window_center)
                #norm_mi = mi / RPS_mi
                #norm_mi = np.nan_to_num(norm_mi, nan=0., posinf=0., neginf=0.)
                
                #maxima = scipy.signal.find_peaks(norm_mi, height=10.)
                
                
                #for m, h in zip(maxima[0], maxima[1]['peak_heights']):
                #    maxima_list.append((sc_window.index[int(len(sc_window)*0.5-1)], cc_out_z[0][m], h))
                
                #plt.tight_layout()
                #figurename = 'garbage'
                #for suffix in ['.png', '.pdf']:
                #    plt.savefig('figures/' + figurename + suffix)
                # plt.savefig('figures/Juno_MI_frames/frame_' + f'{counter:03}' + '.png')
                
                d = {'window_center'     : [window_center for i in range(len(maxima[0]))],
                     'lag'               : cc_out_z[0][maxima[0]],
                     'correlation'       : maxima[1]['peak_heights'],
                     'false_positives'   : n_false_positives,
                     'false_negatives'   : n_false_negatives}  
                output_df = pd.concat([output_df, pd.DataFrame.from_dict(d)], 
                                      ignore_index=True)
                
                print(counter)
                
    return(output_df, spacecraft_data, model_output)
    
def plot_BestTemporalShifts(maxima, spacecraft_data, model_output):
    import matplotlib.pyplot as plt
    import scipy
    import spiceypy as spice
    import datetime as dt
    
    import matplotlib.dates as mdates

    #starttime, stoptime = dt.datetime(2016, 5, 15), dt.datetime(2016, 7, 1) 
    #starttime, stoptime = dt.datetime(2003,10,1), dt.datetime(2004,7,1)
    starttime, stoptime = spacecraft_data.index[0], spacecraft_data.index[-1]
    
    maxima = maxima.iloc[np.where((maxima.window_center >= starttime) &
                                  (maxima.window_center < stoptime))]
    maximal_correlation = maxima.iloc[np.where(maxima.correlation == 1.0)]
    
    #corr = scipy.stats.pearsonr(np.abs(y_at_c1), spacecraft_data['a_tse'][np.where(np.array(c) == 1.0)[0]])
    #print(corr)
    
    #fig, ax = plt.subplots()
    #ax.scatter(np.abs(y_at_c1), spacecraft_data['a_tse'][np.where(np.array(c) == 1.0)[0]])
    #plt.show()
    
    coloring = maxima.correlation
    coloring = (maxima.false_positives + maxima.false_negatives)
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        
        #window_centers_min = [(wc - jr.index[0]).total_seconds()/60. for wc in window_centers]
        fig, axs = plt.subplots(nrows=2, ncols=2, width_ratios=(6,1), figsize=(8,6), sharex='col', sharey=True)
        plt.subplots_adjust(left=0.15, bottom=0.1, right=1.0, hspace=0.1, wspace=0.3)
        
        seconds = [(t - spacecraft_data.index[0]).total_seconds() for t in maxima.window_center]
        
        def line(t, a, b):
            f_t = t*a + b
            return(f_t)
        
        def poly(t, a, b, c):
            f_t = a*t**2 + b*t + c
            return(f_t)
        
        print(len(seconds))
        print(len(maxima.lag))
        print(len(maxima.correlation))
        fit_opt, fit_cov = scipy.optimize.curve_fit(poly, seconds, maxima.lag, sigma=(1./np.array(maxima.correlation))**2)
        print(fit_opt)
        
        for ax in axs[:,0]:
            scatterplot = ax.scatter(maxima.window_center, maxima.lag, c=coloring,
                                     s=36, marker='.', cmap='viridis')
            ax.set_ylim(-6*24*60., 6*24*60.)
            ax.yaxis.set_major_formatter(lambda maxima_index, pos: str(maxima_index/60.))
            ax.set_yticks(np.arange(-6*24, 6*24+48, 48) * 60.)
            ax.set_xlim(starttime, stoptime)
            
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 8, 11)))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            
            #ax.plot(x, [line(t, fit_opt[0], fit_opt[1]) for t in xs])
            #ax.plot(x, [poly(t, *fit_opt) for t in xs], color='xkcd:sky blue')
            
            ax.axhline(np.mean(maxima.lag), color='xkcd:lilac', linestyle='--')
            ax.axhline(np.mean(maximal_correlation.lag), color='gold', linestyle='--')
        
        counts_lo, bins_lo = np.histogram(maxima.lag, bins=np.arange(-6*24*60., 6*24*60.+12*60, 12*60))
        counts_hi, bins_hi = np.histogram(maximal_correlation.lag, bins=np.arange(-6*24*60., 6*24*60.+12*60, 12*60))
        for ax in axs[:,1]:
            ax.stairs(counts_lo, bins_lo, orientation='horizontal', color='xkcd:lilac')
            ax.stairs(counts_hi, bins_hi, orientation='horizontal', color='gold')
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
            ax.set_ylabel('Occurence of Best-fit Shifts')
            ax.yaxis.set_label_position("right")
            
            ax.axhline(np.mean(maxima.lag), color='xkcd:lilac', linestyle='--')
            ax.axhline(np.mean(maximal_correlation.lag), color='gold', linestyle='--')
            
        axs[1,0].set_xlabel('Date')
        
        # =============================================================================
        # F10.4 Radio flux from the sun
        # =============================================================================
        column_headers = ('date', 'observed_flux', 'adjusted_flux',)
        solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                       header = 0, names = column_headers)
        solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                    for d in solar_radio_flux['date']]
        axs_0_y = axs[0,0].twinx()
        axs_0_y.plot(solar_radio_flux['date'], solar_radio_flux['adjusted_flux'], alpha=0.5)
        axs_0_y.set_ylim(50, 300)
        axs_0_y.set_ylabel('F10.7 Radio Flux [SFU]')
        
        # =============================================================================
        # Spacecraft-Sun-Earth Angle
        # =============================================================================
        #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
        #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
        #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
        #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
        #spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
        #spice.furnsh(spacecraft_dict['Juno'].SPICE_METAKERNEL)
        
        #ang_datetimes = np.arange(starttime, stoptime, dt.timedelta(days=1)).astype(dt.datetime)
        #ets = spice.datetime2et(ang_datetimes)
        #ang = [np.abs(spice.trgsep(et, 'Juno', 'POINT', None, 'Earth', 'POINT', None, 'SUN', 'None')*180/np.pi) for et in ets]
        #spice.kclear()
        
        axs_1_y = axs[1,0].twinx()
        axs_1_y.plot(spacecraft_data.index, spacecraft_data['a_tse'], alpha=0.5)
        axs_1_y.set_ylim(0, 180)
        axs_1_y.set_yticks([0, 30, 60, 90, 120, 150, 180])
        axs_1_y.set_ylabel('TSE Angle [deg.]')
        
        plt.text(0.02, 0.5, 'Best-fit Temporal Shift (from cross correlation) [hr]', 
                 fontsize=14, transform=plt.gcf().transFigure, rotation=90, va='center')
        plt.colorbar(scatterplot, label='Normalized Cross Correlation', ax=axs.ravel().tolist(), pad=0.05)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        
        #fig.tight_layout()
        plt.show()

    # plt.savefig('figures/Juno_MI_runningoffset_max_only.png', dpi=300)    
    
    #return(maxima_list)


# def find_FalseFeatures(maxima_list, spacecraft_data, model_output):
    
#     #  Split the combined spacecraft data set into (mostly) continuous chunks
#     spacecraft_deltas = spacecraft_data.index.to_series().diff()
#     gaps_indx = np.where(spacecraft_deltas > dt.timedelta(days=30))[0]
#     start_indx = np.insert(gaps_indx, 0, 0)
#     stop_indx = np.append(gaps_indx, len(spacecraft_data.index))-1
    
#     spacecraft_data_list = []
#     model_output_list = []
#     for i0, i1 in zip(start_indx, stop_indx):
#         spacecraft_data_list.append(spacecraft_data[i0:i1])
#         model_output_list.append(model_output[i0:i1])
    
#     #  Dataframes are indexed so that there's data every 15 minutes
#     #  So timedelta can be replaced with the equivalent integer
#     #  Which allows step size larger than 1
#     timespan_int = int(window_in_days*24*60 / resolution_in_min)
#     timestep_int = int(stepsize_in_days*24*60 / resolution_in_min)
#     max_lag_in_min = max_lag_in_days*24*60.
#     min_lag_in_min = min_lag_in_days*24*60.
    
#     window_centers = list()
#     maxima_list = list()
#     for sc_data, m_output in zip(spacecraft_data_list, model_output_list):
            
#         sc_windows = sc_data.rolling(timespan_int, min_periods=timespan_int, center=True, step=timestep_int)
#         m_windows = m_output.rolling(timespan_int, min_periods=timespan_int, center=True, step=timestep_int)
        
#         for counter, (sc_window, m_window) in enumerate(zip(sc_windows, m_windows)):
#             if len(sc_window) == timespan_int: