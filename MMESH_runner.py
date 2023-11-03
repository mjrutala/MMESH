#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:56:40 2023

@author: mrutala
"""

import datetime as dt
import pandas as pd
import numpy as np

import spacecraftdata
import read_SWModel
import matplotlib.pyplot as plt

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
                'ENLIL'  : '#DCED96',
                'ensemble': '#55FFFF'}
model_symbols = {'Tao'    : 'v',
                'HUXt'   : '*',
                'SWMF-OH': '^',
                'MSWIM2D': 'd',
                'ENLIL'  : 'h',
                'ensemble': 'o'}

r_range = np.array((4.9, 5.5))  #  [4.0, 6.4]
lat_range = np.array((-6.1, 6.1))  #  [-10, 10]


def MMESH_traj_run():
    import string
    
    import sys
    sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    import MMESH as mmesh
    import numpy as np
    import scipy.stats as scstats
    import scipy.optimize as optimize
    import matplotlib.pyplot as plt
    import plot_TaylorDiagram as TD
    
    from sklearn.linear_model import LinearRegression
    
    from numpy.polynomial import Polynomial
    
    spacecraft_name = 'Juno' # 'Ulysses'
    model_names = ['Tao', 'HUXt', 'ENLIL']
    
    #   !!!! This whole pipline is currently (2023 11 03) sensitive to these
    #   input dates, but shouldn't be-- Optimization should not change based on input dates
    #   as long as the input dates span the whole of the spacecraft data being used
    #   since we should always be dropping NaN rows...
    starttime = dt.datetime(2016, 5, 16) # dt.datetime(1997,8,14) # dt.datetime(1991,12,8) # 
    stoptime = dt.datetime(2016, 6, 26) # dt.datetime(1998,1,1) # dt.datetime(1992,2,2) # 
    
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
    spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
    pos_TSE = spacecraft.find_StateToEarth()
    
    #!!!!
    #  NEED ROBUST in-SW-subset TOOL! ADD HERE!
    
    #  Change start and stop times to reflect spacecraft limits-- with optional padding
    padding = dt.timedelta(days=8)
    starttime = spacecraft.data.index[0] - padding
    stoptime = spacecraft.data.index[-1] + padding
    
    #  Initialize a trajectory class and add the spacecraft data
    traj0 = mmesh.Trajectory()
    traj0.addData(spacecraft_name, spacecraft.data)
    
    # traj0.addTrajectory(x)
    
    #  Read models and add to trajectory
    for model_name in model_names:
        model = read_SWModel.choose(model_name, spacecraft_name, 
                                    starttime, stoptime, resolution='60Min')
        traj0.addModel(model_name, model)
    
    # =============================================================================
    #   Plot a Taylor Diagram of the unchanged models
    # =============================================================================
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig = plt.figure(figsize=(6,4.5))
        fig, ax = traj0.plot_TaylorDiagram(tag_name='u_mag', fig=fig)
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.05,1.0,0.15], 
                  loc='lower left', mode='expand', markerscale=1.0)
        plt.show()
    
    # =============================================================================
    #   Optimize the models via constant temporal shifting
    #   Then plot:
    #       - The changes in correlation coefficient 
    #         (more generically, whatever combination is being optimized)
    #       - The changes on a Taylor Diagram
    # =============================================================================
    shifts = np.arange(-96, 96+6, 6)    #  in hours
    shift_stats = traj0.optimize_shifts('u_mag', shifts=shifts)
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        traj0.plot_ConstantTimeShifting_Optimization()
        
        fig = plt.figure(figsize=[6,4.5])
        traj0.plot_ConstantTimeShifting_TD(fig=fig)
        
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) 
        
        for suffix in ['.png']:
            plt.savefig('figures/' + 'test_TD', dpi=300, bbox_inches=extent)
    
    # =============================================================================
    #   Optimize the models via dynamic time warping
    #   Then plot:
    #       - The changes in correlation cooefficient 
    #         (more generically, whatever combination is being optimized)
    #       - The changes on a Taylor Diagram
    # =============================================================================
    #   Binarize the data and models
    smoothing_widths = {'Tao':      4,
                        'HUXt':     2,
                        'ENLIL':    2,
                        'Juno':     6,
                        'Ulysses':  12}   #  hours
    traj0.binarize('u_mag', smooth = smoothing_widths, sigma=3)
    
    traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
    traj0.plot_SingleTimeseries('jumps', starttime, stoptime)
    
    #   Calculate a whole host of statistics
    dtw_stats = traj0.find_WarpStatistics('jumps', 'u_mag', shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)
    #   Write an equation describing the optimization equation
    def optimization_eqn(df):
        f = df['r'] * 2/df['width_68']
        return (f - np.min(f))/(np.max(f)-np.min(f))
    #   Plug in the optimization equation
    traj0.optimize_Warp(optimization_eqn)
    
    #   This should probably be in MMESH/trajectory class
    for model_name in traj0.model_names:
        constant_offset = traj0.best_shifts[model_name]['shift']
        dynamic_offsets = traj0.model_dtw_times[model_name][str(int(constant_offset))]
        traj0._primary_df[model_name, 'empirical_dtime'] = constant_offset + dynamic_offsets

    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        traj0.plot_DynamicTimeWarping_Optimization()
        traj0.plot_DynamicTimeWarping_TD()
    
    
    # =============================================================================
    #    Compare output list of temporal shifts (and, maybe, running standard deviation or similar)
    #    to physical parameters: 
    #       -  T-S-E angle
    #       -  Solar cycle phase
    #       -  Running Mean SW Speed
    #   This should **ALL** ultimately go into the "Ensemble" class 
    # =============================================================================    
    formula = "empirical_dtime ~ solar_radio_flux + target_sun_earth_lon"  #  This can be input
    
    #   traj0.addContext()
    srf = mmesh.read_SolarRadioFlux(traj0._primary_df.index[0], traj0._primary_df.index[-1])
    traj0._primary_df[('context', 'solar_radio_flux')] = srf['adjusted_flux']   
    traj0._primary_df[('context', 'target_sun_earth_lon')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon']
    traj0._primary_df[('context', 'target_sun_earth_lat')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon']
    
    #prediction_df = 
    
    
    mmesh0 = mmesh.MMESH(trajectories=[traj0])
    
    test = mmesh0.linear_regression(formula)
    
    return test

    # formula = "total_dtimes ~ f10p7_flux + TSE_lat + TSE_lon" 
    

    
    #   Encapsulated Plotting
    def plot_LinearRegressions():
        fig = plt.figure(figsize=[6,8])
        gs = fig.add_gridspec(nrows=5, ncols=3, width_ratios=[1,1,1],
                              left=0.1, bottom=0.1, right=0.95, top=0.95,
                              wspace=0.05, hspace=0.05)
        axs = {}
        for i, model_name in enumerate(traj0.model_names):
            if i == 0:
                axs[model_name] = [fig.add_subplot(gs[y,i]) for y in range(5)]
            else:
                axs[model_name] = [fig.add_subplot(gs[y,i], sharey=axs[traj0.model_names[0]][y]) for y in range(5)]

        
        for i, model_name in enumerate(traj0.model_names):
            print('For model: {} ------------------------------'.format(model_name))
            label = '({})'.format(string.ascii_lowercase[i])
            
            constant_offset = best_shifts[model_name]['shift']
            delta_times = traj0.model_dtw_times[model_name][str(int(constant_offset))]
            total_dtimes_inh = (constant_offset + delta_times)
            
            ax = axs[model_name]
            
            # =============================================================================
            #   Total delta times vs. Model SW Flow Speed
            # =============================================================================
            u_mag = traj0.models[model_name]['u_mag'].to_numpy('float64')
            a, b = TD.make_NaNFree(total_dtimes_inh, u_mag)
            
            ax[0].scatter(a, b)
            
            r_stat = scstats.pearsonr(a, b)
            print(r_stat)
            
            ax[0].annotate('({})'.format(string.ascii_lowercase[3*0]), (0,1), xytext=(0.5, -0.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='left', va='top')
            ax[0].annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='right', va='top')
            
            # =============================================================================
            #   Total delta times vs. Model SW Flow Speed - Data
            #   **WARNING** This can't be used for prediction, as we wouldn't have data for
            #   comparison
            # =============================================================================
            u_mag = traj0.models[model_name]['u_mag'] - traj0.data['u_mag'].to_numpy('float64')
            a, b = TD.make_NaNFree(total_dtimes_inh, u_mag)
            ax[1].scatter(a, b)
            r_stat = scstats.pearsonr(a, b)
            
            ax[1].annotate('Not suitable for forecasting', (1,1), (-0.5,-0.5), 
                           xycoords='axes fraction', textcoords='offset fontsize', 
                           annotation_clip=False, ha='right', va='top')
            ax[1].annotate('({})'.format(string.ascii_lowercase[3*1+i]), (0,1), xytext=(0.5, -0.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='left', va='top')
            ax[1].annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='right', va='top')
                
            # =============================================================================
            #   F10.7 Radio flux from the sun
            # =============================================================================
            column_headers = ('date', 'observed_flux', 'adjusted_flux',)
            solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                           header = 0, names = column_headers)
            solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                        for d in solar_radio_flux['date']]
            solar_radio_flux = solar_radio_flux.set_index('date')
            solar_radio_flux = solar_radio_flux.resample("60Min", origin='start_day').mean().interpolate(method='linear')
            solar_radio_flux = solar_radio_flux.reindex(index=total_dtimes_inh.index)

            a, b = TD.make_NaNFree(total_dtimes_inh, solar_radio_flux['adjusted_flux'].to_numpy('float64'))
            ax[2].scatter(a, b)
            b_F107 = b
            
            
            r_stat = scstats.pearsonr(a, b)
            print(r_stat)
            
            ax[2].annotate('({})'.format(string.ascii_lowercase[3*2+i]), (0,1), xytext=(0.5, -0.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='left', va='top')
            ax[2].annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='right', va='top')
            # =============================================================================
            # TSE Angle
            # =============================================================================
            TSE_lon = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lon']
            
            a, b = TD.make_NaNFree(total_dtimes_inh, TSE_lon.to_numpy('float64'))
            r_stat = scstats.pearsonr(a, b)
            print(r_stat)
            
            ax[3].scatter(a,b)
            
            ax[3].annotate('({})'.format(string.ascii_lowercase[3*3+i]), (0,1), xytext=(0.5, -0.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='left', va='top')
            ax[3].annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='right', va='top')
            
            TSE_lat = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lat']
            
            a, b = TD.make_NaNFree(total_dtimes_inh, TSE_lat.to_numpy('float64'))
            r_stat = scstats.pearsonr(a, b)
            print(r_stat)
            
            ax[4].scatter(a,b)
            
            x, b, c, d = TD.make_NaNFree(total_dtimes_inh, solar_radio_flux['adjusted_flux'].to_numpy('float64'), TSE_lon.to_numpy('float64'), TSE_lat.to_numpy('float64'))
            x_prediction = mlr_df['c0'][model_name] + mlr_df['c1'][model_name]*b + mlr_df['c2'][model_name]*c + mlr_df['c3'][model_name]*d
            x_prediction_err = mlr_df['c0_sigma'][model_name] + mlr_df['c1_sigma'][model_name]*b + mlr_df['c2_sigma'][model_name]*c + mlr_df['c3_sigma'][model_name]*d
            
            ax[2].scatter(x_prediction, b, color='C3', s=2)
            ax[3].scatter(x_prediction, c, color='C3', s=2)
            ax[4].scatter(x_prediction, d, color='C3', s=2)
            ax[4].errorbar(x_prediction, d, xerr=x_prediction_err)
            
            ax[4].annotate('({})'.format(string.ascii_lowercase[3*4+i]), (0,1), xytext=(0.5, -0.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='left', va='top')
            ax[4].annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                           xycoords='axes fraction', textcoords='offset fontsize',
                           ha='right', va='top')
            
            print('------------------------------')
            
            for i in range(5): ax[i].set(xlim=[-120,120], xticks=np.arange(-120, 120+24, 24))
            
            return x_prediction, x_prediction_err
     
    x = plot_LinearRegressions()
    
    return mlr_df, x

    fig1, axs1 = plt.subplots(nrows=3, sharex=True, sharey=True)
    fig2, axs2 = plt.subplots(nrows=3, sharex=True, sharey=True)
    fig3, axs3 = plt.subplots(nrows=3, sharex=True, sharey=True)
    
    fig5, axs5 = plt.subplots(nrows=3, sharex=True, sharey=True)
    for i, (model_name, ax1, ax2, ax3, ax5) in enumerate(zip(traj0.model_names, axs1, axs2, axs3, axs5)):
        
        print('For model: {} ----------'.format(model_name))
        label = '({}) {}'.format(string.ascii_lowercase[i], model_name)
        
        offset = best_shifts[model_name]['shift']
        dtimes = traj0.model_dtw_times[model_name][str(int(offset))]
        total_dtimes_inh = (offset + dtimes)
        
        u_mag = traj0.models[model_name]['u_mag'].to_numpy('float64')
        #u_mag = traj0.models[model_name]['u_mag'] - traj0.data['u_mag'].to_numpy('float64')
        a, b = TD.make_NaNFree(total_dtimes_inh, u_mag)
        
        ax1.scatter(a, b)
        ax1.set(xlim=[-120,120], xticks=np.arange(-120, 120+24, 24))
        r_stat = scstats.pearsonr(a, b)
        print(r_stat)
        ax1.annotate(label, (0,1), xytext=(0.5, -0.5),
                    xycoords='axes fraction', textcoords='offset fontsize',
                    ha='left', va='top')
        ax1.annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                     xycoords='axes fraction', textcoords='offset fontsize',
                     ha='right', va='top')
        
        u_mag = traj0.models[model_name]['u_mag'] - traj0.data['u_mag'].to_numpy('float64')
        a, b = TD.make_NaNFree(total_dtimes_inh, u_mag)
        ax2.scatter(a, b)
        r_stat = scstats.pearsonr(a, b)
        ax2.set(xlim=[-120,120], xticks=np.arange(-120, 120+24, 24))
        ax2.annotate('Not suitable for forecasting', (1,1), (-0.5,-0.5), 
                     xycoords='axes fraction', textcoords='offset fontsize', 
                     annotation_clip=False, ha='right', va='top')
        ax2.annotate(label, (0,1), xytext=(0.5, -0.5),
                    xycoords='axes fraction', textcoords='offset fontsize',
                    ha='left', va='top')
        ax2.annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                     xycoords='axes fraction', textcoords='offset fontsize',
                     ha='right', va='top')
        
        # =============================================================================
        # F10.7 Radio flux from the sun
        # =============================================================================
        column_headers = ('date', 'observed_flux', 'adjusted_flux',)
        solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                       header = 0, names = column_headers)
        solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                    for d in solar_radio_flux['date']]
        solar_radio_flux = solar_radio_flux.set_index('date')
        solar_radio_flux = solar_radio_flux.resample("60Min", origin='start_day').mean()
        solar_radio_flux = solar_radio_flux.reindex(index=total_dtimes_inh.index)
        
        
        a, b = TD.make_NaNFree(total_dtimes_inh, solar_radio_flux['adjusted_flux'].to_numpy('float64'))
        ax3.scatter(a, b)
        ax3.set(xlim=[-120,120], xticks=np.arange(-120, 120+24, 24))
        r_stat = scstats.pearsonr(a, b)
        
        def linear_func(x, a, b):
            return a*x + b
        p_fit, p_cov = optimize.curve_fit(linear_func, a, b)
        p_err = np.sqrt(np.diag(p_cov))
        
        #return p_fit, p_err
        print(r_stat)
        ax3.annotate(label, (0,1), xytext=(0.5, -0.5),
                    xycoords='axes fraction', textcoords='offset fontsize',
                    ha='left', va='top')
        ax3.annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                     xycoords='axes fraction', textcoords='offset fontsize',
                     ha='right', va='top')
        # =============================================================================
        # TSE Angle
        # =============================================================================
        TSE_lon = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lon']
        
        a, b = TD.make_NaNFree(total_dtimes_inh, TSE_lon.to_numpy('float64'))
        ax5.scatter(a,b)
        ax5.set(xlim=[-120,120], xticks=np.arange(-120, 120+24, 24))
        r_stat = scstats.pearsonr(a, b)
        print(r_stat)
        ax5.annotate(label, (0,1), xytext=(0.5, -0.5),
                    xycoords='axes fraction', textcoords='offset fontsize',
                    ha='left', va='top')
        ax5.annotate('r={:.3f}'.format(r_stat[0]), (1,1), (-1.5, -1.5),
                     xycoords='axes fraction', textcoords='offset fontsize',
                     ha='right', va='top')
        
        print('------------------------------')
    
    #  Figure 1: Shift times vs. Speed
    fig1.supxlabel('Total Temporal Shifts [h]')
    #fig1.supylabel(r'$\Delta u_{mag}$ (model - data) [km s$^{-1}$]')
    fig1.supylabel(r'Model $u_{mag}$ [km s$^{-1}$]')
    axs1[-1].annotate(r'Model Leads Data $\rightarrow$', (1,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='right', va='top')
    axs1[-1].annotate(r'$\leftarrow$ Model Lags Data', (0,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='left', va='top')
    
    fig2.supxlabel('Total Temporal Shifts [h]')
    fig2.supylabel(r'$\Delta u_{mag}$ (model - data) [km s$^{-1}$]')
    axs2[-1].annotate(r'Model Leads Data $\rightarrow$', (1,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='right', va='top')
    axs2[-1].annotate(r'$\leftarrow$ Model Lags Data', (0,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='left', va='top')
    
    #  Figure 3: Shift times vs. Solar Cycle
    fig3.supxlabel('Total Temporal Shifts [h]')
    fig3.supylabel('Adjusted Solar F10.7 Flux [SFU]')
    axs3[-1].annotate(r'Model Leads Data $\rightarrow$', (1,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='right', va='top')
    axs3[-1].annotate(r'$\leftarrow$ Model Lags Data', (0,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='left', va='top')
    
    fig5.supxlabel('Total Temporal Shifts [h]')
    fig5.supylabel('TSE Angle [degrees]')
    axs5[-1].annotate(r'Model Leads Data $\rightarrow$', (1,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='right', va='top')
    axs5[-1].annotate(r'$\leftarrow$ Model Lags Data', (0,0), (0,-2), 
                      xycoords='axes fraction', textcoords='offset fontsize', 
                      annotation_clip=False, ha='left', va='top')
    
    plt.show()
    
    return traj0
    
    # traj0.ensemble()
    # traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
    # traj0.plot_SingleTimeseries('jumps', starttime, stoptime)
    
    # stats = traj0.baseline()
    
    # ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
    # plot_kw = {'yscale': 'linear', 'ylim': (0, 60),
    #             'yticks': np.arange(350,550+50,100)}
    # ref_std = stats['Tao']['u_mag'][0]
    # with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
    #     fig, ax = TD.plot_TaylorDiagram_fromstats(ref_std)
    
    # x_max = ref_std
    # for model_name in traj0.model_names:
        
    #     r = stats[model_name]['u_mag'][2]
    #     std = stats[model_name]['u_mag'][1]
    #     if std > x_max: x_max = std
    #     print(model_name, r, std)
    #     baseline_plot = ax.scatter(np.arccos(r), std, 
    #                                 marker=model_symbols[model_name], s=36, c=model_colors[model_name],
    #                                 zorder=10, label=model_name)
              
    # ax.set_ylim([0, 1.25*x_max])
        
    # ax.text(0.5, 0.9, ylabel,
    #         horizontalalignment='center', verticalalignment='top',
    #         transform = ax.transAxes)
    # ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.5)
    # ax.set_axisbelow(True)
        
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # extent = extent.expanded(1.0, 1.0)
    # extent.intervalx = extent.intervalx + 0.4

    # savefile_stem = 'TD_Combined_Baseline_u_mag_Ulysses'
    # for suffix in ['.png']:
    #     plt.savefig('figures/' + savefile_stem, dpi=300, bbox_inches=extent)
    # plt.show()            
    
    # return stats

    #baseline_test = traj0.baseline()
    
    #temp = test.ensemble()
    
    #  Shifts should be a dict, only apply to models

        
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1.0, 1.0)
    extent.intervalx = extent.intervalx + 0.4

    savefile_stem = 'TD_Combined_Baseline_Warp_u_mag_Ulysses'
    for suffix in ['.png']:
        plt.savefig('figures/' + savefile_stem, dpi=300, bbox_inches=extent)
    plt.show()            
    
    nmodels = len(traj0.model_names)
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(figsize=(12,6), ncols=2*nmodels, width_ratios=[3,1]*nmodels, sharey=True)
        
    for ax_no, model_name in enumerate(traj0.model_names):
        cols = traj0.model_dtw_times[model_name].columns
        
        for counter, col in enumerate(reversed(cols)):
            histo = np.histogram(traj0.model_dtw_times[model_name][col], range=[-96,96], bins=int(192*0.5))
            total = float(len(traj0.model_dtw_times[model_name][col]))  #  normalization 
            
            #  Offset y based on the shift value of the column
            #  Normalize histogram area to 1 (i.e. density), then enlarge so its visible
            y_offset = float(col)
            norm_factor = (1./float(len(traj0.model_dtw_times[model_name][col]))) * 200.
            
            axs[ax_no*2].plot(histo[1][:-1]+3, histo[0]*norm_factor + y_offset)
            axs[ax_no*2].fill_between(histo[1][:-1]+3, histo[0]*norm_factor + y_offset, y_offset, alpha=0.6)
            
        axs[ax_no*2+1].plot(traj0.model_dtw_stats[model_name]['r'], np.array(cols, dtype='float64'),
                            marker='o')
        
        axs[ax_no*2].set_ylim([-96-2, 96+6+2])
        axs[ax_no*2].set_yticks(np.linspace(-96, 96, 9))
        axs[ax_no*2+1].set_ylim([-96-2, 96+6+2])
        axs[ax_no*2+1].set_yticklabels([])
        
    axs[0].set_yticklabels(np.linspace(-96, 96, 9))
    plt.show()
    
    return traj0
    
    plt.show()
#     #==========================
#     #  Load spacecraft data
#     starttime = dt.datetime(1991, 12, 8)
#     stoptime = dt.datetime(1992, 2, 2)
#     spacecraft_name = 'Ulysses'
    
#     spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
#     spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
#     # spacecraft.find_state(reference_frame, observer)
#     # spacecraft.find_subset(coord1_range=np.array(r_range)*spacecraft.au_to_km, 
#     #                        coord3_range=np.array(lat_range)*np.pi/180., 
#     #                        transform='reclat')
    
#     test2 = mmesh.Trajectory()
#     test2.addData(spacecraft_name, spacecraft.data)
    
#     #  Read models
#     for model in model_names:
#         temp = read_SWModel.choose(model, spacecraft_name, 
#                                        starttime, stoptime, resolution='60Min')
#         test2.addModel(model, temp)
    
#     temp = test2.warp('jumps', 'u_mag')
    
#     #==========================
#     #  Load spacecraft data
#     starttime = dt.datetime(1997, 8, 14)
#     stoptime = dt.datetime(1998, 4, 16)
#     spacecraft_name = 'Ulysses'
    
#     spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
#     spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
#     # spacecraft.find_state(reference_frame, observer)
#     # spacecraft.find_subset(coord1_range=np.array(r_range)*spacecraft.au_to_km, 
#     #                        coord3_range=np.array(lat_range)*np.pi/180., 
#     #                        transform='reclat')
    
#     test3 = mmesh.Trajectory()
#     test3.addData(spacecraft_name, spacecraft.data)
    
#     #  Read models
#     for model in model_names:
#         temp = read_SWModel.choose(model, spacecraft_name, 
#                                        starttime, stoptime, resolution='60Min')
#         test3.addModel(model, temp)
    
#     temp = test3.warp('jumps', 'u_mag')
    
# # =============================================================================
# #     
# # =============================================================================
   
#     joined_model = 'ENLIL'
#     joined = pd.concat([test.model_dfs[joined_model], test2.model_dfs[joined_model], test3.model_dfs[joined_model]])
    
#     fig, ax = plt.subplots(nrows=3)
#     ax[0].scatter(test.model_dfs[joined_model].index, test.model_dfs[joined_model]['time_lag']/3600., s=0.001)
#     ax[1].scatter(test2.model_dfs[joined_model].index, test2.model_dfs[joined_model]['time_lag']/3600., s=0.001)
#     ax[2].scatter(test3.model_dfs[joined_model].index, test3.model_dfs[joined_model]['time_lag']/3600., s=0.001)
#     plt.show()
    
#     plt.scatter(joined.index, joined['time_lag']/3600., s=0.0001)
    
#     return joined