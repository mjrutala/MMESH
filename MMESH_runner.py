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
spacecraft_markers = {'Pioneer 10': 'o',
                      'Pioneer 11': 'o',
                      'Voyager 1' : 'o',
                      'Voyager 2' : 'o',
                      'Ulysses'   : 'o',
                      'Juno'      : 'o'}

model_colors = {'Tao'    : '#C59FE5',
                'HUXt'   : '#A0A5E4',
                'SWMF-OH': '#98DBEB',
                'MSWIM2D': '#A9DBCA',
                'ENLIL'  : '#DCED96'}
model_markers = {'Tao'    : 'v',
                'HUXt'   : '*',
                'SWMF-OH': '^',
                'MSWIM2D': 'd',
                'ENLIL'  : 'h'}

r_range = np.array((4.9, 5.5))  #  [4.0, 6.4]
lat_range = np.array((-6.1, 6.1))  #  [-10, 10]


# def MMESH_traj_run():
#     import string
    
#     import sys
#     sys.path.append('/Users/mrutala/projects/SolarWindEM/')
#     import MMESH as mmesh
#     import numpy as np
#     import scipy.stats as scstats
#     import scipy.optimize as optimize
#     import matplotlib.pyplot as plt
#     import plot_TaylorDiagram as TD
    
#     from sklearn.linear_model import LinearRegression
    
#     from numpy.polynomial import Polynomial
    
#     spacecraft_name = 'Juno' # 'Ulysses'
#     model_names = ['Tao', 'HUXt', 'ENLIL']
    
#     #   !!!! This whole pipline is currently (2023 11 03) sensitive to these
#     #   input dates, but shouldn't be-- Optimization should not change based on input dates
#     #   as long as the input dates span the whole of the spacecraft data being used
#     #   since we should always be dropping NaN rows...
#     starttime = dt.datetime(2016, 5, 16) # dt.datetime(1997,8,14) # dt.datetime(1991,12,8) # 
#     stoptime = dt.datetime(2016, 6, 26) # dt.datetime(1998,1,1) # dt.datetime(1992,2,2) # 
    
#     reference_frame = 'SUN_INERTIAL'
#     observer = 'SUN'
    
#     #!!!!
#     #  NEED ROBUST in-SW-subset TOOL! ADD HERE!
    
#     #  Change start and stop times to reflect spacecraft limits-- with optional padding
#     padding = dt.timedelta(days=8)
#     starttime = starttime - padding
#     stoptime = stoptime + padding
    
#     #  Load spacecraft data
#     spacecraft = spacecraftdata.SpacecraftData(spacecraft_name)
#     spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
#     pos_TSE = spacecraft.find_StateToEarth()
    
#     #  Initialize a trajectory class and add the spacecraft data
#     traj0 = mmesh.Trajectory()
#     traj0.addData(spacecraft_name, spacecraft.data)
    
#     # traj0.addTrajectory(x)
    
#     #  Read models and add to trajectory
#     for model_name in model_names:
#         model = read_SWModel.choose(model_name, spacecraft_name, 
#                                     starttime, stoptime, resolution='60Min')
#         traj0.addModel(model_name, model)
    
#     # =============================================================================
#     #   Plot a Taylor Diagram of the unchanged models
#     # =============================================================================
#     with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
#         fig = plt.figure(figsize=(6,4.5))
#         fig, ax = traj0.plot_TaylorDiagram(tag_name='u_mag', fig=fig)
#         ax.legend(ncols=3, bbox_to_anchor=[0.0,0.05,1.0,0.15], 
#                   loc='lower left', mode='expand', markerscale=1.0)
#         plt.show()
    
#     # =============================================================================
#     #   Optimize the models via constant temporal shifting
#     #   Then plot:
#     #       - The changes in correlation coefficient 
#     #         (more generically, whatever combination is being optimized)
#     #       - The changes on a Taylor Diagram
#     # =============================================================================
#     shifts = np.arange(-96, 96+6, 6)    #  in hours
#     shift_stats = traj0.optimize_shifts('u_mag', shifts=shifts)
#     print(traj0.model_shift_mode)
    
#     with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
#         traj0.plot_ConstantTimeShifting_Optimization()
        
#         fig = plt.figure(figsize=[6,4.5])
#         traj0.plot_ConstantTimeShifting_TD(fig=fig)
        
#         extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) 
        
#         for suffix in ['.png']:
#             plt.savefig('figures/' + 'test_TD', dpi=300, bbox_inches=extent)
    
#     # =============================================================================
#     #   Optimize the models via dynamic time warping
#     #   Then plot:
#     #       - The changes in correlation cooefficient 
#     #         (more generically, whatever combination is being optimized)
#     #       - The changes on a Taylor Diagram
#     # =============================================================================
#     #   Binarize the data and models
#     smoothing_widths = {'Tao':      4,
#                         'HUXt':     2,
#                         'ENLIL':    2,
#                         'Juno':     6,
#                         'Ulysses':  12}   #  hours
#     traj0.binarize('u_mag', smooth = smoothing_widths, sigma=3)
    
#     traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
#     traj0.plot_SingleTimeseries('jumps', starttime, stoptime)
    
#     #   Calculate a whole host of statistics
#     dtw_stats = traj0.find_WarpStatistics('jumps', 'u_mag', shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)
#     #   Write an equation describing the optimization equation
#     def optimization_eqn(df):
#         f = df['r'] * 2/df['width_68']
#         return (f - np.min(f))/(np.max(f)-np.min(f))
#     #   Plug in the optimization equation
#     traj0.optimize_Warp(optimization_eqn)
#     print(traj0.model_shift_mode)
#     #   This should probably be in MMESH/trajectory class
#     for model_name in traj0.model_names:
#         constant_offset = traj0.best_shifts[model_name]['shift']
#         dynamic_offsets = traj0.model_shifts[model_name][str(int(constant_offset))]
#         traj0._primary_df[model_name, 'empirical_dtime'] = constant_offset + dynamic_offsets

#     with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
#         traj0.plot_DynamicTimeWarping_Optimization()
#         traj0.plot_DynamicTimeWarping_TD()
    
    
#     # =============================================================================
#     #    Compare output list of temporal shifts (and, maybe, running standard deviation or similar)
#     #    to physical parameters: 
#     #       -  T-S-E angle
#     #       -  Solar cycle phase
#     #       -  Running Mean SW Speed
#     #   This should **ALL** ultimately go into the "Ensemble" class 
#     # =============================================================================    
#     formula = "empirical_dtime ~ solar_radio_flux + target_sun_earth_lon"  #  This can be input
    
#     #   traj0.addContext()
#     srf = mmesh.read_SolarRadioFlux(traj0._primary_df.index[0], traj0._primary_df.index[-1])
#     traj0._primary_df[('context', 'solar_radio_flux')] = srf['adjusted_flux']   
#     traj0._primary_df[('context', 'target_sun_earth_lon')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon']
#     traj0._primary_df[('context', 'target_sun_earth_lat')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon']
    
#     prediction_df = pd.DataFrame(index = pd.DatetimeIndex(np.arange(traj0._primary_df.index[0], traj0._primary_df.index[-1] + dt.timedelta(days=1000), dt.timedelta(hours=1))))
#     prediction_df['solar_radio_flux'] = mmesh.read_SolarRadioFlux(prediction_df.index[0], prediction_df.index[-1])['adjusted_flux']
    
#     sc_pred = spacecraftdata.SpacecraftData(spacecraft_name)
#     sc_pred.make_timeseries(prediction_df.index[0], 
#                             prediction_df.index[-1] + dt.timedelta(hours=1), 
#                             timedelta=dt.timedelta(hours=1))
#     pos_TSE_pred = sc_pred.find_StateToEarth()
#     prediction_df['target_sun_earth_lon'] = pos_TSE_pred.reindex(index=prediction_df.index)['del_lon']
    
#     mmesh0 = mmesh.MMESH(trajectories=[traj0])
    
#     test = mmesh0.linear_regression(formula)
    
#     import statsmodels.api as sm
#     import statsmodels.formula.api as smf
    
#     # for model_name in traj0.model_names:
#     forecast = test['Tao'].get_prediction(prediction_df)
#     alpha_level = 0.32 
#     result = forecast.summary_frame(alpha_level)
    
#     return prediction_df, result

#     # formula = "total_dtimes ~ f10p7_flux + TSE_lat + TSE_lon" 
    

inputs = {}
# inputs['Voyager1_01'] = {'spacecraft_name':'Voyager 1',
#                          'span':(dt.datetime(1979, 1, 3), dt.datetime(1979, 5, 5))}
inputs['Ulysses_01']  = {'spacecraft_name':'Ulysses',
                          'span':(dt.datetime(1991,12, 8), dt.datetime(1992, 2, 2))}
inputs['Ulysses_02']  = {'spacecraft_name':'Ulysses',
                          'span':(dt.datetime(1997, 8,14), dt.datetime(1998, 4,16))}
inputs['Ulysses_03']  = {'spacecraft_name':'Ulysses',
                          'span':(dt.datetime(2003,10,24), dt.datetime(2004, 6,22))}
inputs['Juno_01']     = {'spacecraft_name':'Juno',
                          'span':(dt.datetime(2016, 5,16), dt.datetime(2016, 6,26))}

model_names = ['ENLIL', 'HUXt', 'Tao']

def MMESH_run(data_dict, model_names):
    import string
    
    import sys
    sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    import MMESH as mmesh
    import MMESH_context as mmesh_c
    import numpy as np
    import scipy.stats as scstats
    import scipy.optimize as optimize
    import matplotlib.pyplot as plt
    import plot_TaylorDiagram as TD
    import copy
    
    import string
    
    from sklearn.linear_model import LinearRegression
    
    from numpy.polynomial import Polynomial
    
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
    
    #   Encapsualted plotting calls for publishing
    
    def plot_BothShiftMethodsTD():
        #   This TD initialization is not totally accurate
        #   Since we take the standard deviation after dropping NaNs in the 
        #   reference only-- would be better to drop all NaN rows among ref and
        #   tests first, then take all standard deviations
        fig = plt.figure(figsize=(6,4.5), constrained_layout=True)
        fig, ax = TD.init_TaylorDiagram(np.nanstd(traj0.data['u_mag']), fig=fig)
        for model_name in traj0.model_names:
            TD.plot_TaylorDiagram(traj0.models[model_name]['u_mag'].loc[traj0.data_index], traj0.data['u_mag'], ax=ax,
                                  s=32, c='black', marker=traj0.plotparams.model_markers[model_name],
                                  label=r'{}'.format(model_name))
            ax.scatter(*constant_shift_dict[model_name],
                       s=32, facecolors=traj0.plotparams.model_colors[model_name], marker=traj0.plotparams.model_markers[model_name],
                       edgecolors='black', linewidths=0.5,
                       label=r'{} + Cons.'.format(model_name))
            ax.scatter(np.arccos(traj0.best_shifts[model_name]['r']), traj0.best_shifts[model_name]['stddev'],
                       s=32, c=traj0.plotparams.model_colors[model_name], marker=traj0.plotparams.model_markers[model_name],
                       label=r'{} + DTW'.format(model_name))
            
        ax.legend(ncols=3, bbox_to_anchor=[0.0,-0.02,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        #print(fig, ax)
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) 
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0.05)
        fig.savefig('figures/Paper/' + 'TD_{}_BothShiftMethods.png'.format('_'.join(traj0.model_names)), 
                    dpi=300)#, bbox_inches=extent)
        plt.show()
    
    def plot_BestShiftWarpedTimeDistribution():
        
        nmodels = len(traj0.model_names)
        fig, axs = plt.subplots(nrows=nmodels, figsize=(3,4.5), sharex=True, sharey=True, squeeze=False)
        plt.subplots_adjust(left=0.2, top=0.975 ,hspace=0.075)
        
        for i, (model_name, ax) in enumerate(zip(traj0.model_names, axs)):
            
            ax[0].hist(traj0.models[model_name].loc[traj0.data_index, 'empirical_time_delta'].to_numpy(), 
                    range=(-192, 192), bins=64,
                    density=True, label=model_name, color=model_colors[model_name])
            ax[0].annotate('({}) {}'.format(string.ascii_lowercase[i], model_name), 
                        (0,1), (1,-1), ha='left', va='center', 
                        xycoords='axes fraction', textcoords='offset fontsize')

        ax[0].set_yticks(ax[0].get_yticks(), (100.*np.array(ax[0].get_yticks())).astype('int64'))
        
        fig.supylabel('Percentage')
        fig.supxlabel('Total Temporal Shifts [hours]')
        
        plt.savefig('figures/Paper/' + 'TemporalShifts_{}_Total.png'.format('_'.join(traj0.model_names)), 
                    dpi=300)
        plt.show()
        #
    def plot_TemporalShifts_All_Total():
        fig, axs = plt.subplots(figsize=(3,4.5), nrows=len(m_traj.model_names), sharex=True, sharey=True)
        plt.subplots_adjust(left=0.2, top=0.8, right=0.95, hspace=0.075)
        for i, model_name in enumerate(sorted(m_traj.model_names)):
            l = []
            c = []
            n = []
            for traj_name, traj in m_traj.trajectories.items():
                l.append(traj.models[model_name].loc[traj.data_index, 'empirical_time_delta'])  #  Only when overlapping w/ data
                c.append(traj.plotparams.data_color)
                n.append(traj.trajectory_name)
            
            #  Hacky
            c = ['#5d9dd5', '#447abc', '#4034f4', '#75036b']
            axs[i].hist(l, range=(-120, 120), bins=20,
                        density=True,
                        histtype='barstacked', stacked=True, label=n, color=c)
            axs[i].annotate('({}) {}'.format(string.ascii_lowercase[i], model_name), 
                        (0,1), (1,-1), ha='left', va='center', 
                        xycoords='axes fraction', textcoords='offset fontsize')
        
        
        fig.supylabel('Percentage')
        fig.supxlabel('Total Temporal Shifts [hours]')
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=2,
                   bbox_to_anchor=[0.2, 0.8, 0.75, 0.2], loc='lower left',
                   mode="expand", borderaxespad=0.)
        
        plt.savefig('figures/Paper/' + 'TemporalShifts_All_Total.png', 
                    dpi=300)
        plt.show()
    #
    def plot_Correlations():
        fig, axs = plt.subplots(figsize=(6,4.5), nrows=4, ncols=len(m_traj.model_names), sharex='col', sharey='row', squeeze=False)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.92, right=0.90, 
                            wspace=0.45, hspace=0.025)
        for i, model_name in enumerate(sorted(m_traj.model_names)):
            axs[0,i].set_xlabel(model_name)
            axs[0,i].xaxis.set_label_position('top')
            
            invariant_param = 'empirical_time_delta'
            variant_params = ['solar_radio_flux', 'target_sun_earth_lon', 'target_sun_earth_lat', 'u_mag']
            model_stats = {key: [] for key in variant_params + [invariant_param]}
            for k, (traj_name, traj) in enumerate(m_traj.trajectories.items()):
                
                #   Record individual trajectory info to calculate the overall stats
                model_stats[invariant_param].extend(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index])
                model_stats['solar_radio_flux'].extend(traj._primary_df.loc[traj.data_index, ('context', 'solar_radio_flux')])
                model_stats['target_sun_earth_lon'].extend(traj._primary_df.loc[traj.data_index, ('context', 'target_sun_earth_lon')])
                model_stats['target_sun_earth_lat'].extend(traj._primary_df.loc[traj.data_index, ('context', 'target_sun_earth_lat')])
                model_stats['u_mag'].extend(traj.models[model_name]['u_mag'].loc[traj.data_index])
                
                axs[0,i].scatter(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                 traj._primary_df.loc[traj.data_index, ('context', 'solar_radio_flux')],
                                 marker='o', s=2, label=traj_name)
                (r,sig), rmsd = TD.find_TaylorStatistics(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                                         traj._primary_df.loc[traj.data_index, ('context', 'solar_radio_flux')])
                axs[0,i].annotate('r={:6.2f}'.format(r), (1,1), (0.5,-k-1), 
                                  xycoords='axes fraction', textcoords='offset fontsize',
                                  ha='left', va='top', color='C'+str(k))
                
                axs[1,i].scatter(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                 traj._primary_df.loc[traj.data_index, ('context', 'target_sun_earth_lon')],
                                 marker='o', s=2, label=traj_name)
                (r,sig), rmsd = TD.find_TaylorStatistics(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                                         traj._primary_df.loc[traj.data_index, ('context', 'target_sun_earth_lon')])
                axs[1,i].annotate('r={:6.2f}'.format(r), (1,1), (0.5,-k-1), 
                                  xycoords='axes fraction', textcoords='offset fontsize',
                                  ha='left', va='top', color='C'+str(k))
                
                axs[2,i].scatter(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                 traj._primary_df.loc[traj.data_index, ('context', 'target_sun_earth_lat')],
                                 marker='o', s=2, label=traj_name)
                (r,sig), rmsd = TD.find_TaylorStatistics(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                                         traj._primary_df.loc[traj.data_index, ('context', 'target_sun_earth_lat')])
                axs[2,i].annotate('r={:6.2f}'.format(r), (1,1), (0.5,-k-1), 
                                  xycoords='axes fraction', textcoords='offset fontsize',
                                  ha='left', va='top', color='C'+str(k))
                
                axs[3,i].scatter(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                 traj.models[model_name]['u_mag'].loc[traj.data_index],
                                 marker='o', s=2, label=traj_name)      
                (r,sig), rmsd = TD.find_TaylorStatistics(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                                         traj.models[model_name]['u_mag'].loc[traj.data_index])
                axs[3,i].annotate('r={:6.2f}'.format(r), (1,1), (0.5,-k-1), 
                                  xycoords='axes fraction', textcoords='offset fontsize',
                                  ha='left', va='top', color='C'+str(k))
            
            for ax_no, variant_param in enumerate(variant_params):
                (r,sig), rmsd = TD.find_TaylorStatistics(model_stats[invariant_param], 
                                                         model_stats[variant_param])
                axs[ax_no,i].annotate('r={:6.2f}'.format(r), (1,1), (0.5, 0), 
                                    xycoords='axes fraction', textcoords='offset fontsize',
                                    ha='left', va='top', color='black')
        
        fig.supxlabel('Total Temporal Shifts (Constant Offsets + Dynamic Warping) [hours]')
        axs[0,0].set_ylabel('Solar Radio Flux' '\n' '[SFU]')
        axs[1,0].set_ylabel(r'TSE Lon.' '\n' '[$^{o}$]')
        axs[2,0].set_ylabel(r'TSE Lat.' '\n' '[$^{o}$]')
        axs[3,0].set_ylabel(r'Modeled u$_{mag}$' '\n' '[km/s]')
        
                
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=4, markerscale=2,
                   bbox_to_anchor=[0.1, 0.96, 0.86, .04], loc='lower left',
                   mode="expand", borderaxespad=0.)
        plt.savefig('figures/Paper/' + 'Correlations_AllModels_AllEpochs.png', 
                    dpi=300)
        plt.show()
    
    def plot_TimeDelta_Grid_AllTrajectories():
        """
        Plot the empirical time deltas, and the fits to each

        Returns
        -------
        None.

        """
        import matplotlib.dates as mdates
        #   Loop over trajectories, getting length of data for plotting
        length_list = []
        for traj_name, traj in m_traj.trajectories.items():
            length_list.append(len(traj.data))
        
        fig, axs = plt.subplots(figsize=(9,6), nrows=len(m_traj.model_names), ncols=len(m_traj.trajectories), 
                                sharex='col', sharey='row', width_ratios=length_list, squeeze=False)
        plt.subplots_adjust(bottom=0.125, left=0.12, top=0.975, right=0.975, 
                            wspace=0.05, hspace=0.05)
        for i, (traj_name, traj) in enumerate(m_traj.trajectories.items()):
            for j, model_name in enumerate(traj.model_names):
                
                axs[j,i].plot(m_traj.nowcast_dict[model_name].index, 
                              m_traj.nowcast_dict[model_name]['mean'], color='C0')
                
                axs[j,i].fill_between(m_traj.nowcast_dict[model_name].index, 
                                      m_traj.nowcast_dict[model_name]['obs_ci_lower'], 
                                      m_traj.nowcast_dict[model_name]['obs_ci_upper'], 
                                      color='C0', alpha=0.5)
                
                axs[j,i].scatter(traj.models[model_name].loc[traj.data_index].index,
                                 traj.models[model_name]['empirical_time_delta'].loc[traj.data_index], 
                                 color='black', marker='o', s=4)
                
                if i == 0:
                    axs[j,i].set_ylabel(model_name)
                
                # axs[j,i].set_xlim(traj.models[model_name].index[0], 
                #                   traj.models[model_name].index[-1])
                
            tick_interval = 50
            subtick_interval = 10
            
            axs[0,i].set_xlim(traj.models[model_name].index[0], 
                              traj.models[model_name].index[-1])
            axs[0,i].xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
            axs[0,i].xaxis.set_minor_locator(mdates.DayLocator(interval=subtick_interval))
            
            #axs[0,i].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
            axs[0,i].xaxis.set_major_formatter(traj.timeseries_formatter(axs[0,i].get_xticks()))
                
        for i, ax in enumerate(axs.flatten()):
            ax.annotate('({})'.format(string.ascii_lowercase[i]),
                        (0,1), (1, -1), xycoords='axes fraction', textcoords='offset fontsize')
        fig.supxlabel('Date [DOY, DD MM, YYYY]')
        fig.supylabel('Timing Uncertainty [hours]')
        fig.savefig('figures/Paper/' + 'TimeDelta_Grid_AllTrajectories.png', 
                    dpi=300)
        plt.plot()
    
    trajectories = []
    for key, val in data_dict.items():
        starttime = val['span'][0]
        stoptime = val['span'][1]
        
        reference_frame = 'SUN_INERTIAL'
        observer = 'SUN'
        
        #  Load spacecraft data
        spacecraft = spacecraftdata.SpacecraftData(val['spacecraft_name'])
        spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
        pos_TSE = spacecraft.find_StateToEarth()
        
        #!!!!
        #  NEED ROBUST in-SW-subset TOOL! ADD HERE!
        
        #  Change start and stop times to reflect spacecraft limits-- with optional padding
        padding = dt.timedelta(days=8)
        starttime = spacecraft.data.index[0] - padding
        stoptime = spacecraft.data.index[-1] + padding
        
        #  Initialize a trajectory class and add the spacecraft data
        traj0 = mmesh.Trajectory(name=key)
        traj0.addData(val['spacecraft_name'], spacecraft.data)
        
        traj0.plotparams.data_color = spacecraft_colors[val['spacecraft_name']]  #  Optional
        traj0.plotparams.data_marker = spacecraft_markers[val['spacecraft_name']]  #  Optional
        
        #  Read models and add to trajectory
        for model_name in model_names:
            model = read_SWModel.choose(model_name, val['spacecraft_name'], 
                                        starttime, stoptime, resolution='60Min')
            traj0.addModel(model_name, model)
            
            traj0.plotparams.model_colors[model_name] = model_colors[model_name]  #  Optional
            traj0.plotparams.model_markers[model_name] = model_markers[model_name]  #  Optional
        
        # =============================================================================
        #   Plot a Taylor Diagram of the unchanged models
        # =============================================================================
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
        
        constant_shift_dict = {}
        for model_name in traj0.model_names:
            constant_shift_dict[model_name] = (np.arccos(traj0.best_shifts[model_name]['r']), 
                                               traj0.best_shifts[model_name]['stddev'])
        
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
        #   Previously used: # {'Tao':4, 'HUXt':2, 'ENLIL':2, 'Juno':6, 'Ulysses':12}   #  hours
        smoothing_widths = traj0.optimize_ForBinarization('u_mag', threshold=1.05)  
        traj0.binarize('u_mag', smooth = smoothing_widths, sigma=3) #  !!!! sigma value can be changed
        
        traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
        traj0.plot_SingleTimeseries('jumps', starttime, stoptime)
        
        #   Calculate a whole host of statistics
        dtw_stats = traj0.find_WarpStatistics('jumps', 'u_mag', shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)

        #   Write an equation describing the optimization equation
        #   This can be played with
        def optimization_eqn(df):
            f = 2.0*df['r'] + 1/(0.5*df['width_68'])  #   normalize width so 24 hours == 1
            return (f - np.min(f))/(np.max(f)-np.min(f))
        #   Plug in the optimization equation
        traj0.optimize_Warp(optimization_eqn)
        
        filepath = 'Paper/Figures/OptimizedDTWOffset_{}_{}.png'.format(traj0.trajectory_name, '-'.join(traj0.model_names))
        traj0.plot_OptimizedOffset('jumps', 'u_mag', filepath=filepath)
    
        traj0.plot_DynamicTimeWarping_Optimization()
        #traj0.plot_DynamicTimeWarping_TD()
        
        plot_BothShiftMethodsTD()
        plot_BestShiftWarpedTimeDistribution()
        
        #   See the effects of the Constant + DTW Shifts
        # traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
        # traj0.shift_Models(time_delta_column='empirical_time_delta')
        # traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
            
        # =============================================================================
        #             
        # =============================================================================
        #   traj0.addContext()
        srf = mmesh_c.read_SolarRadioFlux(traj0._primary_df.index[0], traj0._primary_df.index[-1])
        traj0._primary_df[('context', 'solar_radio_flux')] = srf['adjusted_flux']   
        traj0._primary_df[('context', 'target_sun_earth_lon')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon']
        traj0._primary_df[('context', 'target_sun_earth_lat')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lat']

        trajectories.append(traj0)
        
        # =============================================================================
        #   Uncomment this to perform a single-epoch ensemble
        # =============================================================================
        # traj0.shift_Models()
        # traj0.ensemble()
        
        # traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
       
        # fig = plt.figure(figsize=(6,4.5))
        # fig, ax = traj0.plot_TaylorDiagram(tag_name='u_mag', fig=fig)
        # ax.legend(ncols=3, bbox_to_anchor=[0.0,0.05,1.0,0.15], 
        #           loc='lower left', mode='expand', markerscale=1.0)
        # plt.show()
        
        #return traj0

    
    # =============================================================================
    #   Initialize a MultiTrajecory object
    #   !!!! N.B. This currently predicts at 'spacecraft', because that's easiest
    #   with everything I've set up. We want predictions at 'planet' in the end, though
    # =============================================================================
    m_traj = mmesh.MultiTrajectory(trajectories=trajectories)
    
    plot_Correlations() 
    
    #   Set up the prediction interval
    #original_spantime = stoptime - starttime
    prediction_starttime = dt.datetime(2016, 5, 22)  #  starttime - original_spantime
    prediction_stoptime = dt.datetime(2016, 11, 1)  #  stoptime + original_spantime
    prediction_target = 'Juno'
    
    #  Initialize a trajectory class for the predictions
    traj1 = mmesh.Trajectory()
    
    fullfilepath = mmesh_c.make_PlanetaryContext_CSV(prediction_target, 
                                                     prediction_starttime, 
                                                     prediction_stoptime, 
                                                     filepath='context_files/')
    traj1.context = pd.read_csv(fullfilepath, index_col='datetime', parse_dates=True)
    
    #  Read models and add to trajectory
    for model_name in model_names:
        model = read_SWModel.choose(model_name, prediction_target, 
                                    prediction_starttime, prediction_stoptime, resolution='60Min')
        traj1.addModel(model_name, model)
        
        traj1.plotparams.model_colors[model_name] = model_colors[model_name]  #  Optional
        traj1.plotparams.model_markers[model_name] = model_markers[model_name]  #  Optional
        
    traj1_backup =  copy.deepcopy(traj1)

    #   Add prediction Trajectory as simulcast
    m_traj.cast_intervals['simulcast'] = traj1
    
    #formula = "empirical_time_delta ~ solar_radio_flux + target_sun_earth_lon"  #  This can be input
    #formula = "empirical_time_delta ~ target_sun_earth_lat + u_mag"
    formula = "empirical_time_delta ~ target_sun_earth_lat + solar_radio_flux + u_mag"
    test = m_traj.linear_regression(formula)
    
    m_traj.cast_Models(with_error=True)
    
    plot_TimeDelta_Grid_AllTrajectories()
    
    m_traj.cast_intervals['simulcast'].ensemble()

    #   Example plot of shifted models with errors
    #   And a TD
    
    #  FOR TESTING ACCURACY
    #  Load spacecraft data
    spacecraft = spacecraftdata.SpacecraftData('Juno')
    spacecraft.read_processeddata(prediction_starttime, prediction_stoptime, resolution='60Min')
    
    fig, axs = plt.subplots(nrows=len(m_traj.cast_intervals['simulcast'].model_names), sharex=True)
    
    for i, model_name in enumerate(m_traj.cast_intervals['simulcast'].model_names):
        model = m_traj.cast_intervals['simulcast'].models[model_name]

        axs[i].scatter(spacecraft.data.index, spacecraft.data['u_mag'],
                        color='black', marker='o', s=2)
        
        if 'u_mag_pos_unc' in model.columns:
            axs[i].fill_between(model.index, 
                                model['u_mag'] + model['u_mag_pos_unc'], 
                                model['u_mag'] - model['u_mag_neg_unc'],
                                alpha = 0.5, color=m_traj.cast_intervals['simulcast'].plotparams.model_colors[model_name])
        
        axs[i].plot(model.index, model['u_mag'], color=m_traj.cast_intervals['simulcast'].plotparams.model_colors[model_name])
        
        #   Overplot the original model, skipping the ensemble
        if model_name in traj1_backup.model_names:
            
            axs[i].plot(traj1_backup.models[model_name].index, 
                        traj1_backup.models[model_name]['u_mag'], 
                        color='gray')
        
        axs[i].annotate(model_name, (0,1), (1,-1), xycoords='axes fraction', textcoords='offset fontsize')
    
        (r, std), rmsd = TD.find_TaylorStatistics(model['u_mag'].loc[spacecraft.data.index], spacecraft.data['u_mag'])

        #axs[i].annotate(str(r), (0,1), (1, -2), xycoords='axes fraction', textcoords='offset fontsize')
        
    filename = 'Ensemble_TimeSeries_{}_{}-{}.png'.format(prediction_target.lower().capitalize(),
                                                         prediction_starttime.strftime('%Y%m%dT%H%M%S'),
                                                         prediction_stoptime.strftime('%Y%m%dT%H%M%S'))
    fig.supxlabel('Date [YYYY-MM]')
    fig.supylabel(r'Solar Wind Flow Speed $|u_{SW}| [km/s]$')
    fig.savefig('figures/Paper/' + filename, 
                dpi=300)
    
    def plot_EnsembleTD_DTWMLR():
        fig2, ax2 = TD.init_TaylorDiagram(np.std(spacecraft.data['u_mag']))
        
        for i, model_name in enumerate(m_traj.cast_intervals['simulcast'].model_names):
            model = m_traj.cast_intervals['simulcast'].models[model_name]
            
            r, std, rmsd = [], [], []
            n_mc = int(1e3)
            rng = np.random.default_rng()
            
            fig_new, axs_new = plt.subplots(nrows=2, height_ratios=[2,1])
            
            # for i_mc in range(n_mc):
            #     y1 = rng.normal(model['u_mag'].loc[spacecraft.data.index], model['u_mag_sigma'].loc[spacecraft.data.index])
            #     y2 = spacecraft.data['u_mag']
            #     axs_new[0].plot(spacecraft.data.index, y1, alpha=0.01, color='gray')
            #     axs_new[0].scatter(spacecraft.data.index, y2, color='black', marker='o', s=4)
            #     (r_mc, std_mc), rmsd_mc = TD.find_TaylorStatistics(y1, y2)
            #     r.append(r_mc)
            #     std.append(std_mc)
            #     rmsd.append(rmsd_mc)
            
            axs_new[0].plot(spacecraft.data.index, model['u_mag'].loc[spacecraft.data.index], color=m_traj.cast_intervals['simulcast'].plotparams.model_colors[model_name])
            axs_new[0].fill_between(spacecraft.data.index, 
                                    model['u_mag'].loc[spacecraft.data.index] + model['u_mag_pos_unc'].loc[spacecraft.data.index], 
                                    model['u_mag'].loc[spacecraft.data.index] - model['u_mag_neg_unc'].loc[spacecraft.data.index],
                                    alpha = 0.5, color=m_traj.cast_intervals['simulcast'].plotparams.model_colors[model_name], zorder=1000)
            
            (r_0, std_0), rmsd_0 = TD.find_TaylorStatistics(model['u_mag'].loc[spacecraft.data.index], spacecraft.data['u_mag'])
            
            #  Baseline, compared to unshifted models
            if model_name != 'ensemble':
                model_bl = traj1_backup.models[model_name]
                (r_bl, std_bl), rmsd_bl = TD.find_TaylorStatistics(model_bl['u_mag'].loc[spacecraft.data.index], spacecraft.data['u_mag'])
            
            print('----------------------------------------')
            print(model_name)
            print('r       = {}'.format(r_0))
            print('std_dev = {}'.format(std_0))
            print('rmsd    = {}'.format(rmsd_0))
            if model_name != 'ensemble':
                print('Compared to unshifted baselines of:')
                print('r       = {}'.format(r_bl))
                print('std_dev = {}'.format(std_bl))
                print('rmsd    = {}'.format(rmsd_bl))
            
            # axs_new[1].hist(r, color=m_traj.cast_intervals['simulcast'].plotparams.model_colors[model_name])
            # axs_new[1].axvline(r_0, color='black')
            
            # r_sig, std_sig, rmsd_sig = np.std(r), np.std(std), np.std(rmsd)
            # r, std, rmsd = np.mean(r), np.mean(std), np.mean(rmsd)
            
            ax2.scatter(np.arccos(r_0), std_0, s=36, c=traj0.plotparams.model_colors[model_name], 
                        marker=traj0.plotparams.model_markers[model_name], label=model_name)
            
            # xerr = [[np.arccos(r-r_sig)-np.arccos(r)], [np.arccos(r)-np.arccos(r+r_sig)]]
            # yerr = [[std_sig], [std_sig]]
            
            # ax2.errorbar(np.arccos(r), std, xerr=xerr, yerr=yerr,
            #              color=model_colors[model_name], alpha=0.6)
            # print('For {}: r = {} +/- {}, std = {} +/- {}'.format(model_name,
            #                                                       r, r_sig,
            #                                                       std, std_sig))
            # print('In radians, np.arccos(r) is: {} + {} - {}'.format(np.arccos(r), 
            #                                                          xerr[1], xerr[0]))
            
            ax2.set_rlim([0, 50])
            
        
        ax2.legend(ncols=4, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        fig2.savefig('figures/Paper/' + 'Ensemble_TD_JunoComparison.png', 
                     dpi=300)
        
    plot_EnsembleTD_DTWMLR()
    
    plt.show()
        
    plot_TemporalShifts_All_Total()
    
    return m_traj
