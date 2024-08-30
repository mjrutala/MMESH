#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 29 March 2024

@author: mrutala
"""

import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
import tomllib
import time

import spacecraftdata
import read_SWModel

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

def MMESH_run(config_path):
    import string
    import numpy as np
    import copy
    
    #import sys
    #sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    import MMESH as mmesh
    import MMESH_context as mmesh_c
    import plot_TaylorDiagram as TD
    #breakpoint()
    timer_t0 = time.time()
    
    # =============================================================================
    #     Parse the config .toml file
    # =============================================================================
    with open(config_path, "rb") as f:
        init = tomllib.load(f)
    
    #   This could go into multitrajectory class?
    filepaths = {'output': Path(init['paths']['output']),
                 'figures': Path(init['paths']['output']) / 'figures'}
    
    # =============================================================================
    #   All of the plotting functions have been encapsulated in inner functions
    #   and listed here
    # =============================================================================
   
    # =============================================================================
    #   Figure 4: Taylor Diagram of all alignment methods
    # =============================================================================
    def plot_BothShiftMethodsTD(fullfilename):
        #   This TD initialization is not totally accurate
        #   Since we take the standard deviation after dropping NaNs in the 
        #   reference only-- would be better to drop all NaN rows among ref and
        #   tests first, then take all standard deviations
        
        #fig = plt.figure(figsize=(3,2.5), constrained_layout=True)
        fig, ax = plt.subplots(figsize=(3.5,2.5), subplot_kw={'projection': 'polar'})
        plt.subplots_adjust(left=0.4, bottom=0.15, right=0.9, top=0.9)
        
        points = traj0.baseline('u_mag')
        
        fig, ax = TD.init_TaylorDiagram(points[traj0.spacecraft_name][1], 
                                        fig=fig, ax=ax, half=True,
                                        r_label=r'$\sigma$ of $u_{mag}$',
                                        theta_label=r'$r_P$')
        
        ax.set_rlim(0, 1.33*points[traj0.spacecraft_name][1])
        
        #points = traj0.baseline('u_mag')
        
        for model_name in traj0.model_names:
            
            model_name_text = model_name
            if model_name == 'Tao':
                model_name_text +='+'
            
            ax.scatter(np.arccos(points[model_name][0]), points[model_name][1],
                       s=32, c='black', marker=traj0.gpp('marker', model_name),
                       label=r'{}'.format(model_name_text),
                       zorder=1)
            
            ax.scatter(*constant_shift_dict[model_name],
                       s=32, facecolors=traj0.gpp('color',model_name), marker=traj0.gpp('marker',model_name),
                       edgecolors='black', linewidths=1,
                       label=r'{} + Cons.'.format(model_name_text),
                       zorder=1)
            
            ax.scatter(np.arccos(traj0.best_shifts[model_name]['r']), traj0.best_shifts[model_name]['stddev'],
                       s=32, c=traj0.gpp('color',model_name), marker=traj0.gpp('marker',model_name),
                       label=r'{} + DTW'.format(model_name_text),
                       zorder=1)
            
        #ax.legend(ncols=3, bbox_to_anchor=[0.0,-0.02,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        ax.legend(ncols=1, bbox_to_anchor=[0.0,0.15,0.4,0.9], 
                  bbox_transform=fig.transFigure,
                  loc='lower left', mode='expand', 
                  markerscale=1.0)
        
        #print(fig, ax)
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) 
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0.05)
        
        for suffix in ['.png', '.jpg']:
            fig.savefig(str(fullfilename)+suffix, 
                        dpi=300)#, bbox_inches=extent)
        plt.show()
        
        plt.show()
    
    # =============================================================================
    #   Figure 3: Time series of original models w/ Taylor Diagram
    # =============================================================================
    def plot_OriginalModels(fullfilename):
        param = 'u_mag'
        
        mosaic = []
        for model_name in traj0.model_names:
            mosaic.append([model_name]*2 + ['u_mag_TD'])
        
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,2),
                                      per_subplot_kw={'u_mag_TD': {"projection": "polar"}})
        
        # fig1, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6,2))
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.85,
                            hspace=0.1)
        
        lines = []
        for i, model_name in enumerate(traj0.model_names):
            
            data = axd[model_name].plot(traj0.index_ddoy(traj0.data.index), 
                               traj0.data[param],
                               color='black', alpha=1.0, linewidth=0.5, linestyle='-',
                               marker='o', markersize=0.01,
                               label=traj0.spacecraft_name)
            
            model_name_for_paper = model_name
            if model_name_for_paper == 'Tao': 
                model_name_for_paper += '+'
                
            l = axd[model_name].plot(traj0.index_ddoy(traj0.models[model_name].index), 
                            traj0.models[model_name][param],
                            color=traj0.gpp('color',model_name), linewidth=1.0,
                            marker=traj0.gpp('marker',model_name), markersize=0.01,
                            label=model_name_for_paper)
            lines.append(l[0])
            
            axd[model_name].annotate('({})'.format(string.ascii_lowercase[i]),
                            (0,1), (1,-1), ha='center', va='center',
                            xycoords='axes fraction', 
                            textcoords='offset fontsize')
        
        axd[model_name].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
        _ = [axd[model_name].sharex(axd[traj0.model_names[-1]]) for model_name in traj0.model_names]
        _ = [axd[model_name].sharey(axd[traj0.model_names[-1]]) for model_name in traj0.model_names]
        
        axd[model_name].xaxis.set_major_locator(MultipleLocator(25))
        axd[model_name].xaxis.set_minor_locator(MultipleLocator(5))
        
        axd[model_name].set_xlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
                                   fontsize=plt.rcParams['figure.labelsize'])
        #axd[model_name].xaxis.set_label_coords(0.5,40)
        
        axd[model_name].yaxis.set_major_locator(MultipleLocator(250))
        axd[model_name].yaxis.set_minor_locator(MultipleLocator(50))
        fig.supylabel(r'Flow Speed $u_{mag}$ [km/s]', fontsize=plt.rcParams['figure.labelsize'])
        
        handles = data
        handles.extend(lines)
        labels = [traj0.spacecraft_name]
        labels.extend(traj0.model_names)
        
        bbox = axd[traj0.model_names[0]].get_position()
        fig.legend(handles, labels, loc='upper left', 
                    ncols=4, bbox_to_anchor=[bbox.x0, 
                                             bbox.y0+bbox.height+0.05, 
                                             bbox.width, 
                                             0.1], #mode='expand',
                    markerscale=500.)
        
        points = traj0.baseline('u_mag')
        
        fig, ax = TD.init_TaylorDiagram(points[traj0.spacecraft_name][1], fig=fig, ax=axd['u_mag_TD'], half=True,
                                        r_label=' ',
                                        theta_label=r'$r_{P}$')
        ax.set_rlim(0, 1.33*points[traj0.spacecraft_name][1])
        ax.set_xlabel(r'$\sigma_{u_{mag}}$ [km/s]')
        #ax.xaxis.label.set_position(axd[model_name].xaxis.label.get_position())
        ax.xaxis.set_label_coords(0.5,-0.18)
        
        for model_name in traj0.model_names:
            ax.scatter(np.arccos(points[model_name][0]), points[model_name][1],
                        s=32, zorder=1,
                        c=traj0.gpp('color', model_name), 
                        marker=traj0.gpp('marker', model_name),
                        label=r'{}'.format(model_name))
        
        ax.annotate('({})'.format(string.ascii_lowercase[i+1]),
                    (0,1), (1,-1), ha='center', va='center',
                    xycoords='axes fraction',
                    textcoords='offset fontsize')
        
        
        for suffix in ['.png', '.jpg']:
            plt.savefig(str(fullfilename)+suffix, dpi=300)
        plt.show()
        
        return
    
    # =============================================================================
    #   Figure 8: Time series of shifted models w/ Taylor Diagrams for all params
    # =============================================================================
    def plot_MMESHPerformance(trajectory, fullfilename='', unshifted_trajectory=None):
        #param = 'u_mag'
        
        mosaic = []
        for model_name in trajectory.model_names:
            mosaic.append([model_name]*2 + ['u_mag_TD'])
        mosaic.append(['n_tot_TD', 'p_dyn_TD', 'B_mag_TD'])
        #breakpoint()
        
        #   Make the height of the first n lines 1, then the plot of the n+1 line n
        #   so that the row of Taylor Diagrams are square
        height_ratios = [1]*len(trajectory.model_names) + [len(trajectory.model_names)]
        
        #breakpoint()
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,5), 
                                      height_ratios=height_ratios,
                                      per_subplot_kw={'u_mag_TD': {"projection": "polar"},
                                                      'n_tot_TD': {"projection": "polar"},
                                                      'p_dyn_TD': {"projection": "polar"},
                                                      'B_mag_TD': {"projection": "polar"}})
        
        # fig1, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6,2))
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9,
                            hspace=0.1)
        
        #  Try to hijack the last three axes and shfit them downward
        for label in ['n_tot_TD', 'p_dyn_TD', 'B_mag_TD']:
            axis_pos = axd[label].get_position()
            axd[label].set_position([axis_pos.x0, 0.1, axis_pos.width, axis_pos.height])
        
        lines = []
        for i, model_name in enumerate(trajectory.model_names):
            
            data = axd[model_name].plot(trajectory.index_ddoy(trajectory.data.index), 
                               trajectory.data['u_mag'],
                               color='xkcd:blue grey', alpha=1.0, linewidth=0.5,
                               label=trajectory.spacecraft_name, markersize=0.01, marker='o')
            
            model_name_for_paper = model_name
            if model_name_for_paper == 'Tao': 
                model_name_for_paper += '+'
                
            if 'u_mag_pos_unc' in trajectory.models[model_name].columns:
                axd[model_name].fill_between(trajectory.index_ddoy(trajectory.models[model_name].index), 
                                             trajectory.models[model_name]['u_mag'] + trajectory.models[model_name]['u_mag_pos_unc'], 
                                             trajectory.models[model_name]['u_mag'] - trajectory.models[model_name]['u_mag_neg_unc'],
                                             alpha = 0.5, color=trajectory.gpp('color',model_name),
                                             linewidth=0.5)    
            
            l = axd[model_name].plot(trajectory.index_ddoy(trajectory.models[model_name].index), 
                            trajectory.models[model_name]['u_mag'],
                            color=trajectory.gpp('color',model_name), linewidth=1.0,
                            marker=trajectory.gpp('marker',model_name), markersize=0.01,
                            label=model_name_for_paper)
            lines.append(l[0])
            
            axd[model_name].annotate('({})'.format(string.ascii_lowercase[i]),
                            (0,1), (1,-1), ha='center', va='center',
                            xycoords='axes fraction', 
                            textcoords='offset fontsize')
        
        axd[model_name].set_xlim(trajectory.index_ddoy(trajectory.data.index)[0], trajectory.index_ddoy(trajectory.data.index)[-1])
        _ = [axd[model_name].sharex(axd[trajectory.model_names[-1]]) for model_name in trajectory.model_names]
        _ = [axd[model_name].sharey(axd[trajectory.model_names[-1]]) for model_name in trajectory.model_names]
        
        axd[model_name].xaxis.set_major_locator(MultipleLocator(25))
        axd[model_name].xaxis.set_minor_locator(MultipleLocator(5))
        
        axd[model_name].set_xlabel('Day of Year {}'.format(trajectory._primary_df.index[0].year),
                                   fontsize=plt.rcParams['figure.labelsize'])
        #axd[model_name].xaxis.set_label_coords(0.5,40)
        
        axd[model_name].yaxis.set_major_locator(MultipleLocator(250))
        axd[model_name].yaxis.set_minor_locator(MultipleLocator(50))
        #fig.supylabel(r'Flow Speed $u_{mag}$ [km/s]', fontsize=plt.rcParams['figure.labelsize'])
        
        y_label_xcoord = axd[trajectory.model_names[0]].get_position().x0*0.35
        y_label_ycoord = (axd[trajectory.model_names[0]].get_position().y0 +
                          axd[trajectory.model_names[0]].get_position().height + 
                          axd[trajectory.model_names[-1]].get_position().y0)*0.5
        fig.text(y_label_xcoord, y_label_ycoord,
                 r'Flow Speed $u_{mag}$ [km/s]', 
                 fontsize=plt.rcParams['figure.labelsize'],
                 transform=fig.transFigure, ha='center', va='center', rotation='vertical')
        
        handles = data
        handles.extend(lines)
        labels = [trajectory.spacecraft_name]
        labels.extend(trajectory.model_names)
        
        bbox = axd[trajectory.model_names[0]].get_position()
        fig.legend(handles, labels, loc='lower left', 
                    ncols=4, bbox_to_anchor=[bbox.x0, 
                                             bbox.y0+bbox.height+0.0, 
                                             bbox.width, 
                                             0.09], #mode='expand',
                    markerscale=500.)
        
        for j, var in enumerate(trajectory.variables):
            points = trajectory.baseline(var)
            
            fig, ax = TD.init_TaylorDiagram(points[trajectory.spacecraft_name][1], 
                                            fig=fig, ax=axd[var+'_TD'], half=True,
                                            r_label=' ',
                                            theta_label=r'$r_{P}$')
            ax.set_rlim(0, 1.33*points[trajectory.spacecraft_name][1])
            #ax.set_xlabel(r'$\sigma_{u_{mag}}$ [km/s]')
            ax.set_xlabel(r'$\sigma$ of {}'.format(trajectory._plot_parameters(var)[1]['ylabel']))
            #ax.xaxis.label.set_position(axd[model_name].xaxis.label.get_position())
            ax.xaxis.set_label_coords(0.5,-0.18)
        
            print("For variable {}:".format(var))
            
            for model_name in trajectory.model_names:
                ax.scatter(np.arccos(points[model_name][0]), points[model_name][1],
                            s=32, zorder=1,
                            c=trajectory.gpp('color', model_name), 
                            marker=trajectory.gpp('marker', model_name),
                            label=r'{}'.format(model_name))
                
                print("{} r = {}".format(model_name, points[model_name][0]))
                print("and RMSD = {}".format(points[model_name][2]))
            
            if unshifted_trajectory != None:
                unshifted_points = unshifted_trajectory.baseline(var)
                for model_name in unshifted_trajectory.model_names:
                    ax.scatter(np.arccos(unshifted_points[model_name][0]), 
                               unshifted_points[model_name][1],
                               s=32, zorder=1,
                               c='black', 
                               marker=trajectory.gpp('marker', model_name),
                               label=r'{}'.format(model_name))
                    
                    print("Prior to shifting, ")
                    print("{} r = {}".format(model_name, unshifted_points[model_name][0]))
                    print("and RMSD = {}".format(unshifted_points[model_name][2]))
        
            if var=='p_dyn':
                ax.set_yticks(ax.get_yticks()[0::2])
                #ax.set_yticklabels(ax.get_yticklabels()[0::2])
        
            ax.annotate('({})'.format(string.ascii_lowercase[i+j+1]),
                        (0,1), (1,-1), ha='center', va='center',
                        xycoords='axes fraction',
                        textcoords='offset fontsize')
            
            print("==========================================================")
            
        for suffix in ['.png', '.jpg']:
            plt.savefig(str(fullfilename)+suffix, dpi=300)
        plt.show()
        
        return
    
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
        
    # =============================================================================
    #   Fig 06: All temporal shifts
    # =============================================================================
    def plot_TemporalShifts_All_Total(fullfilename=''):
        fig, axs = plt.subplots(figsize=(3,4.5), nrows=len(mtraj.model_refs), sharex=True, sharey=True, squeeze=False)
        plt.subplots_adjust(left=0.2, top=0.8, right=0.95, hspace=0.075)
        for i, model_ref in enumerate(sorted(mtraj.model_refs)):
            l = []
            c = []
            n = []
            for traj_name, traj in mtraj.trajectories.items():
                l.append(traj.models[model_ref].loc[traj.data_index, 'empirical_time_delta'])  #  Only when overlapping w/ data
                c.append(traj.gpp('color','data'))
                n.append(traj.trajectory_name.replace('_',' '))
            
            #  Hacky
            #c = ['#5d9dd5', '#447abc', '#4034f4', '#75036b'][0:len(mtraj.trajectories)]
            
            axs[i,0].hist(l, range=(-120, 120), bins=20,
                        density=True,
                        histtype='barstacked', stacked=True, label=n, color=c)
            axs[i,0].annotate('({}) {}'.format(string.ascii_lowercase[i], model_ref), 
                        (0,1), (1,-1), ha='left', va='center', 
                        xycoords='axes fraction', textcoords='offset fontsize')
    
        axs[0,0].xaxis.set_major_locator(MultipleLocator(48))
        axs[0,0].xaxis.set_minor_locator(MultipleLocator(12))
        
        axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.001))
        
        fig.supylabel('Density')
        fig.supxlabel('Measured Timing Differences [hours]')
        
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=2,
                   bbox_to_anchor=[0.2, 0.825, 0.75, 0.175], loc='lower left',
                   mode="expand", borderaxespad=0.)

        for suffix in ['.png', '.jpg']:
            plt.savefig(str(fullfilename)+suffix, 
                        dpi=300)
        plt.show()
    #
    def plot_Correlations(fullfilename=''):
        fig, axs = plt.subplots(figsize=(6,4.5), nrows=4, ncols=len(mtraj.model_names), sharex='col', sharey='row', squeeze=False)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.92, right=0.90, 
                            wspace=0.45, hspace=0.025)
        for i, model_name in enumerate(sorted(mtraj.model_names)):
            axs[0,i].set_xlabel(model_name)
            axs[0,i].xaxis.set_label_position('top')
            
            invariant_param = 'empirical_time_delta'
            variant_params = ['solar_radio_flux', 'target_sun_earth_lon', 'target_sun_earth_lat', 'u_mag']
            model_stats = {key: [] for key in variant_params + [invariant_param]}
            for k, (traj_name, traj) in enumerate(mtraj.trajectories.items()):
                
                #   Record individual trajectory info to calculate the overall stats
                model_stats[invariant_param].extend(traj.models[model_name]['empirical_time_delta'].loc[traj.data_index])
                model_stats['solar_radio_flux'].extend(traj.context['solar_radio_flux'].loc[traj.data_index])
                model_stats['target_sun_earth_lon'].extend(traj.context['target_sun_earth_lon'].loc[traj.data_index])
                model_stats['target_sun_earth_lat'].extend(traj.context['target_sun_earth_lat'].loc[traj.data_index])
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
                try:
                    (r,sig), rmsd = TD.find_TaylorStatistics(model_stats[invariant_param], 
                                                             model_stats[variant_param])
                except: breakpoint()
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
         
        for suffix in ['.png', '.jpg']:
            plt.savefig(str(fullfilename) + suffix, 
                        dpi=300)
        plt.show()
    
    def plot_TimeDelta_Grid_AllTrajectories(fullfilepath=''):
        """
        Plot the empirical time deltas, and the fits to each

        Returns
        -------
        None.

        """
        import matplotlib.dates as mdates
        #   Loop over trajectories, getting length of data for plotting
        length_list = []
        for traj_name, traj in mtraj.trajectories.items():
            length_list.append(len(traj.data))
            figsize = (6, 4) # traj.plotprops['figsize']
            adjustments = {'left': 0.1, 'right':0.95, 'bottom':0.1, 'top':0.95, 
                           'wspace':0.1, 'hspace':0.0} #traj.plotprops['adjustments']
        
        #mtraj_model_names = mtraj.model_names
        mtraj_model_refs = mtraj.model_refs  #  ['Tao']  #  OVERRIDE
        
        fig, axs = plt.subplots(figsize=figsize, nrows=len(mtraj_model_refs), ncols=len(mtraj.trajectories), 
                                sharex='col', sharey=True, width_ratios=length_list, squeeze=False)
        plt.subplots_adjust(**adjustments)
        for i, (traj_name, traj) in enumerate(mtraj.trajectories.items()):
            for j, model_ref in enumerate(mtraj_model_refs):
                
                #   Get the indices as decimal days of year
                #   And have that of the trajectory start at the same point as multitrajectory
                mtraj_index = traj.index_ddoy(mtraj.nowcast_dict[model_ref].index)
                traj_index = traj.index_ddoy(traj.models[model_ref].loc[traj.data_index].index, 
                                             start_ref=mtraj.nowcast_dict[model_ref].index[0])
                
                axs[j,i].plot(mtraj_index, 
                              mtraj.nowcast_dict[model_ref]['mean'], color='C0')
                
                axs[j,i].fill_between(mtraj_index, 
                                      mtraj.nowcast_dict[model_ref]['obs_ci_lower'], 
                                      mtraj.nowcast_dict[model_ref]['obs_ci_upper'], 
                                      color='C0', alpha=0.5)
                
                axs[j,i].scatter(traj_index,
                                 traj.models[model_ref]['empirical_time_delta'].loc[traj.data_index], 
                                 color='black', marker='o', s=4)
                
                axs[j,i].axhline(0, linestyle='--', color='gray', alpha=0.6, zorder=-1, linewidth=1)
                
                if j == 0:
                    label_bbox = dict(facecolor=traj.gpp('color','data'), 
                                      edgecolor=traj.gpp('color','data'), 
                                      pad=0.1, boxstyle='round')
                    axs[j,i].annotate('{}'.format(traj.spacecraft_name),
                                      (0.5, 1), xytext=(0, 1), va='center', ha='center',
                                      xycoords='axes fraction', textcoords='offset fontsize',
                                      bbox=label_bbox)
                    
                if i == len(mtraj.trajectories)-1:
                    #axs[j,i].set_ylabel(model_ref)
                    label_bbox = dict(facecolor=traj.gpp('color',model_ref), 
                                      edgecolor=traj.gpp('color',model_ref), 
                                      pad=0.1, boxstyle='round')
                    
                    axs[j,i].annotate('{}'.format(mtraj.model_sources[model_ref]),
                                      (1.0, 0.5), xytext=(1, 0), va='center', ha='center',
                                      xycoords='axes fraction', textcoords='offset fontsize',
                                      rotation='vertical', bbox=label_bbox)
                    
                # axs[j,i].set_xlim(traj.models[model_ref].index[0], 
                #                   traj.models[model_ref].index[-1])
                
            # tick_interval = 50
            # subtick_interval = 10
            
            axs[0,i].set_xlim(traj_index[0], 
                              traj_index[-1])
            # axs[0,i].xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
            # axs[0,i].xaxis.set_minor_locator(mdates.DayLocator(interval=subtick_interval))
            
            # #axs[0,i].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
            # axs[0,i].set_xticks(axs[0,i].get_xticks()[::2])
            # axs[0,i].xaxis.set_major_formatter(traj.timeseries_formatter(axs[0,i].get_xticks()))
                
            
            axs[0,i].xaxis.set_major_locator(MultipleLocator(50))
            axs[0,i].xaxis.set_minor_locator(MultipleLocator(10))
        
        for i, ax in enumerate(axs.flatten()):
            ax.annotate('({})'.format(string.ascii_lowercase[i]),
                        (0,1), (1, -1), xycoords='axes fraction', textcoords='offset fontsize')
        fig.supxlabel('Days Since Jan. 1, {}'.format(mtraj.nowcast_dict[model_ref].index[0].year))
        fig.supylabel('Measured Timing Differences [hours]')
        
        for suffix in ['.png', '.jpg']:
            fig.savefig(str(fullfilepath)+suffix, 
                        transparent=True,
                        dpi=600)
        plt.plot()
        
        return
    
    trajectories = []
    for epoch_name, epoch_info in init['epochs'].items():
        start = epoch_info['start']
        stop  = epoch_info['stop']
        
        reference_frame = 'SUN_INERTIAL'
        observer = 'SUN'
        
        #  Load spacecraft data
        spacecraft = spacecraftdata.SpacecraftData(epoch_info['source'])
        spacecraft.read_processeddata(start, stop, resolution='60Min')
        pos_TSE = spacecraft.find_StateToEarth()

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
            
            # traj0.set_plotprops('color', model_ref, model_info['color'])  #  Optional
            # traj0.set_plotprops('marker', model_ref, model_info['marker'])  #  Optional
        
        # =============================================================================
        #   Plot a Taylor Diagram of the unchanged models
        # =============================================================================
        # plot_OriginalModels(filepaths['figures']/'fig03_OriginalModels')        
        
        #plot_OriginalModels(filepaths['figures']/'fig03_OriginalModels')        
        
        # =============================================================================
        #   Optimize the models via constant temporal shifting
        #   Then plot:
        #       - The changes in correlation coefficient 
        #         (more generically, whatever combination is being optimized)
        #       - The changes on a Taylor Diagram
        # =============================================================================
        # shifts = np.arange(-96, 96+6, 6)    #  in hours
        # shift_stats = traj0.optimize_shifts('u_mag', shifts=shifts)
        
        # constant_shift_dict = {}
        # for model_name in traj0.model_names:
        #     constant_shift_dict[model_name] = (np.arccos(traj0.best_shifts[model_name]['r']), 
        #                                        traj0.best_shifts[model_name]['stddev'])
        def poster_plot_1(fullfilename):
            param = 'u_mag'  #  what to plot
            
            params = {"ytick.color" : "w",
                      "xtick.color" : "w",
                      "axes.labelcolor" : "w",
                      "axes.edgecolor" : "w"}
            plt.rcParams.update(params)
            
            #   Plot as one row
            # mosaic = [[]]
            # for model_ref in traj0.model_names:
            #     mosaic[0].append(model_ref)
            #
            # fig, axd = plt.subplot_mosaic(mosaic, figsize=(12,1.5), sharex=True, sharey=True)
            # plt.subplots_adjust(bottom=0.15, left=0.05625, right=1-0.00625, top=0.95, 
            #                     wspace=0.025)
            
            #   Plot as one column
            mosaic = []
            for model_ref in traj0.model_names:
                mosaic.append([model_ref])
            
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,3), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.16, left=0.120, right=0.9925, top=0.99, 
                                hspace=0.1)
            
            for i, model_ref in enumerate(traj0.model_names):
                
                for axis in ['top','bottom','left','right']:
                    axd[model_ref].spines[axis].set_linewidth(2)
                axd[model_ref].tick_params(width=2, which='major')
                axd[model_ref].tick_params(width=1.5, which='minor')
                
                axd[model_ref].plot(traj0.index_ddoy(traj0.data.index), 
                                     traj0.data[param],
                                     color='black', alpha=1.0, linewidth=1.5,
                                     marker='o', markersize=0.01,
                                     label=traj0.spacecraft_name)
                    
                axd[model_ref].plot(traj0.index_ddoy(traj0.models[model_ref].index),
                                     traj0.models[model_ref][param],
                                     color=traj0.gpp('color',model_ref), linewidth=2,
                                     #marker=traj0.gpp('marker',model_ref), markersize=0.01,
                                     label = traj0.model_sources[model_ref])
                
                label_bbox = dict(facecolor = traj0.gpp('color',model_ref), 
                                  edgecolor = traj0.gpp('color',model_ref), 
                                  pad = 0.1, boxstyle = 'round')
                axd[model_ref].annotate('{}'.format(traj0.model_sources[model_ref]),
                                (0,1), (0.5,-0.5), ha='left', va='top',
                                xycoords='axes fraction', 
                                textcoords='offset fontsize',
                                bbox = label_bbox)
            
            axd[model_ref].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
            
            axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
            axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
            
            fig.supxlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
                          fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            axd[model_ref].set_ylim((350, 650))
            axd[model_ref].yaxis.set_major_locator(MultipleLocator(100))
            axd[model_ref].yaxis.set_minor_locator(MultipleLocator(25))
            fig.supylabel(r'Flow Speed $u_{mag}$ [km/s]', x=0.00625, fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            for suffix in ['.png', '.jpg']:
                plt.savefig(str(fullfilename)+suffix, 
                            dpi = 800,
                            transparent = True)
            plt.show()
            
            return
        
        poster_plot_1(filepaths['figures']/'poster01_OriginalModels')
        
        # # =============================================================================
        # #   Optimize the models via constant temporal shifting
        # #   Then plot:
        # #       - The changes in correlation coefficient 
        # #         (more generically, whatever combination is being optimized)
        # #       - The changes on a Taylor Diagram
        # # =============================================================================
        shifts = np.arange(-96, 96+6, 6)    #  in hours
        shift_stats = traj0.optimize_shifts('u_mag', shifts=shifts)
        
        constant_shift_dict = {}
        for model_name in traj0.model_names:
            constant_shift_dict[model_name] = (np.arccos(traj0.best_shifts[model_name]['r']), 
                                                traj0.best_shifts[model_name]['stddev'])
        
        # fig, axs = traj0.plot_ConstantTimeShifting_Optimization()
        # plt.show()
        
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
        
        #traj0.plot_SingleTimeseries('u_mag', starttime, stoptime, filepath=figurefilepath)
        #traj0.plot_SingleTimeseries('jumps', start, stop, fullfilepath=filepaths['figures']/'figA1_Binarization')
        
        def poster_plot_2(fullfilename):
            params = {"ytick.color" : "w",
                      "xtick.color" : "w",
                      "axes.labelcolor" : "w",
                      "axes.edgecolor" : "w"}
            plt.rcParams.update(params)
            
            #   Plot as one row
            # mosaic = [[]]
            # for model_ref in traj0.model_names:
            #     mosaic[0].append(model_ref)
            # fig, axd = plt.subplot_mosaic(mosaic, figsize=(12,1.5), sharex=True, sharey=True)
            # plt.subplots_adjust(bottom=0.15, left=0.05625, right=1-0.00625, top=0.95, 
            #                     wspace=0.025)
            
            #   Plot as one column
            mosaic = []
            for model_ref in traj0.model_names:
                mosaic.append([model_ref])
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,3), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.16, left=0.120, right=0.9925, top=0.99, 
                                hspace=0.1)
            
            for i, model_ref in enumerate(traj0.model_names):
                
                for axis in ['top','bottom','left','right']:
                    axd[model_ref].spines[axis].set_linewidth(2)
                axd[model_ref].tick_params(width=2, which='major')
                axd[model_ref].tick_params(width=1.5, which='minor')
                
                axd[model_ref].scatter(traj0.index_ddoy(traj0.data.index), 
                                      (traj0.data['jumps']-0.5)*2,
                                      color = 'black', alpha = 1.0,
                                      marker = 'x', s = 100.0,
                                      label = traj0.spacecraft_name)

                axd[model_ref].scatter(traj0.index_ddoy(traj0.models[model_ref].index),
                                      (traj0.models[model_ref]['jumps']-0.5)*4,
                                      color = traj0.gpp('color',model_ref),
                                      marker = 'x', s = 100.0,
                                      label = traj0.model_sources[model_ref])
                
                label_bbox = dict(facecolor = traj0.gpp('color',model_ref), 
                                  edgecolor = traj0.gpp('color',model_ref), 
                                  pad = 0.1, boxstyle = 'round')
                axd[model_ref].annotate('{}'.format(traj0.model_sources[model_ref]),
                                (0,1), (0.5,-0.5), ha='left', va='top',
                                xycoords='axes fraction', 
                                textcoords='offset fontsize',
                                bbox = label_bbox)
            
            axd[model_ref].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
            
            axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
            axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
            
            fig.supxlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
                          fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            axd[model_ref].set_ylim((0.5,3))
            axd[model_ref].set(yticks=(1, 2), yticklabels=('Data', 'Model'))
            
            # fig.supylabel(r'Binarized Time Series', x=0.00625, fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            for suffix in ['.png', '.jpg']:
                plt.savefig(str(fullfilename)+suffix, 
                            dpi=800,
                            transparent=True)
            plt.show()
            
            return
        
        poster_plot_2(filepaths['figures']/'poster02_Binarized')
        
        #   Calculate a whole host of statistics
        dtw_stats = traj0.find_WarpStatistics('jumps', 'u_mag', shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)

        #   Write an equation describing the optimization equation
        #   This can be played with
        def optimization_eqn(df):
            f = df['r'] + (1 - (0.5*df['width_68'])/96.)
            #f = df['r'] / (0.5 * df['width_68'])
            #f = 2.0*df['r'] + 1/(0.5*df['width_68'])  #   normalize width so 24 hours == 1
            return (f - np.min(f))/(np.max(f)-np.min(f))
        
        #   Plug in the optimization equation
        traj0.optimize_Warp(optimization_eqn)
        
        #filename = basefilepath+'/OptimizedDTWOffset_{}_{}.png'.format(traj0.trajectory_name, '-'.join(traj0.model_names))
        
        # traj0.plot_OptimizedOffset('jumps', 'u_mag', fullfilepath=filepaths['figures']/'fig04_DTWIllustration')
        
        def poster_plot_3(fullfilename):
            import matplotlib.dates as mdates
            from matplotlib import collections  as mc
            from matplotlib.patches import ConnectionPatch
            
            params = {"ytick.color" : "w",
                      "xtick.color" : "w",
                      "axes.labelcolor" : "w",
                      "axes.edgecolor" : "w"}
            plt.rcParams.update(params)
            
            #   Plot as one row
            # mosaic = [[]]
            # for model_ref in traj0.model_names:
            #     mosaic[0].append(model_ref)
            # fig, axd = plt.subplot_mosaic(mosaic, figsize=(12,1.5), sharex=True, sharey=True)
            # plt.subplots_adjust(bottom=0.15, left=0.05625, right=1-0.00625, top=0.95, 
            #                     wspace=0.025)
            
            #   Plot as one column
            mosaic = []
            for model_ref in traj0.model_names:
                mosaic.append([model_ref])
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,3), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.16, left=0.120, right=0.9925, top=0.99, 
                                hspace=0.1)
            
            for i, model_ref in enumerate(traj0.model_names):
                
                for axis in ['top','bottom','left','right']:
                    axd[model_ref].spines[axis].set_linewidth(2)
                axd[model_ref].tick_params(width=2, which='major')
                axd[model_ref].tick_params(width=1.5, which='minor')
                
                axd[model_ref].scatter(traj0.index_ddoy(traj0.data.loc[traj0.data['jumps']!=0, 'jumps'].index), 
                                  traj0.data.loc[traj0.data['jumps']!=0, 'jumps'].values,
                                  color='black', edgecolors='black',
                                  marker='x', s = 100,
                                  zorder=3)
            
                axd[model_ref].scatter(traj0.index_ddoy(traj0.models[model_ref].loc[traj0.models[model_ref]['jumps']!=0, 'jumps'].index), 
                                  traj0.models[model_ref].loc[traj0.models[model_ref]['jumps']!=0, 'jumps'].values*2,
                                  color = traj0.gpp('color',model_ref), edgecolors='black',
                                  marker = 'x', s = 100,
                                  zorder=2)
                
                label_bbox = dict(facecolor = traj0.gpp('color',model_ref), 
                                  edgecolor = traj0.gpp('color',model_ref), 
                                  pad = 0.1, boxstyle = 'round')
                axd[model_ref].annotate('{}'.format(traj0.model_sources[model_ref]),
                                (0,1), (0.5,-0.5), ha='left', va='top',
                                xycoords='axes fraction', 
                                textcoords='offset fontsize',
                                bbox = label_bbox)
                                                     
                connections_basis = []
                for i, (index, value) in enumerate(traj0.models[model_ref].loc[traj0.models[model_ref]['jumps']!=0, 'jumps'].items()):
                    
                    #  Find the time-coordinate for where each 'jump' got shifted to
                    t_elapsed = (traj0.models[model_ref].index - traj0.models[model_ref].index[0]).total_seconds()/3600.
                    x = t_elapsed - traj0.models[model_ref].loc[:, 'empirical_time_delta'].values
                    x_tie = (index - traj0.models[model_ref].index[0]).total_seconds()/3600.
                    shift_index_elapsed = np.interp(x_tie, x, t_elapsed)
                    
                    shift_index = index + dt.timedelta(hours=(shift_index_elapsed-x_tie))
                    
                    #   !!! Can't use built-in self.index_ddoy with an individual entry...
                    #   
                    index_ddoy = (index - dt.datetime(traj0.data_index[0].year,1,1)).total_seconds()/(24*60*60) + 1/24.
                    shift_index_ddoy = (shift_index - dt.datetime(traj0.data_index[0].year,1,1)).total_seconds()/(24*60*60) + 1/24.
                    
                    #   This first plot shows the matched features, so only add
                    #   the connection if both features exist                
                    if shift_index in traj0.data.index:
                        if traj0.data.loc[shift_index, 'jumps'] == 1:
                            pair1 = (index_ddoy, 2)
                            pair2 = (shift_index_ddoy, 1)
                            
                            connections_basis.append([pair1, pair2])
                
                lc = mc.LineCollection(connections_basis, 
                                        linewidths=1.5, linestyles="-", color='gray', linewidth=1.5, zorder=1)
                axd[model_ref].add_collection(lc)
            
            axd[model_ref].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
            
            axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
            axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
            
            fig.supxlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
                          fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            axd[model_ref].set_ylim((0.5,3))
            axd[model_ref].set(yticks=(1, 2), yticklabels=('Data', 'Model'))
            
            # fig.supylabel(r'Binarized Time Series', x=0.00625, fontsize=plt.rcParams['figure.labelsize'])
            
            for suffix in ['.png', '.eps', '.jpg']:
                fig.savefig(fullfilename, dpi=800, transparent=True)
            plt.show()
            return
        
        def poster_plot_4(fullfilename):
            import matplotlib.dates as mdates
            from matplotlib import collections  as mc
            from matplotlib.patches import ConnectionPatch
            
            params = {"ytick.color" : "w",
                      "xtick.color" : "w",
                      "axes.labelcolor" : "w",
                      "axes.edgecolor" : "w"}
            plt.rcParams.update(params)
            
            #   Plot as one row
            # mosaic = [[]]
            # for model_ref in traj0.model_names:
            #     mosaic[0].append(model_ref)
            # fig, axd = plt.subplot_mosaic(mosaic, figsize=(12,1.5), sharex=True, sharey=True)
            # plt.subplots_adjust(bottom=0.15, left=0.05625, right=1-0.00625, top=0.95, 
            #                     wspace=0.025)
            
            #   Plot as one column
            mosaic = []
            for model_ref in traj0.model_names:
                mosaic.append([model_ref])
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,3), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.16, left=0.120, right=0.9925, top=0.99, 
                                hspace=0.1)
            
            for i, model_ref in enumerate(traj0.model_names):
                
                for axis in ['top','bottom','left','right']:
                    axd[model_ref].spines[axis].set_linewidth(2)
                axd[model_ref].tick_params(width=2, which='major')
                axd[model_ref].tick_params(width=1.5, which='minor')
                
                axd[model_ref].plot(traj0.index_ddoy(traj0.data.index), 
                                     traj0.data['u_mag'],
                                     color='black', alpha=1.0, linewidth=1.5,
                                     marker='o', markersize=0.01,
                                     label=traj0.spacecraft_name)
                    
                axd[model_ref].plot(traj0.index_ddoy(traj0.models[model_ref].index),
                                     traj0.models[model_ref]['u_mag'],
                                     color=traj0.gpp('color',model_ref), linewidth=2, linestyle=':',
                                     #marker=traj0.gpp('marker',model_ref), markersize=0.01,
                                     label = traj0.model_sources[model_ref])
                
                temp_shift = traj0._shift_Models(time_delta_column='empirical_time_delta')       
                axd[model_ref].plot(traj0.index_ddoy(traj0.models[model_ref].index), temp_shift[(model_ref, 'u_mag')],
                               color = traj0.gpp('color',model_ref), linewidth=2)  
                
                label_bbox = dict(facecolor = traj0.gpp('color',model_ref), 
                                  edgecolor = traj0.gpp('color',model_ref), 
                                  pad = 0.1, boxstyle = 'round')
                axd[model_ref].annotate('{}'.format(traj0.model_sources[model_ref]),
                                (0,1), (0.5,-0.5), ha='left', va='top',
                                xycoords='axes fraction', 
                                textcoords='offset fontsize',
                                bbox = label_bbox)
                                                     
                connections_metric = []
                for i, (index, value) in enumerate(traj0.models[model_ref].loc[traj0.models[model_ref]['jumps']!=0, 'jumps'].items()):
                    
                    #  Find the time-coordinate for where each 'jump' got shifted to
                    t_elapsed = (traj0.models[model_ref].index - traj0.models[model_ref].index[0]).total_seconds()/3600.
                    x = t_elapsed - traj0.models[model_ref].loc[:, 'empirical_time_delta'].values
                    x_tie = (index - traj0.models[model_ref].index[0]).total_seconds()/3600.
                    shift_index_elapsed = np.interp(x_tie, x, t_elapsed)
                    
                    shift_index = index + dt.timedelta(hours=(shift_index_elapsed-x_tie))
                    
                    #   !!! Can't use built-in self.index_ddoy with an individual entry...
                    #   
                    index_ddoy = (index - dt.datetime(traj0.data_index[0].year,1,1)).total_seconds()/(24*60*60) + 1/24.
                    shift_index_ddoy = (shift_index - dt.datetime(traj0.data_index[0].year,1,1)).total_seconds()/(24*60*60) + 1/24.
                    
                    #   For the shifted abcissa (y) value, just take the unshifted y value
                    #   Shifts only change x, not y, so the y value should not change
                    #   And this saves having to interpolate
                    pair3 = (index_ddoy, traj0.models[model_ref].loc[index, 'u_mag'])
                    pair4 = (shift_index_ddoy, traj0.models[model_ref].loc[index, 'u_mag'])
                    
                    connections_metric.append([pair3, pair4])
                    
                    # axd[model_ref].plot(pair3, pair4,
                    #                     linewidth=1.5, color='gray')
                    axd[model_ref].annotate("", xy=pair4, xytext=pair3,
                                            arrowprops=dict(arrowstyle="->", linewidth=1, color='xkcd:orange red'))
                    
                    # con = ConnectionPatch(xyA=pair3, xyB=pair4, 
                    #                      coordsA="data", coordsB="data",
                    #                      axesA=ax_col[1], axesB=ax_col[2],
                    #                      color="gray", linewidth=1.5, linestyle=":", alpha=0.8, zorder=5)
                    # ax_col[2].add_artist(con)
                
                lc = mc.LineCollection(connections_metric, 
                                       linewidths=1.5, linestyles="-", color='gray', linewidth=1.5, zorder=1)
                #axd[model_ref].add_collection(lc)
            
            axd[model_ref].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
            
            axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
            axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
            
            fig.supxlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
                          fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            axd[model_ref].set_ylim((350, 650))
            axd[model_ref].yaxis.set_major_locator(MultipleLocator(100))
            axd[model_ref].yaxis.set_minor_locator(MultipleLocator(25))
            fig.supylabel(r'Flow Speed $u_{mag}$ [km/s]', x=0.00625, fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            for suffix in ['.png', '.jpg']:
                fig.savefig(fullfilename, dpi=800, transparent=True)
            plt.show()
            return

        poster_plot_3(filepaths['figures']/'poster03_DTWLinks')
        poster_plot_4(filepaths['figures']/'poster04_TimeSeriesShifts')
        
        
        traj0.plot_DynamicTimeWarping_Optimization()
        #traj0.plot_DynamicTimeWarping_TD()
        
        # plot_BothShiftMethodsTD(filepaths['figures']/'fig05_ShiftMethodComparison_TD')
        
        #plot_BestShiftWarpedTimeDistribution()
        
        #   See the effects of the Constant + DTW Shifts
        # traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
        # traj0.shift_Models(time_delta_column='empirical_time_delta')
        # traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
        
        # =============================================================================
        #             
        # =============================================================================
        srf = mmesh_c.read_SolarRadioFlux(traj0._primary_df.index[0], traj0._primary_df.index[-1])
        # traj0._primary_df[('context', 'solar_radio_flux')] = srf['adjusted_flux']   
        # traj0._primary_df[('context', 'target_sun_earth_lon')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon']
        # traj0._primary_df[('context', 'target_sun_earth_lat')] = pos_TSE.reindex(index=traj0._primary_df.index)['del_lat']
        traj0.context = srf['adjusted_flux'].to_frame(name='solar_radio_flux') 
        traj0.context = pos_TSE.reindex(index=traj0._primary_df.index)['del_lon'].to_frame(name='target_sun_earth_lon')
        traj0.context = pos_TSE.reindex(index=traj0._primary_df.index)['del_lat'].to_frame(name='target_sun_earth_lat')

        trajectories.append(traj0)
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Finished initializing MMESH with models. ")
    print("Time elapsed: {} s".format(time.time()-timer_t0))
    print("--------------------------------------------------")
    timer_t0 = time.time()
    
    # =============================================================================
    #   Initialize a MultiTrajecory object
    # =============================================================================
    mtraj = mmesh.MultiTrajectory(trajectories=trajectories)
    
    #plot_Correlations() 
    
    # #   Set up the prediction interval
    # #original_spantime = stoptime - starttime
    prediction_starttime = dt.datetime(2016, 5, 16)  #  starttime - original_spantime
    prediction_stoptime = dt.datetime(2016, 6, 26)  #  stoptime + original_spantime
    prediction_target = 'Juno'
    
    #  Initialize a trajectory class for the predictions
    cast_traj_juno = mmesh.Trajectory()
    
    spacecraft = spacecraftdata.SpacecraftData('Juno')
    spacecraft.read_processeddata(prediction_starttime, prediction_stoptime, resolution='60Min')
    cast_traj_juno.addData('Juno', spacecraft.data)
    #cast_traj_juno.set_plotprops('color', 'data', epoch_colors[epoch_name])
    
    fullfilepath = mmesh_c.make_PlanetaryContext_CSV(prediction_target, 
                                                     prediction_starttime, 
                                                     prediction_stoptime, 
                                                     filepath=filepaths['output'] / 'context/')
    cast_traj_juno.context = pd.read_csv(fullfilepath, index_col='datetime', parse_dates=True)
    
    #  Read models and add to trajectory
    for model_ref, model_info in init['models'].items():
        
        model = read_SWModel.read_model(model_info['source'], prediction_target, 
                                        prediction_starttime, prediction_stoptime, 
                                        resolution='60Min')
        
        cast_traj_juno.addModel(model_ref, model, model_source=model_info['source'])
    
        # cast_traj_juno.set_plotprops('color', model_name, model_colors[model_name])  #  Optional
        # cast_traj_juno.set_plotprops('marker', model_name, model_markers[model_name])  #  Optional
        
    # cast_traj_juno_backup =  copy.deepcopy(cast_traj_juno)
    
    #   Add prediction Trajectory as simulcast
    mtraj.cast_intervals['juno_hindcast'] = cast_traj_juno
    
    #   Set up the prediction interval
    #original_spantime = stoptime - starttime
    forecast_starttime = dt.datetime(2016, 3, 1)  #  starttime - original_spantime
    forecast_stoptime = dt.datetime(2017, 3, 1)  #  stoptime + original_spantime
    forecast_target = 'Jupiter'
    
    #  Initialize a trajectory class for the predictions
    cast_traj_jupiter = mmesh.Trajectory()
    fullfilepath = mmesh_c.make_PlanetaryContext_CSV(forecast_target, 
                                                     forecast_starttime, 
                                                     forecast_stoptime, 
                                                     filepath=filepaths['output'] / 'context/')
    cast_traj_jupiter.context = pd.read_csv(fullfilepath, index_col='datetime', parse_dates=True)
    
    #  Read models and add to trajectory
    for model_ref, model_info in init['models'].items():
        
        model = read_SWModel.read_model(model_info['source'], forecast_target, 
                                        forecast_starttime, forecast_stoptime, 
                                        resolution='60Min')
        
        cast_traj_jupiter.addModel(model_ref, model, model_source=model_info['source'])
        
        # cast_traj_jupiter.set_plotprops('color', model_name, model_colors[model_name])  #  Optional
        # cast_traj_jupiter.set_plotprops('marker', model_name, model_markers[model_name])  #  Optional

    #   Add prediction Trajectory as simulcast
    mtraj.cast_intervals['jupiter_hindcast'] = cast_traj_jupiter
    
    #formula = "empirical_time_delta ~ solar_radio_flux + target_sun_earth_lon"  #  This can be input
    #formula = "empirical_time_delta ~ target_sun_earth_lat + u_mag"
    formula = "empirical_time_delta ~ target_sun_earth_lat + solar_radio_flux + u_mag"
    test = mtraj.linear_regression(formula)
    breakpoint()
    print(time.time() - timer_t0)
    #breakpoint()
    
    mtraj.cast_Models(with_error=True)
    
    mtraj.cast_intervals['jupiter_simulcast'].sample()
    breakpoint()
    
    #   "Reacceleration Section"
    variable = 'p_dyn'
    trajectory_name = list(mtraj.trajectories.keys())[0]
    
    subset_df = mtraj.trajectories[trajectory_name]._primary_df.xs(variable, axis=1, level=1)
    
    #   Plot cumulative distribution functions
    fig, ax = plt.subplots()
    
    #   Data CDF
    cdf_y = np.linspace(0, 1, 100)
    temp_x = np.sort(subset_df['Juno'].dropna()).astype('float64')
    temp_y = np.linspace(0, 1, len(temp_x))
    cdf_data = np.interp(cdf_y, temp_y, temp_x)
    
    #   Model CDFs
    cdf_models = {}
    for col in subset_df.columns:
        # bins = np.logspace(-3, 1, 101)
        # hist, _ = np.histogram(subset_df[col], bins=bins, 
        #                        density = True)
        # ax.plot(bins[1:], np.cumsum(hist)*(bins[-1] - bins[:-1]),
        #         label = col)
        
        temp_x = np.sort(subset_df[col].dropna()).astype('float64')
        temp_y = np.linspace(0, 1, len(temp_x))
        cdf_model = np.interp(cdf_y, temp_y, temp_x)
        cdf_models[col] = cdf_model
        
        ax.plot(cdf_model, cdf_y, label = col)
        
    ax.set(xscale = 'log')
    ax.legend()
        
    
    print(time.time() - timer_t0)
    #breakpoint()
    
    plot_TimeDelta_Grid_AllTrajectories(fullfilepath=filepaths['figures']/'fig07_MLRTimingFits')
    
    #mtraj.cast_intervals['juno_simulcast'].ensemble()
    breakpoint()
    mtraj.cast_intervals['jupiter_simulcast'].ensemble()
    def poster_plot_5(fullfilename):
        import matplotlib.dates as mdates
        
        params = {"ytick.color" : "w",
                  "xtick.color" : "w",
                  "axes.labelcolor" : "w",
                  "axes.edgecolor" : "w"}
        plt.rcParams.update(params)
            
        #   Loop over trajectories, getting length of data for plotting
        length_list = []
        for traj_name, traj in mtraj.trajectories.items():
            length_list.append(len(traj.data))
            
        #   Separate plot for each trajectory
        for i, (traj_name, traj) in enumerate(mtraj.trajectories.items()):
            
            #   Plot as one row
            # mosaic = [[]]
            # for model_ref in traj.model_names:
            #     mosaic[0].append(model_ref)
            # fig, axd = plt.subplot_mosaic(mosaic, figsize=(12,1.5), sharex=True, sharey=True)
            # plt.subplots_adjust(bottom=0.15, left=0.05625, right=1-0.00625, top=0.95, 
            #                     wspace=0.025)
            
            #   Plot as one column
            mosaic = []
            for model_ref in traj0.model_names:
                mosaic.append([model_ref])
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,3), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.16, left=0.120, right=0.9925, top=0.99, 
                                hspace=0.1)
            
            traj_model_refs = traj.model_names  #  !!!! Rename this in trajectory class eventurally
            for j, model_ref in enumerate(traj_model_refs):
                
                for axis in ['top','bottom','left','right']:
                    axd[model_ref].spines[axis].set_linewidth(2)
                axd[model_ref].tick_params(width=2, which='major')
                axd[model_ref].tick_params(width=1.5, which='minor')
                
                #   Get the indices as decimal days of year
                #   And have that of the trajectory start at the same point as multitrajectory
                mtraj_index = traj.index_ddoy(mtraj.nowcast_dict[model_ref].index)
                traj_index = traj.index_ddoy(traj.models[model_ref].loc[traj.data_index].index, 
                                             start_ref=mtraj.nowcast_dict[model_ref].index[0])
                
                axd[model_ref].plot(traj.index_ddoy(traj.data.index), 
                                    mtraj.nowcast_dict[model_ref].loc[traj.data.index, 'mean'], 
                                    color='xkcd:orange red')
                
                axd[model_ref].fill_between(traj.index_ddoy(traj.data.index), 
                                            mtraj.nowcast_dict[model_ref].loc[traj.data.index, 'obs_ci_lower'], 
                                            mtraj.nowcast_dict[model_ref].loc[traj.data.index, 'obs_ci_upper'], 
                                            color='xkcd:orange red', alpha=0.5)
                
                axd[model_ref].scatter(traj.index_ddoy(traj.data.index),
                                       traj.models[model_ref]['empirical_time_delta'].loc[traj.data.index], 
                                       color='black', marker='*', s=4)
                
                axd[model_ref].axhline(0, linestyle='--', color='gray', alpha=0.6, zorder=-1, linewidth=1)
                
                label_bbox = dict(facecolor = traj.gpp('color',model_ref), 
                                  edgecolor = traj.gpp('color',model_ref), 
                                  pad = 0.1, boxstyle = 'round')
                axd[model_ref].annotate('{}'.format(traj.model_sources[model_ref]),
                                (0,1), (0.5,-0.5), ha='left', va='top',
                                xycoords='axes fraction', 
                                textcoords='offset fontsize',
                                bbox = label_bbox)
            
            axd[model_ref].set_xlim(traj.index_ddoy(traj.data.index)[0], traj0.index_ddoy(traj.data.index)[-1])
            
            axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
            axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
            
            # fig.supxlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
            #               fontsize=plt.rcParams['figure.labelsize'])
            
            axd[model_ref].set_ylim((-100,100))
            axd[model_ref].yaxis.set_major_locator(MultipleLocator(48))
            axd[model_ref].yaxis.set_minor_locator(MultipleLocator(12))
                
            fig.supxlabel('Days Since Jan. 1, {}'.format(traj.data.index.year[0]),
                          fontsize=plt.rcParams['figure.labelsize'], color='w')
            fig.supylabel('Timing Offsets [hrs]',  x=0.00625, fontsize=plt.rcParams['figure.labelsize'], color='w')
            
            for suffix in ['.png', '.jpg']:
                fig.savefig(str(fullfilename)+traj_name+suffix, 
                            transparent=True,
                            dpi=800)
            plt.show()
        
        return
    
    poster_plot_5(filepaths['figures']/'poster05_MLR')
    breakpoint()
    
    mtraj.cast_intervals['juno_hindcast'].ensemble()
    mtraj.cast_intervals['jupiter_hindcast'].ensemble()

    # plot_MMESHPerformance(mtraj.cast_intervals['juno_simulcast'], 
    #                       filepaths['figures']/'Fig08_MMESHPerformance',
    #                       unshifted_trajectory=cast_traj_juno_backup)
    
    #  !!!! ++++++++++++++++++
    #   FOR TESTING ACCURACY
    #  !!!! KEEP THIS BELOW
    #  !!!! ------------------
    # def plot_ComparisonResults():
        
    #     fig, axs = plt.subplots(figsize=mtraj.cast_intervals['juno_simulcast'].plotprops['figsize'],
    #                             nrows=len(mtraj.cast_intervals['juno_simulcast'].model_names), sharex=True)
    #     plt.subplots_adjust(**mtraj.cast_intervals['juno_simulcast'].plotprops['adjustments'])
        
    #     for i, model_name in enumerate(mtraj.cast_intervals['juno_simulcast'].model_names):
    #         model = mtraj.cast_intervals['juno_simulcast'].models[model_name]
    
    #         axs[i].plot(spacecraft.data.index, spacecraft.data['u_mag'],
    #                         color='black', linewidth=1)
            
    #         if 'u_mag_pos_unc' in model.columns:
    #             axs[i].fill_between(model.index, 
    #                                 model['u_mag'] + model['u_mag_pos_unc'], 
    #                                 model['u_mag'] - model['u_mag_neg_unc'],
    #                                 alpha = 0.5, color=mtraj.cast_intervals['juno_simulcast'].gpp('color', model_name),
    #                                 linewidth=0.5)
            
    #         axs[i].plot(model.index, model['u_mag'], color=mtraj.cast_intervals['juno_simulcast'].gpp('color', model_name),
    #                     linewidth=1.5)
            
    #         #   Overplot the original model, skipping the ensemble
    #         if model_name in cast_traj_juno_backup .model_names:
                
    #             axs[i].plot(cast_traj_juno_backup .models[model_name].index, 
    #                         cast_traj_juno_backup .models[model_name]['u_mag'], 
    #                         color='gray', linewidth=1.5, alpha=0.8)
            
    #         label_bbox = dict(facecolor=mtraj.cast_intervals['juno_simulcast'].gpp('color', model_name), 
    #                           edgecolor=mtraj.cast_intervals['juno_simulcast'].gpp('color', model_name), 
    #                           pad=0.1, boxstyle='round')
    #         axs[i].annotate(model_name, (1,0.5), (0.2,0), va='center', ha='left',
    #                         xycoords='axes fraction', textcoords='offset fontsize',
    #                         bbox = label_bbox, rotation='vertical')
        
    #         (r, std), rmsd = TD.find_TaylorStatistics(model['u_mag'].loc[spacecraft.data.index], spacecraft.data['u_mag'])
    
    #         #axs[i].annotate(str(r), (0,1), (1, -2), xycoords='axes fraction', textcoords='offset fontsize')
            
    #     filename = 'Ensemble_TimeSeries_{}_{}-{}.png'.format(prediction_target.lower().capitalize(),
    #                                                           prediction_starttime.strftime('%Y%m%dT%H%M%S'),
    #                                                           prediction_stoptime.strftime('%Y%m%dT%H%M%S'))
    #     fig.supxlabel('Date [YYYY-MM]')
    #     fig.supylabel(r'Solar Wind Flow Speed $|u_{SW}| [km/s]$')
    #     fig.savefig(figurefilepath + filename, 
    #                 dpi=300)
    
    #plot_ComparisonResults()
    
    #plot_BothShiftMethodsTD(mtraj.cast_intervals['juno_simulcast'])
    
    def plot_EnsembleTD_DTWMLR():
        fig2 = plt.figure(figsize=mtraj.cast_intervals['juno_simulcast'].plotprops['figsize'])
        plt.subplots_adjust(left=0.35, right=0.9, bottom=0.15, top=0.9)
        fig2, ax2 = TD.init_TaylorDiagram(np.std(spacecraft.data['u_mag']), fig=fig2,
                                          half=True, r_label=r'$\sigma_{u_{SW}}$ [km/s]',
                                          theta_label=r'Pearson $r$')
        
        for i, model_name in enumerate(mtraj.cast_intervals['juno_simulcast'].model_names):
            model = mtraj.cast_intervals['juno_simulcast'].models[model_name]
            
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
            
            axs_new[0].plot(spacecraft.data.index, model['u_mag'].loc[spacecraft.data.index], color=mtraj.cast_intervals['juno_simulcast'].gpp('color',model_name))
            axs_new[0].fill_between(spacecraft.data.index, 
                                    model['u_mag'].loc[spacecraft.data.index] + model['u_mag_pos_unc'].loc[spacecraft.data.index], 
                                    model['u_mag'].loc[spacecraft.data.index] - model['u_mag_neg_unc'].loc[spacecraft.data.index],
                                    alpha = 0.5, color=mtraj.cast_intervals['juno_simulcast'].gpp('color',model_name), zorder=1000)
            
            (r_0, std_0), rmsd_0 = TD.find_TaylorStatistics(model['u_mag'].loc[spacecraft.data.index], spacecraft.data['u_mag'])
            
            ax2.scatter(np.arccos(r_0), std_0, s=36, c=traj0.gpp('color',model_name), 
                        marker=traj0.gpp('marker',model_name), label=model_name+' + DTW',
                        zorder=11)
            
            #  Baseline, compared to unshifted models
            if model_name != 'ensemble':
                model_bl = cast_traj_juno_backup.models[model_name]
                (r_bl, std_bl), rmsd_bl = TD.find_TaylorStatistics(model_bl['u_mag'].loc[spacecraft.data.index], spacecraft.data['u_mag'])
            
                ax2.scatter(np.arccos(r_bl), std_bl, s=36, c='gray', 
                            marker=traj0.gpp('marker',model_name), label=model_name, 
                            zorder=10)
            
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
            
            # axs_new[1].hist(r, color=mtraj.cast_intervals['simulcast'].plotparams.model_colors[model_name])
            # axs_new[1].axvline(r_0, color='black')
            
            # r_sig, std_sig, rmsd_sig = np.std(r), np.std(std), np.std(rmsd)
            # r, std, rmsd = np.mean(r), np.mean(std), np.mean(rmsd)
            
            
            
            
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
            # ax2.set_thetamin(0)
            # ax2.set_thetamax(90)
        breakpoint()
        
        fig2.legend(ncols=1, bbox_to_anchor=[0.0, 0.4, 0.35, 0.5], loc='upper right', mode='expand', markerscale=1.0)
        fig2.savefig(filepaths['figures'] / 'Ensemble_TD_JunoComparison.png', 
                     dpi=300)
        
    # plot_EnsembleTD_DTWMLR()
    
    #plt.show()
        
    plot_TemporalShifts_All_Total(filepaths['figures']/'fig06_AllTimingUncertainties')
    
    # =============================================================================
    #   Fig 09: Example MMESH forecasts
    # =============================================================================
    def plot_MMESHExample(traj, fullfilename=''):
        
        fig, axs = plt.subplots(figsize=(6,4),
                                nrows=len(traj.model_names),
                                sharex=True, sharey=True)
        plt.subplots_adjust(left=0.125, bottom=0.125, right=0.95, top=0.90, 
                            hspace=0.1)
        
        # plt.subplots_adjust(**mtraj.cast_intervals['juno_simulcast'].plotprops['adjustments'])
        
        lines = []
        for i, model_name in enumerate(traj.model_names):
            
            # model_name_for_paper = model_name
            # if model_name_for_paper == 'Tao': 
            #     model_name_for_paper += '+'
            model_name_for_paper = traj.model_sources[model_name]
            #breakpoint()
            
            index = traj.index_ddoy(traj.models[model_name].index)
            
            if 'u_mag_pos_unc' in traj.models[model_name].columns:
                axs[i].fill_between(index, 
                                    traj.models[model_name]['u_mag'] + traj.models[model_name]['u_mag_pos_unc'], 
                                    traj.models[model_name]['u_mag'] - traj.models[model_name]['u_mag_neg_unc'],
                                    alpha = 0.5, color=traj.gpp('color', model_name),
                                    linewidth=0.5)    
            
            l = axs[i].plot(index, 
                            traj.models[model_name]['u_mag'],
                            color=traj.gpp('color', model_name), linewidth=2.0)
                            #label=model_name_for_paper)
            
            lines.append(l[0])
            
            axs[i].annotate('({})'.format(string.ascii_lowercase[i]),
                            (0,1), (1,-1), ha='center', va='center',
                            xycoords='axes fraction', 
                            textcoords='offset fontsize')
            #   Limit to 6-months so it's legible
            axs[i].set_xlim(index[0], index[0]+365)
            axs[i].xaxis.set_major_locator(MultipleLocator(25))
            axs[i].xaxis.set_minor_locator(MultipleLocator(5))
            
            axs[i].yaxis.set_major_locator(MultipleLocator(200))
            axs[i].yaxis.set_minor_locator(MultipleLocator(20))
            
        axs[0].set_ylim(250,700)
        
        fig.supxlabel('Days since Jan. 1, {}'.format(traj._primary_df.index[0].year))
        fig.supylabel('Solar Wind Flow Speed $u_{mag}$ [km/s]')
        
        handles = lines
        labels = list(traj.model_sources.values())
        
        bbox = axs[0].get_position()
        fig.legend(handles, labels, loc='lower left', 
                    ncols=4, bbox_to_anchor=[bbox.x0, 
                                             bbox.y0+bbox.height+0.0, 
                                             bbox.width, 
                                             0.1], mode='expand')
        
        for suffix in ['.png', '.jpg', '.svg']:
            plt.savefig(str(fullfilename)+suffix, dpi=600, transparent=True)
        plt.show()
        return
    
    plot_MMESHExample(mtraj.cast_intervals['jupiter_hindcast'], filepaths['figures']/'fig09_MMESHExample')
    
    def poster_plot_6(fullfilename):
        param = 'u_mag'  #  what to plot
        
        traj0 = mtraj.cast_intervals['juno_hindcast']
    
        params = {"ytick.color" : "w",
                  "xtick.color" : "w",
                  "axes.labelcolor" : "w",
                  "axes.edgecolor" : "w"}
        plt.rcParams.update(params)
        
        #   Plot as one row
        # mosaic = [[]]
        # for model_ref in list(set(traj0.model_names) - set(['ensemble'])):
        #     mosaic[0].append(model_ref)
        # fig, axd = plt.subplot_mosaic(mosaic, figsize=(12,1.5), sharex=True, sharey=True)
        # plt.subplots_adjust(bottom=0.15, left=0.05625, right=1-0.00625, top=0.95, 
        #                     wspace=0.025)
        
        #   Plot as one column
        mosaic = []
        for model_ref in list(set(traj0.model_names) - set(['ensemble'])):
            mosaic.append([model_ref])
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,3), sharex=True, sharey=True)
        plt.subplots_adjust(bottom=0.16, left=0.120, right=0.9925, top=0.99, 
                            hspace=0.1)
        
        for i, model_ref in enumerate(list(set(traj0.model_names) - set(['ensemble']))):
            
            for axis in ['top','bottom','left','right']:
                axd[model_ref].spines[axis].set_linewidth(2)
            axd[model_ref].tick_params(width=2, which='major')
            axd[model_ref].tick_params(width=1.5, which='minor')
            
            axd[model_ref].plot(traj0.index_ddoy(traj0.data.index), 
                                traj0.data[param],
                                color='black', alpha=1.0, linewidth=1.5,
                                marker='o', markersize=0.01)
                
            axd[model_ref].plot(traj0.index_ddoy(traj0.models[model_ref].index),
                                 traj0.models[model_ref][param],
                                 color=traj0.gpp('color',model_ref), linewidth=2)
                                 #marker=traj0.gpp('marker',model_ref), markersize=0.01)
            
            axd[model_ref].fill_between(traj0.index_ddoy(traj0.models[model_ref].index),
                                        traj0.models[model_ref][param] - traj0.models[model_ref][param+'_neg_unc'],
                                        traj0.models[model_ref][param] + traj0.models[model_ref][param+'_pos_unc'],
                                        color=traj0.gpp('color',model_ref), linewidth=1.5, alpha=0.5)
                                        #marker=traj0.gpp('marker',model_ref), markersize=0.01)
            
            label_bbox = dict(facecolor = traj0.gpp('color',model_ref), 
                              edgecolor = traj0.gpp('color',model_ref), 
                              pad = 0.1, boxstyle = 'round')
            
            axd[model_ref].annotate('{}'.format(traj0.model_sources[model_ref]),
                                    (0,1), (0.5,-0.5), ha='left', va='top',
                                    xycoords='axes fraction', 
                                    textcoords='offset fontsize',
                                    bbox = label_bbox)
        
        axd[model_ref].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
        
        axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
        axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
        
        axd[model_ref].set_ylim((350, 650))
        axd[model_ref].yaxis.set_major_locator(MultipleLocator(100))
        axd[model_ref].yaxis.set_minor_locator(MultipleLocator(25))
        fig.supxlabel('Days Since Jan. 1 {}'.format(traj0._primary_df.index[0].year),
                      fontsize=plt.rcParams['figure.labelsize'], color='w')
        fig.supylabel(r'Flow Speed $u_{mag}$ [km/s]', x=0.00625, 
                      fontsize=plt.rcParams['figure.labelsize'], color='w')
        
        for suffix in ['.png', '.jpg']:
            plt.savefig(str(fullfilename)+suffix, 
                        dpi = 800,
                        transparent = True)
        plt.show()
        
        #  Plot the ensemble separately
        mosaic = [[]]
        for model_ref in ['ensemble']:
            mosaic[0].append(model_ref)
        
        #   We want the plot itself to be ~3.55 across, ~1.2 vertical
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(4,1.5), sharex=True, sharey=True)
        plt.subplots_adjust(bottom=0.15, left=0.12, right=0.9925, top=0.99,
                            hspace=0)
        
        for i, model_ref in enumerate(['ensemble']):
            
            for axis in ['top','bottom','left','right']:
                axd[model_ref].spines[axis].set_linewidth(2)
            axd[model_ref].tick_params(width=2, which='major')
            axd[model_ref].tick_params(width=1.5, which='minor')
            
            axd[model_ref].plot(traj0.index_ddoy(traj0.data.index), 
                                traj0.data[param],
                                color='black', alpha=1.0, linewidth=1.5,
                                marker='o', markersize=0.01,
                                label=traj0.spacecraft_name)
                
            axd[model_ref].plot(traj0.index_ddoy(traj0.models[model_ref].index),
                                 traj0.models[model_ref][param],
                                 color=traj0.gpp('color',model_ref), linewidth=2,
                                 #marker=traj0.gpp('marker',model_ref), markersize=0.01,
                                 label = traj0.model_sources[model_ref])
            
            axd[model_ref].fill_between(traj0.index_ddoy(traj0.models[model_ref].index),
                                        traj0.models[model_ref][param] - traj0.models[model_ref][param+'_neg_unc'],
                                        traj0.models[model_ref][param] + traj0.models[model_ref][param+'_pos_unc'],
                                        color=traj0.gpp('color',model_ref), linewidth=1.5, alpha=0.5,
                                        #marker=traj0.gpp('marker',model_ref), markersize=0.01,
                                        label = traj0.model_sources[model_ref])
            
            label_bbox = dict(facecolor = traj0.gpp('color',model_ref), 
                              edgecolor = traj0.gpp('color',model_ref), 
                              pad = 0.1, boxstyle = 'round')
            axd[model_ref].annotate('{}'.format(traj0.model_sources[model_ref]),
                            (0,1), (0.5,-0.5), ha='left', va='top',
                            xycoords='axes fraction', 
                            textcoords='offset fontsize',
                            bbox = label_bbox)
        
        axd[model_ref].set_xlim(traj0.index_ddoy(traj0.data.index)[0], traj0.index_ddoy(traj0.data.index)[-1])
        
        axd[model_ref].xaxis.set_major_locator(MultipleLocator(10))
        axd[model_ref].xaxis.set_minor_locator(MultipleLocator(2))
        
        # fig.supxlabel('Day of Year {}'.format(traj0._primary_df.index[0].year),
        #               fontsize=plt.rcParams['figure.labelsize'])
        
        axd[model_ref].set_ylim((350, 650))
        axd[model_ref].yaxis.set_major_locator(MultipleLocator(100))
        axd[model_ref].yaxis.set_minor_locator(MultipleLocator(25))
        #fig.supylabel(r'Flow Speed $u_{mag}$ [km/s]', x=0.00625, fontsize=plt.rcParams['figure.labelsize'])
        
        for suffix in ['.png', '.jpg']:
            plt.savefig(str(fullfilename)+'_'+model_ref+suffix, 
                        dpi = 800,
                        transparent = True)
        plt.show()
        
        return
    
    poster_plot_6(filepaths['figures']/'poster06_AdjustedModels')
    
    return mtraj


def write_cast_for_publication(traj, fullfilename):
    
    output = traj._primary_df
    
    output = output.drop('context', axis=1, level=0)
    output = output.drop('mlr_time_delta', axis=1, level=1)
    output = output.drop('mlr_time_delta_sigma', axis=1, level=1)
    
    output = output.reindex(sorted(output.columns), axis=1)
    #output.index.name = 'datetime'
    
    ensemble_output = output.drop(['enlil', 'huxt', 'tao'], axis=1, level=0)
    # ensemble_output = output.drop('HUXt', axis=1, level=0)
    # ensemble_output = output.drop('Tao', axis=1, level=0)
    
    output_fullfilename = fullfilename[:-4] + '_withConstituentModels' + fullfilename[-4:]
    ensemble_fullfilename = fullfilename
    
    header = ['# Multi-Model Ensemble System for the outer Heliosphere (MMESH)',
              '# Output MME for target: Jupiter',
              '# Spanning: 2016/07/04 - 2023/07/04',
              '# Run by: M. J. Rutala, 2024/02/21',
              '# First row (if present) indicates the source of the parameter',
              '# (whether consituent model or the ensemble itself)',
              '# Second row indicates the parameter',
              '# Units are as follows:',
              '# B_mag (magnitude of the IMF): nT',
              '# n_tot (total number density of solar wind, assumed to be all protons): cm^-3',
              '# p_dyn (solar wind dynamic pressure): nPa',
              '# u_mag (magnitude of the solar wind flow speed): km/s',
              '###################################################################################']
    header = [line + ' \n' for line in header]
    
    with open(output_fullfilename, 'w') as f:
        f.writelines(header)
        output.to_csv(f, float_format='%.4f')
        
    with open(ensemble_fullfilename, 'w') as f:
        f.writelines(header)
        ensemble_output.to_csv(f, float_format='%.4f')