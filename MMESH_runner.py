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


def MMESH_run():
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
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig = plt.figure(figsize=[6,4.5])
        traj0.plot_ConstantTimeShifting_TD(fig=fig)
        
    
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    for suffix in ['.png']:
        plt.savefig('figures/' + 'test_TD', dpi=300, bbox_inches=extent)
    
    return
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
                        'Juno':     6} # 'Ulysses':  12}   #  hours
    traj0.binarize('u_mag', smooth = smoothing_widths, sigma=3)
    
    traj0.plot_SingleTimeseries('u_mag', starttime, stoptime)
    traj0.plot_SingleTimeseries('jumps', starttime, stoptime)
    
    dtw_stats = traj0.optimize_warp('jumps', 'u_mag', shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)
    
    best_shifts = {}
    for i, model_name in enumerate(traj0.model_names):
        #  Optimize
        best_shift_indx = np.argmax(dtw_stats[model_name]['r'] * 6/(dtw_stats[model_name]['width_68']))
        #shift = dtw_stats[model_name].iloc[best_shift_indx][['shift']]
        best_shifts[model_name] = dtw_stats[model_name].iloc[best_shift_indx]
    
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
    #with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        
    def plot_DTWOptimization():
        fig = plt.figure(figsize=[6,4.5])
        gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1,1,1],
                              left=0.1, bottom=0.1, right=0.95, top=0.95,
                              wspace=0.0, hspace=0.1)
        axs0 = [fig.add_subplot(gs[y,0]) for y in range(3)]
        axs1 = [fig.add_subplot(gs[y,2]) for y in range(3)]
        
        for i, (model_name, ax0, ax1) in enumerate(zip(traj0.model_names, axs0, axs1)):
            
            #   Get some statistics for easier plotting
            shift, r, sig, width = best_shifts[model_name][['shift', 'r', 'stddev', 'width_68']]
            label1 = '({}) {}'.format(string.ascii_lowercase[2*i], model_name)
            label2 = '({}) {}'.format(string.ascii_lowercase[2*i+1], model_name)
            
            #   Get copies of first ax1 for plotting 3 different parameters
            ax0_correl = ax0.twinx()
            ax0_width = ax0.twinx()
            ax0_width.spines.right.set_position(("axes", 1.35))
            
            #   Plot the optimization function, and its components (shift and dist. width)
            #   Plot half the 68% width, or the quasi-1-sigma value, in hours
            ax0.plot(dtw_stats[model_name]['shift'], dtw_stats[model_name]['r'] * 6/(dtw_stats[model_name]['width_68']),
                     color=model_colors[model_name])
            ax0_correl.plot(dtw_stats[model_name]['shift'], dtw_stats[model_name]['r'],
                            color='C1')
            ax0_width.plot(dtw_stats[model_name]['shift'], (dtw_stats[model_name]['width_68']/2.),
                           color='C3')

            #   Mark the maximum of the optimization function
            ax0.axvline(shift, color='C0', linestyle='--', linewidth=1)
            ax0.annotate('({:.0f}, {:.3f}, {:.1f})'.format(shift, r, width/2.), 
                        (shift, r), xytext=(0.5, 0.5),
                        textcoords='offset fontsize')
            
            ax0.annotate(label1, (0,1), xytext=(0.5, -0.5),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               ha='left', va='top')
            
            dtw_opt_shifts = traj0.model_dtw_times[model_name]['{:.0f}'.format(shift)]
            histo = np.histogram(dtw_opt_shifts, range=[-96,96], bins=int(192/6))
            
            ax1.stairs(histo[0]/np.sum(histo[0]), histo[1],
                          color=model_colors[model_name], linewidth=2)
            ax1.axvline(np.median(dtw_opt_shifts), linewidth=1, alpha=0.5, label=r'P_{50}')
            ax1.axvline(np.percentile(dtw_opt_shifts, 16), linestyle=':', linewidth=1, alpha=0.5, label=r'P_{16}')
            ax1.axvline(np.percentile(dtw_opt_shifts, 84), linestyle=':', linewidth=1, alpha=0.5, label=r'P_{84}')
            ax1.annotate(label2, (0,1), (0.5, -0.5), 
                            xycoords='axes fraction', textcoords='offset fontsize',
                            ha='left', va='top')
        
            ax0.set_ylim(0.0, 0.2)
            ax0_correl.set_ylim(0.0, 1.0)
            ax0_width.set_ylim(0.0, 72)
            ax0.set_xlim(-102, 102)
            
            ax1.set_ylim(0.0, 0.5)
            ax1.set_xlim(-102, 102)
            
            if i < len(traj0.model_names)-1:
                ax0.set_xticks(np.arange(-96, 96+24, 24), labels=['']*9)
                ax1.set_xticks(np.arange(-96, 96+24, 24), labels=['']*9)
            else:
                ax0.set_xticks(np.arange(-96, 96+24, 24))
                ax1.set_xticks(np.arange(-96, 96+24, 24))

        #   Plotting coordinates
        ymid = 0.50*(axs0[0].get_position().y1 - axs0[-1].get_position().y0) + axs0[-1].get_position().y0
        ybot = 0.25*axs0[-1].get_position().y0
        
        xleft = 0.25*axs0[0].get_position().x0
        xmid1 = axs0[0].get_position().x1 + 0.25*axs0[0].get_position().size[0]
        xmid2 = axs0[0].get_position().x1 + 0.55*axs0[0].get_position().size[0]
        xmid3 = axs1[0].get_position().x0 - 0.2*axs1[0].get_position().size[0]
        xbot1 = axs0[0].get_position().x0 + 0.5*axs0[0].get_position().size[0]
        xbot2 = axs1[0].get_position().x0 + 0.5*axs1[0].get_position().size[0]
        
        fig.text(xbot1, ybot, 'Constant Temporal Offset [hours]',
                 ha='center', va='center', 
                 fontsize=plt.rcParams["figure.labelsize"])
        fig.text(xleft, ymid, 'Optimization Function [arb.]',
                 ha='center', va='center', rotation='vertical',
                 fontsize=plt.rcParams["figure.labelsize"])
        
        fig.text(xmid1, ymid, 'Correlation Coefficient (r)', 
                 ha='center', va='center', rotation='vertical', fontsize=plt.rcParams["figure.labelsize"],
                 bbox = dict(facecolor='C1', edgecolor='C1', pad=0.1, boxstyle='round'))
        fig.text(xmid2, ymid, 'Distribution Half-Width (34%) [hours]', 
                 ha='center', va='center', rotation='vertical', fontsize=plt.rcParams["figure.labelsize"],
                 bbox = dict(facecolor='C3', edgecolor='C3', pad=0.1, boxstyle='round'))

        fig.text(xbot2, ybot, 'Dynamic Temporal Offset [hours]',
                 ha='center', va='center', 
                 fontsize=plt.rcParams["figure.labelsize"])
        fig.text(xmid3, ymid, 'Fractional Numbers',
                 ha='center', va='center', rotation='vertical', 
                 fontsize=plt.rcParams["figure.labelsize"])

        plt.show()
        
    plot_DTWOptimization()
            
    ref_std = np.nanstd(traj0.data['u_mag'])
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig = plt.figure(figsize=(6,4.5))
        fig, ax = TD.init_TaylorDiagram(ref_std, fig=fig)

    for model_name in traj0.model_names:
        (r, std), rmse = TD.find_TaylorStatistics(traj0.models[model_name]['u_mag'].to_numpy('float64'), 
                                                  traj0.data['u_mag'].to_numpy('float64'))
        ax.scatter(np.arccos(r), std, 
                   marker=model_symbols[model_name], s=24, c='black',
                   zorder=9,
                   label=model_name)
        
        #best_shift_indx = np.argmax(dtw_stats[model_name]['r'] * 6/(dtw_stats[model_name]['width_68']))
        # best_shift_indx = np.where(traj0.model_dtw_stats[model_name]['shift'] == best_shifts[model_name])[0]
        # r, sig = traj0.model_dtw_stats[model_name].iloc[best_shift_indx][['r', 'stddev']].values.flatten()
        r, sig = best_shifts[model_name][['r', 'stddev']].values.flatten()
        ax.scatter(np.arccos(r), sig, 
                   marker=model_symbols[model_name], s=48, c=model_colors[model_name],
                   zorder=10,
                   label='{} + DTW'.format(model_name))
        
    ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
    ax.set_axisbelow(True)
    
    
    # =============================================================================
    #    Compare output list of temporal shifts (and, maybe, running standard deviation or similar)
    #    to physical parameters: 
    #       -  T-S-E angle
    #       -  Solar cycle phase
    #       -  Running Mean SW Speed
    #   This should **ALL** ultimately go into the "Ensemble" class 
    # =============================================================================
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    #formula = "total_dtimes ~ f10p7_flux + TSE_lat"  #  This can be input
    formula = "total_dtimes ~ f10p7_flux + TSE_lat + TSE_lon" 
    
    #   Do the work first, plot second
    #   We want to characterize linear relationships independently and together
    #   Via single-target (for now) multiple linear regression
    
    #   For now: single-target MLR within each model-- treat models fully independently
    lr_dict = {}
    for i, model_name in enumerate(traj0.model_names):
        print('For model: {} ----------'.format(model_name))
        label = '({}) {}'.format(string.ascii_lowercase[i], model_name)
        
        #   For each model, loop over each dataset (i.e. each Trajectory class)
        #   for j, trajectory in self.trajectories...
        
        offset = best_shifts[model_name]['shift']
        dtimes = traj0.model_dtw_times[model_name][str(int(offset))]
        total_dtimes_inh = (offset + dtimes)
        
        # =============================================================================
        # F10.7 Radio flux from the sun
        # !!!! This needs a reader so this can be one line
        # =============================================================================
        column_headers = ('date', 'observed_flux', 'adjusted_flux',)
        solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                       header = 0, names = column_headers)
        solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                    for d in solar_radio_flux['date']]
        solar_radio_flux = solar_radio_flux.set_index('date')
        solar_radio_flux = solar_radio_flux.resample("60Min", origin='start_day').mean().interpolate(method='linear')
        
        #   Reindex the radio data to the cadence of the model
        solar_radio_flux = solar_radio_flux.reindex(index=total_dtimes_inh.index)['adjusted_flux']
        
        # training_values, target_values = TD.make_NaNFree(solar_radio_flux.to_numpy('float64'), total_dtimes_inh.to_numpy('float64'))
        # training_values = training_values.reshape(-1, 1)
        # target_values = target_values.reshape(-1, 1)
        # reg = LinearRegression().fit(training_values, target_values)
        # print(reg.score(training_values, target_values))
        # print(reg.coef_)
        # print(reg.intercept_)
        # print('----------')
        
        TSE_lon = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lon']
        
        # training_values, target_values = TD.make_NaNFree(TSE_lon.to_numpy('float64'), total_dtimes_inh.to_numpy('float64'))
        # training_values = training_values.reshape(-1, 1)
        # target_values = target_values.reshape(-1, 1)
        # reg = LinearRegression().fit(training_values, target_values)
        # print(reg.score(training_values, target_values))
        # print(reg.coef_)
        # print(reg.intercept_)
        # print('----------')
        
        TSE_lat = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lat']
        
        # training_values, target_values = TD.make_NaNFree(TSE_lat.to_numpy('float64'), total_dtimes_inh.to_numpy('float64'))
        # training_values = training_values.reshape(-1, 1)
        # target_values = target_values.reshape(-1, 1)
        # reg = LinearRegression().fit(training_values, target_values)
        # print(reg.score(training_values, target_values))
        # print(reg.coef_)
        # print(reg.intercept_)
        # print('----------')
        
        training_df = pd.DataFrame({'total_dtimes': total_dtimes_inh,
                                    'f10p7_flux': solar_radio_flux,
                                    'TSE_lon': TSE_lon,
                                    'TSE_lat': TSE_lat}, index=total_dtimes_inh.index)
        
        training_df.dropna(axis='index')
        #training1, training3, target = TD.make_NaNFree(solar_radio_flux, TSE_lat, total_dtimes_inh.to_numpy('float64'))
        #training = np.array([training1, training3]).T
        #target = target.reshape(-1, 1)
        
        # n = int(1e4)
        # mlr_arr = np.zeros((n, 4))
        # for sample in range(n):
        #     rand_indx = np.random.Generator.integers(0, len(target), len(target))
        #     reg = LinearRegression().fit(training[rand_indx,:], target[rand_indx])
        #     mlr_arr[sample,:] = np.array([reg.score(training, target), 
        #                                   reg.intercept_[0], 
        #                                   reg.coef_[0,0], 
        #                                   reg.coef_[0,1]])
            
        # fig, axs = plt.subplots(nrows = 4)
        # axs[0].hist(mlr_arr[:,0], bins=np.arange(0, 1+0.01, 0.01))
        # axs[1].hist(mlr_arr[:,1])
        # axs[2].hist(mlr_arr[:,2])
        # axs[3].hist(mlr_arr[:,3])
        
        # lr_dict[model_name] = [np.mean(mlr_arr[:,0]), np.std(mlr_arr[:,0]),
        #                        np.mean(mlr_arr[:,1]), np.std(mlr_arr[:,1]),
        #                        np.mean(mlr_arr[:,2]), np.std(mlr_arr[:,2]),
        #                        np.mean(mlr_arr[:,3]), np.std(mlr_arr[:,3])]
        # print(lr_dict[model_name])
        # print('------------------------------------------')
        
        
        reg = smf.ols(formula = formula, data=training_df).fit()
        print(reg.summary())
        
        prediction_test = reg.get_prediction(exog=training_df, transform=True)
        alpha_level = 0.32  #  alpha is 1-CI or 1-sigma_level
                            #  alpha = 0.05 gives 2sigma or 95%
                            #  alpha = 0.32 gives 1sigma or 68%
        pred = prediction_test.summary_frame(alpha_level)
        
        #print(reg.predict({trai}))
        
        # print('Training data is shape: {}'.format(np.shape(sm_training)))
        # ols = sm.OLS(target, sm_training)
        # ols_result = ols.fit()
        # summ = ols_result.summary()
        # print(summ)
        
        # mlr_df = pd.DataFrame.from_dict(lr_dict, orient='index',
        #                                 columns=['r2', 'r2_sigma', 
        #                                          'c0', 'c0_sigma',
        #                                          'c1', 'c1_sigma', 
        #                                          'c2', 'c2_sigma'])
    return pred
    
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