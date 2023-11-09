#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:38:04 2023

@author: mrutala
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd

#   To set up file
SolarRadioFlux_filepath = '/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv'

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

"""
How I imagine this framework to work:
    
    The SolarWindEM class is loaded with spacecraft data and corresponding models
        (The input data is assumed to be trimmed appropriately, i.e. only solar wind data)
    The model then iterates over each spacecraft and model combination:
        First, baselining the input relationship
        Then, applying Dynamic Time Warping to the model timeseries and comparing it to the spacecraft
        Then, optionally display how the delta t changes with time, other statistics, etc.
        ...
        Finally, store the shifted and/or warped model outputs
    Average over the model outputs
        Have option for simple average, or weighting based on some factor
        (Primarily simple average, as this seems to be an agreed upon standard in EM)
        
        Finally, store the ensemble model outputs (after averaging) in a top level dataframe
        Additionally, save the times at which the EM is available (i.e., times when all models are present)
""" 
class MMESH:
    
    def __init__(self, trajectories=[]):
        
        self.spacecraft_names = []
        self.trajectories = {}
        
        possible_models = []
        for trajectory in trajectories:
            self.spacecraft_names.append(trajectory.spacecraft_name)
            self.trajectories[trajectory.spacecraft_name] = trajectory
            possible_models += trajectory.model_names
        
        self.model_names = list(set(possible_models))
        
        
    def linear_regression(self, formula):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import re
        
        # =============================================================================
        #   Concatenate trajectories into a single training set based on model
        #   
        # =============================================================================
        predictions_dict = {}
        for model_name in self.model_names:
            
            d = {'datetime':[], 
                 'empirical_dtime':[], 
                 'solar_radio_flux':[], 
                 'target_sun_earth_lon':[], 
                 'target_sun_earth_lat':[], 
                 'u_mag_model':[]}
            
            for spacecraft_name, trajectory in self.trajectories.items():
                d['empirical_dtime'].extend(trajectory.models[model_name]['empirical_dtime'])
                d['u_mag_model'].extend(trajectory.models[model_name]['u_mag'])
                
                #   Need to write a better getter/setter for 'context'
                d['solar_radio_flux'].extend(trajectory._primary_df[('context', 'solar_radio_flux')])
                d['target_sun_earth_lon'].extend(trajectory._primary_df[('context', 'target_sun_earth_lon')])
                d['target_sun_earth_lat'].extend(trajectory._primary_df[('context', 'target_sun_earth_lat')])
                
                d['datetime'] = trajectory._primary_df.index
            
            training_df = pd.DataFrame.from_dict(d)
            training_df = training_df.set_index('datetime')
            #predictions_dict[model_name] = training_df
            
            #   !!! Might need to allow different formulae for differen models...
            #   i.e. formula = formulae[model_name]
            needed_columns = re.findall(r"[\w']+", formula)
            training_df = training_df.dropna(subset=needed_columns, axis='index')
            
            #   Need to predict for spacecraft for comparison,
            #   then for Jupiter for cross-spacecraft comparison
            # sc_pred = spacecraftdata.SpacecraftData(spacecraft_name)
            # sc_pred.make_timeseries(total_dtimes_inh.index[0]-dt.timedelta(days=30), 
            #                         total_dtimes_inh.index[-1]+dt.timedelta(days=365*5), 
            #                         timedelta=dt.timedelta(hours=1))
            # srf_prediction = solar_radio_flux.reindex(index=sc_pred.data.index)
            # pos_TSE_pred = sc_pred.find_StateToEarth()
            # prediction_df = pd.DataFrame({'f10p7_flux': srf_prediction['adjusted_flux'],
            #                               'TSE_lon': pos_TSE_pred['del_lon'],
            #                               'TSE_lat': pos_TSE_pred['del_lat']}, 
            #                               index=sc_pred.data.index)
            
            mlr_fit = smf.ols(formula = formula, data=training_df).fit()
            print(mlr_fit.summary())
            
            # prediction_test = reg.get_prediction(exog=prediction_df, transform=True)
            # alpha_level = 0.32  #  alpha is 1-CI or 1-sigma_level
            #                     #  alpha = 0.05 gives 2sigma or 95%
            #                     #  alpha = 0.32 gives 1sigma or 68%
            # pred = prediction_test.summary_frame(alpha_level)
            
            nowcast = mlr_fit.get_prediction(exog=training_df, transform=True)
            alpha_level = 0.32  #  alpha is 1-CI or 1-sigma_level
                                #  alpha = 0.05 gives 2sigma or 95%
                                #  alpha = 0.32 gives 1sigma or 68%
            result = nowcast.summary_frame(alpha_level)
            
            predictions_dict[model_name] = mlr_fit
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,2))
            ax.plot(training_df.index, result['mean'], color='C0')
            ax.fill_between(training_df.index, result['obs_ci_lower'], result['obs_ci_upper'], color='C0', alpha=0.5)
            ax.plot(training_df.index, training_df['empirical_dtime'], color='black')
            ax.set_xlabel('Date')
            ax.set_ylabel(r'$\Delta$ Time [hours]')
            ax.annotate(model_name, 
                        (0,1), xytext=(1,-1),
                        xycoords='axes fraction',
                        textcoords='offset fontsize')
            plt.show()
            
        return predictions_dict
            
    #   We want to characterize linear relationships independently and together
    #   Via single-target (for now) multiple linear regression
        
    # #     #   For now: single-target MLR within each model-- treat models fully independently
    # #     lr_dict = {}
    # #     for i, model_name in enumerate(traj0.model_names):
    #         print('For model: {} ----------'.format(model_name))
    #         label = '({}) {}'.format(string.ascii_lowercase[i], model_name)
            
    #         #   For each model, loop over each dataset (i.e. each Trajectory class)
    #         #   for j, trajectory in self.trajectories...
            
    #         
            
            
            
    #         #   Reindex the radio data to the cadence of the model
    #         srf_training = solar_radio_flux.reindex(index=total_dtimes_inh.index)['adjusted_flux']
            
    #         # training_values, target_values = TD.make_NaNFree(solar_radio_flux.to_numpy('float64'), total_dtimes_inh.to_numpy('float64'))
    #         # training_values = training_values.reshape(-1, 1)
    #         # target_values = target_values.reshape(-1, 1)
    #         # reg = LinearRegression().fit(training_values, target_values)
    #         # print(reg.score(training_values, target_values))
    #         # print(reg.coef_)
    #         # print(reg.intercept_)
    #         # print('----------')
            
    #         TSE_lon = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lon']
            
    #         # training_values, target_values = TD.make_NaNFree(TSE_lon.to_numpy('float64'), total_dtimes_inh.to_numpy('float64'))
    #         # training_values = training_values.reshape(-1, 1)
    #         # target_values = target_values.reshape(-1, 1)
    #         # reg = LinearRegression().fit(training_values, target_values)
    #         # print(reg.score(training_values, target_values))
    #         # print(reg.coef_)
    #         # print(reg.intercept_)
    #         # print('----------')
            
    #         TSE_lat = pos_TSE.reindex(index=total_dtimes_inh.index)['del_lat']
            
    #         # training_values, target_values = TD.make_NaNFree(TSE_lat.to_numpy('float64'), total_dtimes_inh.to_numpy('float64'))
    #         # training_values = training_values.reshape(-1, 1)
    #         # target_values = target_values.reshape(-1, 1)
    #         # reg = LinearRegression().fit(training_values, target_values)
    #         # print(reg.score(training_values, target_values))
    #         # print(reg.coef_)
    #         # print(reg.intercept_)
    #         # print('----------')
            
    #         training_df = pd.DataFrame({'total_dtimes': total_dtimes_inh,
    #                                     'f10p7_flux': srf_training,
    #                                     'TSE_lon': TSE_lon,
    #                                     'TSE_lat': TSE_lat}, index=total_dtimes_inh.index)
    #         training_df.dropna(axis='index')
            
    #         #training1, training3, target = TD.make_NaNFree(solar_radio_flux, TSE_lat, total_dtimes_inh.to_numpy('float64'))
    #         #training = np.array([training1, training3]).T
    #         #target = target.reshape(-1, 1)
            
    #         # n = int(1e4)
    #         # mlr_arr = np.zeros((n, 4))
    #         # for sample in range(n):
    #         #     rand_indx = np.random.Generator.integers(0, len(target), len(target))
    #         #     reg = LinearRegression().fit(training[rand_indx,:], target[rand_indx])
    #         #     mlr_arr[sample,:] = np.array([reg.score(training, target), 
    #         #                                   reg.intercept_[0], 
    #         #                                   reg.coef_[0,0], 
    #         #                                   reg.coef_[0,1]])
                
    #         # fig, axs = plt.subplots(nrows = 4)
    #         # axs[0].hist(mlr_arr[:,0], bins=np.arange(0, 1+0.01, 0.01))
    #         # axs[1].hist(mlr_arr[:,1])
    #         # axs[2].hist(mlr_arr[:,2])
    #         # axs[3].hist(mlr_arr[:,3])
            
    #         # lr_dict[model_name] = [np.mean(mlr_arr[:,0]), np.std(mlr_arr[:,0]),
    #         #                        np.mean(mlr_arr[:,1]), np.std(mlr_arr[:,1]),
    #         #                        np.mean(mlr_arr[:,2]), np.std(mlr_arr[:,2]),
    #         #                        np.mean(mlr_arr[:,3]), np.std(mlr_arr[:,3])]
    #         # print(lr_dict[model_name])
    #         # print('------------------------------------------')
            
    #         #print(reg.predict({trai}))
            
    #         # print('Training data is shape: {}'.format(np.shape(sm_training)))
    #         # ols = sm.OLS(target, sm_training)
    #         # ols_result = ols.fit()
    #         # summ = ols_result.summary()
    #         # print(summ)
            
    #         # mlr_df = pd.DataFrame.from_dict(lr_dict, orient='index',
    #         #                                 columns=['r2', 'r2_sigma', 
    #         #                                          'c0', 'c0_sigma',
    #         #                                          'c1', 'c1_sigma', 
    #         #                                          'c2', 'c2_sigma'])
            
    #         fig, ax = plt.subplots()
    #         ax.plot(prediction_df.index, pred['mean'], color='red')
    #         ax.fill_between(prediction_df.index, pred['obs_ci_lower'], pred['obs_ci_upper'], color='red', alpha=0.5)
    #         ax.plot(training_df.index, training_df['total_dtimes'], color='black')
    #         ax.set_ylim((-24*10, 24*10))
    #     return training_df, prediction_df, pred
        
    # # def addData(self, spacecraft_name, spacecraft_df):
        
    # #     self.spacecraft_names.append(spacecraft_name)
        
        
    # #     self.spacecraft_dict[spacecraft_name] = {'spacecraft_data': spacecraft_df,
    # #                                                      'models': {}}
        
    # # def addModel(self, model_name, model_df, spacecraft_name = 'Previous'):
        
    # #     #  Parse spacecraft_name
    # #     if (spacecraft_name == 'Previous') and (len(self.spacecraft_dictionaries) > 0):
    # #         spacecraft_name = self.spacecraft_dictionaries.keys()[-1]
    # #     elif spacecraft_name == 'Previous':
    # #         logging.warning('No spacecraft loaded yet! Either specify a spacecraft, or load spacecraft data first.')
               
    # #     self.spacecraft_dict[spacecraft_name]['models'][model_name] = model_df


    # # def baseline(self):
        
    # #     for key, value in self.spacecraft_dict.items():
    # #         print()
            
            
    # # def warp(self, basis_tag):
        
    # #     for spacecraft_name, d in self.spacecraft_dict.items():
            
    # #         for model_name, model_output in d['models'].items():
                
    # #             print()
                
class Trajectory:
    """
    The Trajectory class is designed to hold trajectory information in 4D
    (x, y, z, t), together with model and spacecraft data along the same path.
    A given trajectory may correspond to 1 spacecraft but any number of models.
    """
    
    def __init__(self):
        
        self.metric_tags = ['u_mag', 'p_dyn', 'n_tot', 'B_mag']
        self.variables =  ['u_mag', 'n_tot', 'p_dyn', 'B_mag']
        
        self.spacecraft_name = ''
        self.spacecraft_df = ''
        
        self.model_names = []
        self.model_dfs = {}
        self.model_shift_stats = {}
        self.model_dtw_stats = {}
        self.model_dtw_times = {}
        
        #self.trajectory = pd.Dataframe(index=index)
        
        self._data = None
        self._primary_df = pd.DataFrame()
        
    @property
    def trajectory(self):
        return self._trajectory
    
    @trajectory.setter
    def trajectory(self, df):
        self._trajectory = df.upper()

    @property
    def data(self):
        return self._primary_df[self.spacecraft_name]
    
    @data.setter
    def data(self, df):
        
        #  Find common columns between the supplied DataFrame and internals
        int_columns = list(set(df.columns).intersection(self.variables))
        df = df[int_columns]
        
        if self._primary_df.empty:
            df.columns = pd.MultiIndex.from_product([[self.spacecraft_name], df.columns])
            self._primary_df = df
        else:
            new_keys = [self._primary_df.columns.levels[0], self.spacecraft_name]
            self._primary_df = pd.concat([self._primary_df, df], axis=1,
                                        keys=new_keys)
    
    def addData(self, spacecraft_name, spacecraft_df):
        self.spacecraft_name = spacecraft_name
        self.data = spacecraft_df
    
    @property
    def models(self):
        return self._primary_df[self.model_names]
    
    @models.setter
    def models(self, df):
        int_columns = list(set(df.columns).intersection(self.variables))
        df = df[int_columns]
        
        if self._primary_df.empty:
            new_keys = [self.model_names[-1]]
        else:

            new_keys = [*self._primary_df.columns.levels[0], self.model_names[-1]]
            df.columns = pd.MultiIndex.from_product([[self.model_names[-1]], df.columns])

        self._primary_df = pd.concat([self._primary_df, df], axis=1)
    
    def addModel(self, model_name, model_df):
        self.model_names.append(model_name)
        self.models = model_df

    def baseline(self):
        import scipy
        
        models_stats = {}
        for model_name in self.model_names:
            
            model_stats = {}
            for m_tag in self.metric_tags:
                
                #  Find overlap between data and this model, ignoring NaNs
                i1 = self.data.dropna(subset=[m_tag]).index
                i2 = self.models[model_name].dropna(subset=[m_tag]).index
                intersection_indx = i1.intersection(i2) 
                
                if len(intersection_indx) > 2:
                    #  Calculate r (corr. coeff.) and both standard deviations
                    r, pvalue = scipy.stats.pearsonr(self.data[m_tag].loc[intersection_indx],
                                                     self.models[model_name][m_tag].loc[intersection_indx])
                    model_stddev = np.std(self.models[model_name][m_tag].loc[intersection_indx])
                    spacecraft_stddev = np.std(self.data[m_tag].loc[intersection_indx])
                else:
                    r = 0
                    model_stddev = 0
                    spacecraft_stddev = 0
                
                model_stats[m_tag] = (spacecraft_stddev, model_stddev, r)
                
            models_stats[model_name] = model_stats
        
        return models_stats
    
    def plot_TaylorDiagram(self, tag_name='', fig=None, ax=None, **plt_kwargs):
        
        import plot_TaylorDiagram as TD

        #  If only _primary_df, then:
        #    plot models with shapes and colors
        #  If _primary_df and _secondary_df, then:
        #    plot _primary_df with shapes in black
        #    plot _seconday_df with shapts and colors
        
        if tag_name == '': tag_name = self._primary_df.columns[0][1]
        
        ref_data = self.data[tag_name].to_numpy(dtype='float64')
        ref_std = np.nanstd(ref_data)
        
        fig, ax = TD.init_TaylorDiagram(ref_std, fig=fig, ax=ax, **plt_kwargs)
        
        #  Plot non-reference data
        for model_name in self.model_names:
            
            #model_nonan, ref_nonan = TD.make_NaNFree(self.models[model_name][tag_name].to_numpy(dtype='float64'), ref_data)

            (r, std), rmsd = TD.find_TaylorStatistics(self.models[model_name][tag_name].to_numpy(dtype='float64'), ref_data)
            
            ax.scatter(np.arccos(r), std, label=model_name,
                       marker=model_symbols[model_name], s=72, c=model_colors[model_name],
                       zorder=10)
        
        return (fig, ax)
    
    def binarize(self, parameter, smooth = 1, sigma = 3, reswidth=0.0):
        from SWData_Analysis import find_Jumps
        
        if type(smooth) != dict:
            smoothing_kernels = dict.fromkeys([self.spacecraft_name, *self.model_names], smooth)
        else:
            smoothing_kernels = smooth
        
        df = find_Jumps(self.data, parameter, sigma, smoothing_kernels[self.spacecraft_name])
        self._primary_df[self.spacecraft_name, 'jumps'] = df['jumps']
        
        for model_name in self.model_names:
            df = find_Jumps(self.models[model_name], parameter, sigma, smoothing_kernels[model_name], resolution_width=reswidth)
            self._primary_df[model_name, 'jumps'] = df['jumps']
        
        self._primary_df.sort_index(axis=1, inplace=True)
        
    def optimize_ForBinarization(self, parameter, smooth_max = 648, threshold=1):
        
        #   Isolate the parameter we're interested in binarizing
        param_df = self._primary_df.xs(parameter, axis=1, level=1)
        param_df = (param_df - param_df.min())/(param_df.max() - param_df.min())
        
        #   Get the smoothed derivative
        def smooth_deriv(series, smooth_window):
            smooth_series = series.rolling(smooth_window, min_periods=1, 
                                           center=True, closed='left').mean()
            smooth_deriv = np.gradient(smooth_series, 
                                       smooth_series.index.values.astype(np.float64))
            return smooth_deriv
        
        param_sigs = {}
        for name, series in param_df.items():
            sd = smooth_deriv(param_df[name], dt.timedelta(hours=1))
            param_sigs[name] = np.nanstd(sd)
       
        ref_std = min(param_sigs.values())
        
        #   Iterate over second-level column labels matching parameter
        result = {}
        for name, series in param_df.items():
            #print(name)
            
            smooth_series = series
            sd  = smooth_deriv(smooth_series, dt.timedelta(hours=1))
            smooth_score = np.nanstd(sd) / ref_std
            
            smooth_window = dt.timedelta(hours=1)
            while smooth_score > threshold:
                
                smooth_window += dt.timedelta(hours=1)
                
                sd = smooth_deriv(series, smooth_window)
                smooth_score = np.nanstd(sd) / ref_std
                
                if smooth_window > dt.timedelta(hours=smooth_max):
                    break
                
            #   Can't smooth to a width of 0; for hourly data, this is no change
            if smooth_window.total_seconds() == 0.: smooth_window = dt.timedelta(hours=1)
            
            result[name] = smooth_window.total_seconds()/3600.
            #print("Smoothing of {} hours yields a standard deviation of {} for {}".format(str(smooth_window), smooth_score, name))
            
        return result
            
    
    def optimize_shifts(self, metric_tag, shifts=[0]):
        """
        Trajectory.optimize_shifts tests various constant temporal shifts of 
        modeled solar wind values to measured data and calculates various 
        statistics to be used in determining the optimal constant shift

        Parameters
        ----------
        basis_tag : TYPE
            DESCRIPTION.
        metric_tag : TYPE
            DESCRIPTION.
        shifts : TYPE, optional
            DESCRIPTION. The default is [0].

        Returns
        -------
        A pandas DataFrame of statistics (columns) vs shifts (rows)

        """
        import plot_TaylorDiagram as TD
        
        for model_name in self.model_names:
            
            stats = []
            time_deltas = []
            for shift in shifts:
                
                shift_model_df = self.models[model_name].copy(deep=True)
                
                shift_model_df.index += dt.timedelta(hours=int(shift))
                
                shift_model_df = pd.concat([self.data, shift_model_df], axis=1,
                                            keys=[self.spacecraft_name, model_name])
                
                model_nonan, ref_nonan = TD.make_NaNFree(shift_model_df[model_name][metric_tag].to_numpy(dtype='float64'), 
                                                         shift_model_df[self.spacecraft_name][metric_tag].to_numpy(dtype='float64'))

                (r, sig), rmsd = TD.find_TaylorStatistics(model_nonan, ref_nonan)
                
                stat = {'shift': [shift],
                        'r': [r],
                        'stddev': [sig],
                        'rmsd': [rmsd]}
                        # 'true_negative': cm[0,0],
                        # 'true_positive': cm[1,1],
                        # 'false_negative': cm[1,0],
                        # 'false_positive': cm[0,1],
                        # 'accuracy': accuracy,
                        # 'precision': precision,
                        # 'recall': recall,
                        # 'f1_score': f1_score}
                
                
                stats.append(pd.DataFrame.from_dict(stat))
            
            stats_df = pd.concat(stats, axis='index', ignore_index=True)
            self.model_shift_stats[model_name] = stats_df
            
        return self.model_shift_stats
    
    def find_WarpStatistics(self, basis_tag, metric_tag, shifts=[0], intermediate_plots=True):
        import sys
        import DTW_Application as dtwa
        import SWData_Analysis as swda
        
        #shifts = [0]  #  Needs to be input
        sigma_cutoff = 3  #  Needs to be input
        smoothing_time_spacecraft = 2.0  #  Needs to be input
        smoothing_time_model = 2.0  #  Needs to be input
        
        resolution_width_spacecraft = 0.0  #  Input?
        resolution_width_model = 0.0  #  Input?
        
        # self.spacecraft_df = swda.find_Jumps(self.spacecraft_df, 'u_mag', 
        #                                      sigma_cutoff, 
        #                                      smoothing_time_spacecraft, 
        #                                      resolution_width=resolution_width_spacecraft)
        
        for model_name in self.model_names:
            
        #     model_df = swda.find_Jumps(model_df, 'u_mag', 
        #                                sigma_cutoff, 
        #                                smoothing_time_model, 
        #                                resolution_width=resolution_width_model)
            
            stats = []
            time_deltas = []
            for shift in shifts:
                stat, time_delta = dtwa.find_SolarWindDTW(self.data, self.models[model_name], shift, 
                                                          basis_tag, metric_tag, 
                                                          total_slope_limit=96.0,
                                                          intermediate_plots=intermediate_plots,
                                                          model_name = model_name,
                                                          spacecraft_name = self.spacecraft_name)
                time_delta = time_delta.rename(columns={'time_lag': str(shift)})
                
                stats.append(pd.DataFrame.from_dict(stat))
                time_deltas.append(time_delta)
            
            stats_df = pd.concat(stats, axis='index', ignore_index=True)
            times_df = pd.concat(time_deltas, axis='columns')
            self.model_dtw_stats[model_name] = stats_df
            self.model_dtw_times[model_name] = times_df
            
        return self.model_dtw_stats
    
    def optimize_Warp(self, eqn):
        
        best_shifts = {}
        for model_name in self.model_names:
            
            form = eqn(self.model_dtw_stats[model_name])
            
            best_shift_indx = np.argmax(form)
            #shift = dtw_stats[model_name].iloc[best_shift_indx][['shift']]
            best_shifts[model_name] = self.model_dtw_stats[model_name].iloc[best_shift_indx]
            
        self.best_shifts = best_shifts
        self._dtw_optimization_equation = eqn
    
    def ensemble(self, weights = None):
        """
        Produces an ensemble model from inputs.
        
        The stickiest part of this is considering the weights:
            Potentially, each model, parameters, and timestep could have a unique weight
            Model Weights: account for general relative performance of models
            Parameters Weights: account for better performance of individual parameters within a model
                i.e., Model 1 velocity may outperform Model 2, but Model 2 pressure may outperform Model 1
            Timestep Weights: Certain models may relatively outperform others at specific times. 
                Also allows inclusion of partial models, i.e., HUXt velocity but with 0 weight in pressure

        Parameters
        ----------
        weights : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        import pandas as pd
        from functools import reduce
        
        #weights_df = pd.DataFrame(columns = self.em_parameters)
        weights = dict.fromkeys(self.variables, 1.0/len(self.model_names))
        
        # if weights == None:
        #     for model in self.model_names:
        #         all_nans = ~self.model_dfs[model].isna().all()
        #         overlap_col = list(set(all_nans.index) & set(weights_df.columns))
        #         weights_df.loc[model] = all_nans[overlap_col].astype(int)
        #         weights_df = weights_df.div(weights_df.sum())
        
        # if dict, use dict
        # if list/array, assign to dict in order
        # if none, make dict with even weights
        
        partials_list = []
        for model_name in self.model_names:
            df = self.models[model_name][self.variables].mul(weights, fill_value=0.)
            partials_list.append(df)
        
        ensemble = reduce(lambda a,b: a.add(b, fill_value=0.), partials_list)
        
        self.addModel('ensemble', ensemble)
        #return ensemble 
    
        # self.ensemble = pd.DataFrame(columns = self.em_parameters)
        
        # for model, model_df in self.model_dfs.items(): print(len(model_df))
        
        # for tag in self.em_parameters:
        #     for model, model_df in self.model_dfs.items():
                
        #         self.ensemble[tag] = weights_df[tag][model] * model_df[tag]
        
        # print(weights_df.div(weights_df.sum()))
        
        
    # =============================================================================
    #   Plotting stuff
    # =============================================================================
    def _plot_parameters(self, parameter):
        #  Check which parameter was specified and set up some plotting keywords
        match parameter:
            case ('u_mag' | 'flow speed'):
                tag = 'u_mag'
                ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'linear', 'ylim': (250, 600),
                           'yticks': np.arange(350,600+50,100)}
            case ('p_dyn' | 'pressure'):
                tag = 'p_dyn'
                ylabel = r'Solar Wind Dynamic Pressure $p_{dyn}$ [nPa]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'log', 'ylim': (1e-3, 2e0),
                           'yticks': 10.**np.arange(-3,0+1,1)}
            case ('n_tot' | 'density'):
                tag = 'n_tot'
                ylabel = r'Solar Wind Ion Density $n_{tot}$ [cm$^{-3}$]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'log', 'ylim': (5e-3, 5e0),
                           'yticks': 10.**np.arange(-2, 0+1, 1)}
            case ('B_mag' | 'magnetic field'):
                tag = 'B_mag'
                ylabel = r'Solar Wind Magnetic Field Magnitude $B_{mag}$ [nT]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'linear', 'ylim': (0, 5),
                           'yticks': np.arange(0, 5+1, 2)}
            case ('jumps'):
                tag = 'jumps'
                ylabel = r'$\sigma$-normalized derivative of $u_{mag}$, binarized at 3'
                plot_kw =  {'ylabel': ylabel,
                            'yscale': 'linear',
                            'ylim': (-0.25, 1.25)}
                
        return tag, plot_kw
        
    def plot_SingleTimeseries(self, parameter, starttime, stoptime, filename=''):
        """
        Plot the time series of a single parameter from a single spacecraft,
        separated into 4/5 panels to show the comparison to each model.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        import PlottingConstants
        
        tag, plot_kw = self._plot_parameters(parameter)
        
        # save_filestem = 'Timeseries_{}_{}_{}-{}'.format(spacecraft_name.replace(' ', ''),
        #                                                 tag,
        #                                                 starttime.strftime('%Y%m%d'),
        #                                                 stoptime.strftime('%Y%m%d'))
        
        with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
            fig, axs = plt.subplots(figsize=(8,6), nrows=len(self.model_names), sharex=True, squeeze=False)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.0)
            
            for indx, ax in enumerate(axs.flatten()):
                
                ax.plot(self.data.index, 
                        self.data[tag],
                        color='xkcd:blue grey', alpha=0.6, linewidth=1.5,
                        label='Juno/JADE (Wilson+ 2018)')
                
                ax.set(**plot_kw)
                ax.set_ylabel('')
                #  get_yticks() for log scale doesn't work as expected; workaround:
                yticks = ax.get_yticks()
                yticks = [yt for yt in yticks if ax.get_ylim()[0] <= yt <= ax.get_ylim()[1]]
                if indx != len(axs)-1: 
                    if (yticks[0] in ax.get_ylim()) and yticks[-1] in ax.get_ylim():
                        ax.set(yticks=yticks[1:])
                #ax.grid(linestyle='-', linewidth=2, alpha=0.1, color='xkcd:black')
            print('Plotting models...')  
            for ax, model_name in zip(axs.flatten(), self.model_names):
                if tag in self.models[model_name].columns:
                    ax.plot(self.models[model_name].index, self.models[model_name][tag], 
                            color=model_colors[model_name], label=model_name, linewidth=2)
                
                model_label_box = dict(facecolor=model_colors[model_name], 
                                       edgecolor=model_colors[model_name], 
                                       pad=0.1, boxstyle='round')
               
                ax.text(0.01, 0.95, model_name, color='black',
                        horizontalalignment='left', verticalalignment='top',
                        bbox = model_label_box,
                        transform = ax.transAxes)
            
            fig.text(0.5, 0.025, 'Day of Year {:.0f}'.format(starttime.year), ha='center', va='center')
            fig.text(0.025, 0.5, plot_kw['ylabel'], ha='center', va='center', rotation='vertical')
            
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
            # for suffix in ['.png', '.pdf']:
            #     plt.savefig('figures/' + save_filestem, dpi=300, bbox_inches='tight')
            plt.show()
            #return models
            
    def plot_ConstantTimeShifting_Optimization(self):
        import matplotlib.pyplot as plt
        
        import string
        
        fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6, 4.5))
        plt.subplots_adjust(hspace=0.1)
            
        for i, (model_name, ax) in enumerate(zip(self.model_names, axs)):
            
            ax.plot(self.model_shift_stats[model_name]['shift'], 
                    self.model_shift_stats[model_name]['r'],
                    color=model_colors[model_name])
            
            #!!!!  Need to allow custom optimization formulae
            best_shift_indx = np.argmax(self.model_shift_stats[model_name]['r'])
            
            
            shift, r, sig, rmsd = self.model_shift_stats[model_name].iloc[best_shift_indx][['shift', 'r', 'stddev', 'rmsd']]
            
            ax.axvline(shift, color='black', linestyle='--', linewidth=1)
            
            ax.annotate('({:.0f}, {:.3f})'.format(shift, r), 
                        (shift, r), xytext=(0.5, 0.5),
                        textcoords='offset fontsize')
            
            label = '({}) {}'.format(string.ascii_lowercase[i], model_name)
            
            ax.annotate(label, (0,1), xytext=(0.5, -0.5),
                        xycoords='axes fraction', textcoords='offset fontsize',
                        ha='left', va='top')
            
            ax.set_ylim(-0.4, 0.6)
            ax.set_xlim(-102, 102)
            ax.set_xticks(np.arange(-96, 96+24, 24))
            
        fig.supxlabel('Temporal Shift [hours]')
        fig.supylabel('Correlation Coefficient (r)')
        
        return fig, axs
    
    def plot_ConstantTimeShifting_TD(self, fig=None):
        import matplotlib.pyplot as plt
        import plot_TaylorDiagram as TD
        
        fig, ax = TD.init_TaylorDiagram(np.nanstd(self.data['u_mag']), fig=fig)
        
        for model_name in self.model_names:
            
            ref = self.data['u_mag'].to_numpy(dtype='float64')
            tst = self.models[model_name]['u_mag'].to_numpy(dtype='float64')  
            
            fig, ax = TD.plot_TaylorDiagram(tst, ref, fig=fig, ax=ax, zorder=9, 
                                            s=36, c='black', marker=model_symbols[model_name],
                                            label=model_name)
            
            # fig, ax = TD.plot_TaylorDiagram(tst, ref, fig=fig, ax=ax, zorder=10, 
            #                                 s=36, c=model_colors[model_name], marker=model_symbols[model_name],
            #                                 label=model_name + ' ' + str(shift) + ' h')
            # (r, sig), rmsd = TD.find_TaylorStatistics(tst, ref)
            # ax.scatter(np.arccos(r), sig, zorder=9, 
            #            s=36, c='black', marker=model_symbols[model_name],
            #            label=model_name)
            
            best_shift_indx = np.argmax(self.model_shift_stats[model_name]['r'])
            shift = int((self.model_shift_stats[model_name].iloc[best_shift_indx])['shift'])
            
            shift_model_df = self.models[model_name].copy(deep=True)
            shift_model_df.index += dt.timedelta(hours=shift)
            shift_model_df = pd.concat([self.data, shift_model_df], axis=1,
                                        keys=[self.spacecraft_name, model_name])
            
            (r, sig), rmsd = TD.find_TaylorStatistics(shift_model_df[(model_name, 'u_mag')], 
                                                           shift_model_df[(self.spacecraft_name, 'u_mag')]) #self.model_shift_stats[model_name].iloc[best_shift_indx][['shift', 'r', 'stddev', 'rmsd']]
            
            ax.scatter(np.arccos(r), sig, zorder=10, 
                        s=36, c=model_colors[model_name], marker=model_symbols[model_name],
                        label=model_name + ' ' + str(shift) + ' h')
        
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        
    def plot_DynamicTimeWarping_Optimization(self, fig=None):
        import matplotlib.pyplot as plt
        import string
        
        if fig == None:
            fig = plt.figure(figsize=[6,4.5])
            
        gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1,1,1],
                              left=0.1, bottom=0.1, right=0.95, top=0.95,
                              wspace=0.0, hspace=0.1)
        
        axs0 = [fig.add_subplot(gs[y,0]) for y in range(3)]
        axs1 = [fig.add_subplot(gs[y,2]) for y in range(3)]
        
        for i, (model_name, ax0, ax1) in enumerate(zip(self.model_names, axs0, axs1)):
            
            #   Get some statistics for easier plotting
            shift, r, sig, width = self.best_shifts[model_name][['shift', 'r', 'stddev', 'width_68']]
            label1 = '({}) {}'.format(string.ascii_lowercase[2*i], model_name)
            label2 = '({}) {}'.format(string.ascii_lowercase[2*i+1], model_name)
            
            #   Get copies of first ax1 for plotting 3 different parameters
            ax0_correl = ax0.twinx()
            ax0_width = ax0.twinx()
            ax0_width.spines.right.set_position(("axes", 1.35))
            
            #   Plot the optimization function, and its components (shift and dist. width)
            #   Plot half the 68% width, or the quasi-1-sigma value, in hours
            ax0.plot(self.model_dtw_stats[model_name]['shift'], self._dtw_optimization_equation(self.model_dtw_stats[model_name]),
                     color=model_colors[model_name], zorder=10)
            ax0_correl.plot(self.model_dtw_stats[model_name]['shift'], self.model_dtw_stats[model_name]['r'],
                            color='C1', zorder=1)
            ax0_width.plot(self.model_dtw_stats[model_name]['shift'], (self.model_dtw_stats[model_name]['width_68']/2.),
                           color='C3', zorder=1)

            #   Mark the maximum of the optimization function
            ax0.axvline(shift, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax0.annotate('({:.0f}, {:.3f}, {:.1f})'.format(shift, r, width/2.), 
                        (shift, r), xytext=(0.5, 0.5),
                        textcoords='offset fontsize', zorder=10)
            
            ax0.annotate(label1, (0,1), xytext=(0.5, -0.5),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               ha='left', va='top')
            
            dtw_opt_shifts = self.model_dtw_times[model_name]['{:.0f}'.format(shift)]
            histo = np.histogram(dtw_opt_shifts, range=[-96,96], bins=int(192/6))
            
            ax1.stairs(histo[0]/np.sum(histo[0]), histo[1],
                          color=model_colors[model_name], linewidth=2)
            ax1.axvline(np.median(dtw_opt_shifts), linewidth=1, alpha=0.8, label=r'P_{50}', color='black')
            ax1.axvline(np.percentile(dtw_opt_shifts, 16), linestyle=':', linewidth=1, alpha=0.8, label=r'P_{16}', color='black')
            ax1.axvline(np.percentile(dtw_opt_shifts, 84), linestyle=':', linewidth=1, alpha=0.8, label=r'P_{84}', color='black')
            ax1.annotate(label2, (0,1), (0.5, -0.5), 
                            xycoords='axes fraction', textcoords='offset fontsize',
                            ha='left', va='top')
        
            ax0.set(ylim = (0.0, 1.1), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax0_correl.set_ylim(0.0, 1.0)
            ax0_width.set_ylim(0.0, 72)
            ax0.set_xlim(-102, 102)
            
            ax1.set_ylim(0.0, 0.5)
            ax1.set_xlim(-102, 102)
            
            if i < len(self.model_names)-1:
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
        
        
    def plot_DynamicTimeWarping_TD(self, fig=None):
        import matplotlib.pyplot as plt
        import plot_TaylorDiagram as TD
        
        #   There's probably a much neater way to do this
        #   But we need to drop all NaNs in the data + model sets
        #   Rather than just the NaNs in the data
        #   Since we drop all the NaNs when calculating the other TD params
        subset = []
        for first in [self.spacecraft_name] + self.model_names:
            subset.append((first, 'u_mag'))
        ref_std = np.std(self._primary_df.dropna(subset=subset)[self.spacecraft_name, 'u_mag'])
        
        if fig == None:
            fig = plt.figure(figsize=(6,4.5))
        print(ref_std)
        fig, ax = TD.init_TaylorDiagram(ref_std, fig=fig)

        for model_name in self.model_names:
            (r, sig), rmse = TD.find_TaylorStatistics(self.models[model_name]['u_mag'].to_numpy('float64'), 
                                                      self.data['u_mag'].to_numpy('float64'))
            ax.scatter(np.arccos(r), sig, 
                        marker=model_symbols[model_name], s=24, c='black',
                        zorder=9,
                        label=model_name)
            
            #best_shift_indx = np.argmax(dtw_stats[model_name]['r'] * 6/(dtw_stats[model_name]['width_68']))
            # best_shift_indx = np.where(traj0.model_dtw_stats[model_name]['shift'] == best_shifts[model_name])[0]
            # r, sig = traj0.model_dtw_stats[model_name].iloc[best_shift_indx][['r', 'stddev']].values.flatten()
            r, sig = self.best_shifts[model_name][['r', 'stddev']].values.flatten()
            ax.scatter(np.arccos(r), sig, 
                        marker=model_symbols[model_name], s=48, c=model_colors[model_name],
                        zorder=10,
                        label='{} + DTW'.format(model_name))
            
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        ax.set_axisbelow(True)


def read_SolarRadioFlux(startdate, stopdate, sample_rate='60Min'):
    
    # =============================================================================
    # F10.7 Radio flux from the sun
    # !!!! This needs a reader so this can be one line
    # =============================================================================
    column_headers = ('date', 'observed_flux', 'adjusted_flux',)
    srf_df = pd.read_csv(SolarRadioFlux_filepath,
                         header = 0, names = column_headers)
    srf_df['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                for d in srf_df['date']]
    srf_df = srf_df.set_index('date')
    srf_df = srf_df.resample(sample_rate, origin='start_day').mean().interpolate(method='linear')

    result = srf_df[(srf_df.index >= startdate) & (srf_df.index <= stopdate)]
    
    return result