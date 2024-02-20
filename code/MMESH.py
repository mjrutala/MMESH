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

import plot_TaylorDiagram as TD

from matplotlib.ticker import MultipleLocator
import string

# spacecraft_colors = {'Pioneer 10': '#F6E680',
#                      'Pioneer 11': '#FFD485',
#                      'Voyager 1' : '#FEC286',
#                      'Voyager 2' : '#FEAC86',
#                      'Ulysses'   : '#E6877E',
#                      'Juno'      : '#D8696D'}
# model_colors = {'Tao'    : '#C59FE5',
#                 'HUXt'   : '#A0A5E4',
#                 'SWMF-OH': '#98DBEB',
#                 'MSWIM2D': '#A9DBCA',
#                 'ENLIL'  : '#DCED96',
#                 'ensemble': '#55FFFF'}
# model_symbols = {'Tao'    : 'v',
#                 'HUXt'   : '*',
#                 'SWMF-OH': '^',
#                 'MSWIM2D': 'd',
#                 'ENLIL'  : 'h',
#                 'ensemble': 'o'}

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
class MultiTrajectory:
    
    def __init__(self, trajectories=[]):
        
        self.spacecraft_names = []
        self.trajectories = {}   #  dict of Instances of Trajectory class
        self.cast_intervals = {}    #  ???? of Instances of Trajectory class
                                    #   i.e. hindcast, nowcast, forecase, simulcast
                                    #   all relative to data span (although cound be between data spans)
        
        possible_models = []
        for trajectory in trajectories:
            self.spacecraft_names.append(trajectory.spacecraft_name)
            self.trajectories[trajectory.trajectory_name] = trajectory
            possible_models += trajectory.model_names
        
        self.model_names = list(set(possible_models))
        self.model_names.sort()
        
        self.offset_times = []  #   offset_times + (constant_offset) + np.linspace(0, total_hours, n_steps) give interpolation index
        self.optimized_shifts = []
        
        
    def linear_regression(self, formula):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import re
        
        # =============================================================================
        #   Concatenate trajectories into a single training set based on model
        #   
        # =============================================================================
        predictions_dict = {}
        nowcast_dict = {}
        for model_name in self.model_names:
            
            #   !!! This should be dynamic based on input equation
            d = {'datetime':[], 
                 'empirical_time_delta':[], 
                 'solar_radio_flux':[], 
                 'target_sun_earth_lon':[], 
                 'target_sun_earth_lat':[], 
                 'u_mag':[]}
            
            for trajectory_name, trajectory in self.trajectories.items():
                d['empirical_time_delta'].extend(trajectory.models[model_name]['empirical_time_delta'].loc[trajectory.data_index])
                d['u_mag'].extend(trajectory.models[model_name]['u_mag'].loc[trajectory.data_index])
                
                #   Need to write a better getter/setter for 'context'
                d['solar_radio_flux'].extend(trajectory._primary_df[('context', 'solar_radio_flux')].loc[trajectory.data_index])
                d['target_sun_earth_lon'].extend(trajectory._primary_df[('context', 'target_sun_earth_lon')].loc[trajectory.data_index])
                d['target_sun_earth_lat'].extend(trajectory._primary_df[('context', 'target_sun_earth_lat')].loc[trajectory.data_index])
                
                d['datetime'].extend(trajectory._primary_df.loc[trajectory.data_index].index)
            
            training_df = pd.DataFrame.from_dict(d)
            training_df = training_df.set_index('datetime')
            
            #   Set the input formula internally, then separate into individual
            #   terms by splitting on plus signs. Drop terms from the training
            #   DataFrame if they have null values for any equation terms
            
            #   !!! Might need to allow different formulae for different models...
            #   !!! i.e. formula = formulae[model_name]
            #   !!! Also, should catch all acceptable arithmetic symbols, not just plus
            self.mlr_formula = formula
            self.mlr_formula_terms = re.findall(r"[\w']+", formula)
            training_df = training_df.dropna(subset=self.mlr_formula_terms, axis='index')
            
            mlr_fit = smf.ols(formula = formula, data=training_df).fit()  #!!!! Does this have weights available?
            #   We should weight by correlation coefficient to the original data
            #   Either assign those weights in the input, if possible
            #   OR do each MLR individually, then multiply coefficients by weights and add....
            
            nowcast = mlr_fit.get_prediction(exog=training_df, transform=True)
            alpha_level = 0.32  #  alpha is 1-CI or 1-sigma_level
                                #  alpha = 0.05 gives 2sigma or 95%
                                #  alpha = 0.32 gives 1sigma or 68%
            result = nowcast.summary_frame(alpha_level)
            result = result.set_index(training_df.index)
            
            predictions_dict[model_name] = mlr_fit
            nowcast_dict[model_name] = result
            
        self.predictions_dict = predictions_dict
        self.nowcast_dict = nowcast_dict
            
        return predictions_dict
    
    def cast_Models(self, with_error=True):
        
        #   For now, assume that we get a MLR prediction from linear_regression()
        #   May want to retool this eventually to just add errors from shifts to models?
        #   i.e., characterize arrival time error with DTW, average those distributions across all epochs, then propagate
        #   without shifting?
        
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import re
        
        for cast_interval_name, cast_interval in self.cast_intervals.items():
            models_to_be_removed = []
        #for model_name in self.model_names:
            #   First: get the time_delta predictions for each model
            for model_name in cast_interval.model_names:
            #for cast_interval_name, cast_interval in self.cast_intervals.items():
                print("For model: {} ========================================".format(model_name))
                print(self.predictions_dict[model_name].summary())
                
                #   Need all mlr_formula_terms to cast the model, so drop terms where they aren't present
                cast_context_df = pd.concat([cast_interval.models[model_name], 
                                            cast_interval._primary_df['context']],
                                            axis='columns')
                
                rhs_terms = list(set(self.mlr_formula_terms) - set(['empirical_time_delta']))
                
                cast_context_df = cast_context_df.dropna(subset=rhs_terms, axis='index', how='any')
                
                if len(cast_context_df) > 0:
                    cast = self.predictions_dict[model_name].get_prediction(cast_context_df)
                    alpha_level = 0.32  #  alpha is 1-CI or 1-sigma_level
                                        #  alpha = 0.05 gives 2sigma or 95%
                                        #  alpha = 0.32 gives 1sigma or 68%
                    result = cast.summary_frame(alpha_level)
                
                    #   !!!! N.B. Where data is present, the interval is smaller--
                    sig = (result['obs_ci_upper'] - result['obs_ci_lower'])/2.
                    
                    #   s_mu = shift_mu = mean shift
                    s_mu = result['mean'].set_axis(cast_context_df.index)
                    s_sig = sig.set_axis(cast_context_df.index)
                
                    self.cast_intervals[cast_interval_name]._primary_df.loc[:, (model_name, 'mlr_time_delta')] = s_mu
                    self.cast_intervals[cast_interval_name]._primary_df.loc[:, (model_name, 'mlr_time_delta_sigma')] = s_sig
                else:
                    message = ('Context for Model: {} is partially or fully '
                               'missing in cast interval: {}. Removing this '
                               'model from this cast interval now...')
                    logging.warning(message.format(model_name, cast_interval_name))
                    models_to_be_removed.append(model_name)
        
            for model_name in models_to_be_removed:
                self.cast_intervals[cast_interval_name].delModel(model_name)
            
           # breakpoint()
            #   Second: shift the models according to the time_delta predictions    
            if with_error:
                cast_interval.shift_Models(time_delta_column='mlr_time_delta', time_delta_sigma_column='mlr_time_delta_sigma')
            else:
                cast_interval.shift_Models(time_delta_column='mlr_time_delta')
                
        
        # if with_error:
        #     for cast_interval_name, cast_interval in self.cast_intervals.items():
        #         cast_interval.shift_Models(time_delta_column='mlr_time_delta', time_delta_sigma_column='mlr_time_delta_sigma')
        # else:
        #     for cast_interval_name, cast_interval in self.cast_intervals.items():
        #         cast_interval.shift_Models(time_delta_column='mlr_time_delta')
        # return
        
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
              
import _MMESH_mixins
class Trajectory(_MMESH_mixins.visualization):
    """
    The Trajectory class is designed to hold trajectory information in 4D
    (x, y, z, t), together with model and spacecraft data along the same path.
    A given trajectory may correspond to 1 spacecraft but any number of models.
    """
    
    def __init__(self, name=''):
        
        self.metric_tags = ['u_mag', 'p_dyn', 'n_tot', 'B_mag']
        self.variables =  ['u_mag', 'n_tot', 'p_dyn', 'B_mag']
        self.variables_unc = [var+'_pos_unc' for var in self.variables] + [var+'_neg_unc' for var in self.variables]
        
        self.trajectory_name = name
        
        self.spacecraft_name = ''
        self.spacecraft_df = ''
        
        self.model_names = []
        self.model_dfs = {}
        
        #   Can only choose one shift method at a time
        self.model_shifts = {}
        self.model_shift_stats = {}
        self.model_shift_method = None
        self.best_shifts = {}
        self.best_shift_functions = {}
        
        #self.model_shift_stats = {}
        #self.model_dtw_stats = {}
        #self.model_dtw_times = {}
        
        #self.trajectory = pd.Dataframe(index=index)
        
        
        
        self._data = None
        self._primary_df = pd.DataFrame()
    
        self.initialize_plotprops()
    
    def __len__(self):
        return len(self._primary_df)
    
    def _add_to_primary_df(self, label, df):
        
        #   Relabel the columns using MultiIndexing to add the label
        df.columns = pd.MultiIndex.from_product([[label], df.columns])
        
        if self._primary_df.empty:
            self._primary_df = df
        else:
            new_keys = [*self._primary_df.columns.levels[0], label]
            self._primary_df = pd.concat([self._primary_df, df], axis=1)
        
        #   Using this method will always re-set the index to datetime
        #   which is useful for preventing errors
        self._primary_df.index = pd.to_datetime(self._primary_df.index)
        self._primary_df = self._primary_df.sort_index()
        
        return
    
    @property
    def context(self):
        return self._primary_df['context']
    
    @context.setter
    def context(self, df):
        self._add_to_primary_df('context', df)

    @property
    def data(self):
        return self._primary_df[self.spacecraft_name].dropna(axis='index', how='all')
    
    @data.setter
    def data(self, df):
        
        #  Find common columns between the supplied DataFrame and internals
        int_columns = list(set(df.columns).intersection(self.variables))
        df = df[int_columns]

        self._add_to_primary_df(self.spacecraft_name, df)
        
        #   data_index is useful for statistical comparison between model and data
        #   data_span is useful for propagating the models
        self.data_index = self.data.dropna(axis='index', how='all').index
        self.data_span = np.arange(self.data_index[0], self.data_index[-1], dt.timedelta(hours=1))
    
    def addData(self, spacecraft_name, spacecraft_df):
        self.spacecraft_name = spacecraft_name
        self.data = spacecraft_df
    
    @property
    def models(self):
        return self._primary_df[self.model_names]
    
    @models.setter
    def models(self, df):
        int_columns = list(set(df.columns).intersection(self.variables))
        int_columns.extend(list(set(df.columns).intersection(self.variables_unc)))
        df = df[int_columns]
        
        self._add_to_primary_df(self.model_names[-1], df)
        
    def addModel(self, model_name, model_df):
        self.model_names.append(model_name)
        #self.model_names = sorted(self.model_names)
        self.models = model_df
        
    # @deleter
    # def models(self, model_name):
        
    def delModel(self, model_name):
        self.model_names.remove(model_name)
        self._primary_df.drop(model_name, axis='columns', inplace=True)

    # def baseline(self):
    #     import scipy
        
    #     models_stats = {}
    #     for model_name in self.model_names:
            
    #         model_stats = {}
    #         for m_tag in self.metric_tags:
                
    #             #  Find overlap between data and this model, ignoring NaNs
    #             i1 = self.data.dropna(subset=[m_tag]).index
    #             i2 = self.models[model_name].dropna(subset=[m_tag]).index
    #             intersection_indx = i1.intersection(i2) 
                
    #             if len(intersection_indx) > 2:
    #                 #  Calculate r (corr. coeff.) and both standard deviations
    #                 r, pvalue = scipy.stats.pearsonr(self.data[m_tag].loc[intersection_indx],
    #                                                  self.models[model_name][m_tag].loc[intersection_indx])
    #                 model_stddev = np.std(self.models[model_name][m_tag].loc[intersection_indx])
    #                 spacecraft_stddev = np.std(self.data[m_tag].loc[intersection_indx])
    #             else:
    #                 r = 0
    #                 model_stddev = 0
    #                 spacecraft_stddev = 0
                
    #             model_stats[m_tag] = (spacecraft_stddev, model_stddev, r)
                
    #         models_stats[model_name] = model_stats
        
    #     return models_stats
    
    def baseline(self, parameter):
        
        df = self._primary_df.xs(parameter, axis=1, level=1, drop_level=False)
        
        #   Record which models are all NaNs, then remove them from the DataFrame
        nan_models = df.columns[df.isnull().all()].get_level_values(0)
        df = df.dropna(axis='columns', how='all')
        remaining_models = list(set(self.model_names) - set(nan_models))
        
        #   Then, drop any remaining NaN rows
        df = df.dropna(axis='index', how='any')
        
        try: results_list = {self.spacecraft_name: (0, np.std(df[self.spacecraft_name].values), 0)}
        except: breakpoint()
        for model_name in remaining_models:
            ref = df[self.spacecraft_name][parameter].values
            tst = df[model_name][parameter].values
            (r, sig), rmsd = TD.find_TaylorStatistics(tst, ref)
            results_list[model_name] = (r, sig, rmsd)
        
        for model_name in nan_models:
            results_list[model_name] = (np.nan, np.nan, np.nan)
            
        return results_list
    
    def plot_TaylorDiagram(self, tag_name='', fig=None, ax=None, **plt_kwargs):

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
            
            (r, std), rmsd = TD.find_TaylorStatistics(self.models[model_name][tag_name].loc[self.data_index].to_numpy(dtype='float64'), ref_data)
            
            ax.scatter(np.arccos(r), std, label=model_name,
                       marker=self.gpp('marker',model_name), s=72, c=self.gpp('color',model_name),
                       zorder=10)
        
        return (fig, ax)
    
    def binarize(self, parameter, smooth = 1, sigma = 3, reswidth=0.0):
        """

        Parameters
        ----------
        dataframe : TYPE
            Pandas DataFrame
        tag : TYPE
            Column name in dataframe in which you're trying to find the jump
        smoothing_window : TYPE
            Numerical length of time, in hours, to smooth the input dataframe
        resolution_window : TYPE
            Numerical length of time, in hours, to assign as a width for the jump

        Returns
        -------
        A copy of dataframe, with the new column "jumps" appended.

        """
        #from SWData_Analysis import find_Jumps
        
        if type(smooth) != dict:
            smoothing_kernels = dict.fromkeys([self.spacecraft_name, *self.model_names], smooth)
        else:
            smoothing_kernels = smooth
        
        def find_Jumps(dataframe, tag, sigma_cutoff, smoothing_width, resolution_width=0.0):
            halftimedelta = dt.timedelta(hours=0.5*smoothing_width)
            half_rw = dt.timedelta(hours=0.5*resolution_width)
            
            output_tag = 'jumps'
            
            outdf = dataframe.copy()
            
            rolling_dataframe = dataframe.rolling(2*halftimedelta, min_periods=1, 
                                                  center=True, closed='left')  
            smooth = rolling_dataframe[tag].mean()
            smooth_derivative = np.gradient(smooth, smooth.index.values.astype(np.float64))
            
            #  Take the z-score of the derivative of the smoothed input,
            #  then normalize it based on a cutoff sigma
            smooth_derivative_zscore = smooth_derivative / np.nanstd(smooth_derivative)
            rough_jumps = np.where(smooth_derivative_zscore > sigma_cutoff, 1, 0)
            
            #!!!!!!!! NEED TO CATCH NO JUMPS!
            if (rough_jumps == 0.).all():
                print('No jumps in this time series! Must be quiescent solar wind conditions...')
                plt.plot(smooth_derivative_zscore)
                outdf[output_tag] = rough_jumps
                return(dataframe)
            
            #  Separate out each jump into a list of indices
            rough_jumps_indx = np.where(rough_jumps == 1)[0]
            rough_jump_spans = np.split(rough_jumps_indx, np.where(np.diff(rough_jumps_indx) > 1)[0] + 1)
            
            #  Set the resolution (width) of the jump
            jumps = np.zeros(len(rough_jumps))
            for span in rough_jump_spans:
                center_indx = np.mean(span)
                center = dataframe.index[int(center_indx)]
                span_width_indx = np.where((dataframe.index >= center - half_rw) &
                                           (dataframe.index <= center + half_rw))
                jumps[np.array(span_width_indx)+1] = 1  #  !!!!! Check +1 logic
            
            outdf[output_tag] = jumps
            return(outdf)
        
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
            try: 
                sd = smooth_deriv(param_df[name], dt.timedelta(hours=1))
            except ValueError:
                breakpoint()
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
            print("Smoothing of {} hours yields a standard deviation of {} for {}".format(str(smooth_window), smooth_score, name))
            
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
        
        for model_name in self.model_names:
            
            stats = []
            time_deltas = []
            for shift in shifts:
                
                shift_model_df = self.models[model_name].copy(deep=True)
                
                shift_model_df.index += dt.timedelta(hours=int(shift))
                
                shift_model_df = pd.concat([self.data, shift_model_df], axis=1,
                                            keys=[self.spacecraft_name, model_name])
                
                #  !!!! This could use the internal 'baseline' method now!
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
                time_delta = pd.DataFrame([shift]*len(self.models[model_name].index), index=self.models[model_name].index, columns=[str(shift)])
                time_deltas.append(time_delta)
                stats.append(pd.DataFrame.from_dict(stat))
            
            stats_df = pd.concat(stats, axis='index', ignore_index=True)
            times_df = pd.concat(time_deltas, axis='columns')
            self.model_shifts[model_name] = times_df
            self.model_shift_stats[model_name] = stats_df
            
            best_shift_indx = np.argmax(stats_df['r'])
            self.best_shifts[model_name] = stats_df.iloc[best_shift_indx]
        
        self.model_shift_method = 'constant'
        return self.model_shift_stats
    
    # def find_WarpStatistics_old(self, basis_tag, metric_tag, shifts=[0], intermediate_plots=True):
    #     import sys
    #     import DTW_Application as dtwa
    #     import SWData_Analysis as swda
        
    #     for model_name in self.model_names:
               
    #         #  Only compare where data exists
    #         df_nonan = self._primary_df.dropna(subset=[(self.spacecraft_name, basis_tag)], axis='index')
            
    #         stats = []
    #         time_deltas = []
    #         for shift in shifts:
    #             stat, time_delta = dtwa.find_SolarWindDTW(df_nonan[self.spacecraft_name], df_nonan[model_name], shift, 
    #                                                       basis_tag, metric_tag, 
    #                                                       total_slope_limit=96.0,
    #                                                       intermediate_plots=intermediate_plots,
    #                                                       model_name = model_name,
    #                                                       spacecraft_name = self.spacecraft_name)
    #             time_delta = time_delta.rename(columns={'time_lag': str(shift)})
                
    #             stats.append(pd.DataFrame.from_dict(stat))
    #             time_deltas.append(time_delta)
            
    #         stats_df = pd.concat(stats, axis='index', ignore_index=True)
    #         times_df = pd.concat(time_deltas, axis='columns')
    #         self.model_shifts[model_name] = times_df
    #         self.model_shift_stats[model_name] = stats_df
        
    #     self.model_shift_method = 'dynamic'
    #     return self.model_shift_stats
    
    def find_WarpStatistics(self, basis_tag, metric_tag, shifts=[0], intermediate_plots=True):
        import scipy
        import dtw
        
        from sklearn.metrics import confusion_matrix
        
        #   Sets the global slope limit at +/-4*24 hours-- i.e., the maximum 
        #   deviation from nominal time is +/- 4 days
        #   Since we shift the full model, it would be nice if this could be 
        #   asymmetric, such that the combined shift cannot exceed |4| days
        total_slope_limit = 4*24.  #  !!!!! Adjust, add as argument !!!!!!!!!!!!!!! SHOULD BE DIVIDED BY 2???????????????
        
        for model_name in self.model_names:
            
            #   Only compare where data exists
            #   df_nonan = self._primary_df.dropna(subset=[(self.spacecraft_name, basis_tag)], axis='index')
            
            stats = []
            time_deltas = []
            
            for shift in shifts:
                 
                window_args = {'window_size': total_slope_limit, 'constant_offset': shift}
                
                # =============================================================================
                #   Reindex the reference by the shifted amount, 
                #   then reset the index to match the query
                # ============================================================================= 
                shift_model_df = self.models[model_name].copy()
                shift_model_df.index = shift_model_df.index + dt.timedelta(hours=float(shift))
                shift_model_df = shift_model_df.reindex(self.data.index, method='nearest')        
                 
                reference = shift_model_df[basis_tag].to_numpy('float64')
                query = self.data[basis_tag].to_numpy('float64') 
                
                #   Custom window
                def custom_window(iw, jw, query_size, reference_size, window_size, constant_offset):
                    diagj = (iw * reference_size / query_size)
                    return abs(jw - diagj - constant_offset) <= window_size
                
                alignment = dtw.dtw(query, reference, keep_internals=True, 
                                    step_pattern='symmetric2', open_end=False,
                                    window_type=custom_window, window_args=window_args)
                #alignment.plot(type="threeway")
                #alignment.plot(type='density')
                
                def find_BinarizedTiePoints(alignment, query=True, reference=True,
                                            open_end=False, open_begin=False):
                    tie_points = []
                    if query == True:
                        for unity_indx in np.where(alignment.query == 1.0)[0]:
                            i1 = np.where(alignment.index1 == unity_indx)[0][0]
                            tie_points.append((alignment.index1[i1], alignment.index2[i1]))
                    
                    if reference == True:
                        for unity_indx in np.where(alignment.reference == 1.0)[0]:
                            i2 = np.where(alignment.index2 == unity_indx)[0][0]
                            tie_points.append((alignment.index1[i2], alignment.index2[i2]))
                        
                    #   If appropriate, add initial and final points as ties
                    #   If DTW is performed w/o open_begin or open_end, then 
                    #   the start and stop points are forced to match
                    if open_begin == False:
                        tie_points.insert(0, (0,0))
                    if open_end == False:
                        tie_points.append((alignment.N-1, alignment.M-1))
                    
                    return sorted(tie_points)
                
                #   List of tuples; each tuple matches query index (first) to 
                #   reference index (second)
                tie_points = find_BinarizedTiePoints(alignment)
                
                #   Map the reference to the query
                reference_to_query_indx = np.zeros(len(reference))
                for (iq_start, ir_start), (iq_stop, ir_stop) in zip(tie_points[:-1], tie_points[1:]):
                    reference_to_query_indx[iq_start:iq_stop+1] = np.linspace(ir_start, ir_stop, (iq_stop-iq_start+1))
                
                #   Convert from indices to time units
                q_time = (self.data.index - self.data.index[0]).to_numpy('timedelta64[s]').astype('float64') / 3600.
                r_time = (shift_model_df.index - shift_model_df.index[0]).to_numpy('timedelta64[s]').astype('float64') / 3600.
                
                #   This should work to interpolate any model quantity to the 
                #   warp which best aligns with data
                #   !!! r_time_warped is IDENTICAL to reference_to_query_indx...
                r_time_warped = np.interp(reference_to_query_indx, np.arange(0,len(reference)), r_time)
                r_time_deltas = r_time - r_time_warped  #   !!!! changed from r_time_warped - r_time to current form MJR 2023/11/30
                
                #   Don't spline from the zeros at the start and end of the time series
                #   These are an artifact of fixing the two time series to match at start and end
                r_time_deltas[tie_points[0][0]:tie_points[1][0]] = r_time_deltas[tie_points[1][0]+1]
                r_time_deltas[tie_points[-2][0]:tie_points[-1][0]+1] = r_time_deltas[tie_points[-2][0]-1]
                
                #   Warp the reference (i.e., shifted model at basis tag)
                #   This DOES work to interpolate any model quantity! Finally!
                #   I guess the confusing thing is that the resulting time 
                #   series uses unwarped abcissa and warped ordinate
                def shift_function(arr):
                    return np.interp(r_time - r_time_deltas, r_time, arr)
                
                r_basis_warped = shift_function(reference)
                r_metric_warped = shift_function(shift_model_df[metric_tag].to_numpy('float64'))
                
                #   If the basis metric is binarized, the warping may introduce
                #   non-binary ordinates due to interpolation. Check for these
                #   and replace them with binary values.
                #   !!!! May not work with feature widths/implicit errors
                if basis_tag.lower() == 'jumps':
                    temp_arr = np.zeros(len(query))
                    for i, (val1, val2) in enumerate(zip(r_basis_warped[0:-1], r_basis_warped[1:])):
                        if val1 + val2 >= 1:
                            if val1 > val2:
                                temp_arr[i] = 1.0
                            else:
                                temp_arr[i+1] = 1.0
                    r_basis_warped = temp_arr
                
                cm = confusion_matrix(query, r_basis_warped, labels=[0,1])
                
                #  Block divide by zero errors from printing to console
                #   Then get some confusion-matrix-based statistics
                with np.errstate(divide='log'):
                    accuracy = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,0] + cm[0,1] + cm[1,1])
                    precision = (cm[1,1])/(cm[0,1] + cm[1,1])
                    recall = (cm[1,1])/(cm[1,0] + cm[1,1])
                    f1_score = 2 * (precision * recall)/(precision + recall)
                
                #   Get the Taylor statistics (r, sigma, rmse)
                (r, sig), rmsd = TD.find_TaylorStatistics(r_metric_warped, self.data[metric_tag].to_numpy('float64'))
                
                if intermediate_plots == True:
                    dtwa.plot_DTWViews(self.data, shift_model_df, shift, alignment, basis_tag, metric_tag,
                                  model_name = model_name, spacecraft_name = self.spacecraft_name)    
                
                    #   New plotting function
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    from matplotlib import collections  as mc
                    from matplotlib.patches import ConnectionPatch
                    
                    fig, axs = plt.subplots(nrows=3, height_ratios=[1,2,2], sharex=True, figsize=(2, 4.5))
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.0)
                    
                    axs[0].plot(self.data.loc[self.data[basis_tag]!=0, basis_tag].index, 
                                self.data.loc[self.data[basis_tag]!=0, basis_tag].values,
                                linestyle='None', color=spacecraft_colors[self.spacecraft_name], marker='o', zorder=3)
                    
                    axs[0].plot(shift_model_df.loc[shift_model_df[basis_tag]!=0, basis_tag].index, 
                                shift_model_df.loc[shift_model_df[basis_tag]!=0, basis_tag].values*2,
                                linestyle='None', color=self.gpp('color',model_name), marker='o', zorder=2)
                    
                    axs[0].set(ylim=[0.5,2.5], yticks=[1, 2], yticklabels=['Data', 'Model'])
                    
                    tie_points = find_BinarizedTiePoints(alignment, query=False)
                    connecting_lines = []
                    for point in tie_points:
                        query_date = mdates.date2num(self.data.index[point[0]])
                        reference_date = mdates.date2num(shift_model_df.index[point[1]])
                        connecting_lines.append([(query_date, 1), (reference_date, 2)])
    
                    lc = mc.LineCollection(connecting_lines, 
                                           linewidths=1, linestyles=":", color='gray', linewidth=2, zorder=1)
                    axs[0].add_collection(lc)
                    
                    axs[1].plot(self.data.index, self.data[metric_tag], color=spacecraft_colors[self.spacecraft_name])
                    axs[1].plot(shift_model_df.index, shift_model_df[metric_tag], color=self.gpp('color',model_name))
                    
                    axs[2].plot(self.data.index, self.data[metric_tag], color=spacecraft_colors[self.spacecraft_name])
                    axs[2].plot(shift_model_df.index, r_metric_warped,  color=self.gpp('color',model_name))
                    
                    for point in tie_points:
                        query_date = mdates.date2num(self.data.index[point[0]])
                        reference_date = mdates.date2num(shift_model_df.index[point[1]])
                        #shift_date = mdates.date2num(shift_model_df.index[point[1]] + dt.timedelta(hours=r_time_deltas[point[1]]))
                        
                        xy_top = (reference_date, shift_model_df[metric_tag].iloc[point[1]])
                        xy_bottom = (query_date, r_metric_warped[point[0]])
                        
                        con = ConnectionPatch(xyA=xy_top, xyB=xy_bottom, 
                                              coordsA="data", coordsB="data",
                                              axesA=axs[1], axesB=axs[2], color="gray", linewidth=2, linestyle=":")
                        axs[2].add_artist(con)
                    plt.show()
                
                #   20231019: Alright, I'm 99% sure these are reporting shifts correctly
                #   That is, a positive delta_t means you add that number to the model
                #   datetime index to best match the data. Vice versa for negatives
                #   20240131: This is almost right-- a positive delta_t does NOT mean
                #   that the current point should be shifted by +dt forward; it means 
                #   that the point that should be in the current position *needs* to be 
                #   shifted +dt forward
                
                time_delta_df = pd.DataFrame(data=r_time_deltas, index=shift_model_df.index, columns=[str(shift)])
                
                width_68 = np.percentile(r_time_deltas, 84) - np.percentile(r_time_deltas, 16)
                width_95 = np.percentile(r_time_deltas, 97.5) - np.percentile(r_time_deltas, 2.5)
                width_997 = np.percentile(r_time_deltas, 99.85) - np.percentile(r_time_deltas, 0.15)
                d = {'shift': [shift],
                     'distance': [alignment.distance],
                     'normalizeddistance': [alignment.normalizedDistance],
                     'r': [r],
                     'stddev': [sig],
                     'rmsd': [rmsd],
                     'true_negative': cm[0,0],
                     'true_positive': cm[1,1],
                     'false_negative': cm[1,0],
                     'false_positive': cm[0,1],
                     'accuracy': accuracy,
                     'precision': precision,
                     'recall': recall,
                     'f1_score': f1_score,
                     'width_68': [width_68],
                     'width_95': [width_95],
                     'width_997': [width_997]}
                
                stats.append(pd.DataFrame.from_dict(d))
                time_deltas.append(time_delta_df)
            
            stats_df = pd.concat(stats, axis='index', ignore_index=True)
            times_df = pd.concat(time_deltas, axis='columns')
            self.model_shifts[model_name] = times_df
            self.model_shift_stats[model_name] = stats_df
            
        self.model_shift_method = 'dynamic'
        return self.model_shift_stats
    
    def optimize_Warp(self, eqn):
        for model_name in self.model_names:
            
            form = eqn(self.model_shift_stats[model_name])
            
            best_shift_indx = np.argmax(form)
            self.best_shifts[model_name] = self.model_shift_stats[model_name].iloc[best_shift_indx]
            
            #   Add the new shift times to the model dataframe
            print(self.best_shifts[model_name])
            constant_offset = self.best_shifts[model_name]['shift']
            print(constant_offset)
            dynamic_offsets = self.model_shifts[model_name][str(int(constant_offset))]
            
            #   So, this time_delta DOES NOT mean 'point 0 belongs at time[0] + time_delta[0]'
            #   Instead, it means that 'the point that belongs at time[0] is currently at time[0]-time_delta[0]'
            #   Apply the constant offset everywhere, but the dynamic ones only where data exist
            #   !!!! May want to reconsider this at some point
            self._primary_df.loc[:, (model_name, 'empirical_time_delta')] = constant_offset
            self._primary_df.loc[self.data_index, (model_name, 'empirical_time_delta')] += dynamic_offsets
            
        self._dtw_optimization_equation = eqn
        
        return
    
    # =============================================================================
    #   Figure 5
    # =============================================================================
    def plot_OptimizedOffset(self, basis_tag, metric_tag, fullfilepath=''):
        #   New plotting function
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib import collections  as mc
        from matplotlib.patches import ConnectionPatch
        
        #   Optionally, choose which models get plotted
        model_names = ['HUXt']
        
        fig, axs = plt.subplots(nrows=3, ncols=len(model_names), 
                                figsize=self.plotprops['figsize'], 
                                height_ratios=[1,2,2],
                                sharex=True, sharey='row', squeeze=False)
        plt.subplots_adjust(**self.plotprops['adjustments'])
        
        for num, (model_name, ax_col) in enumerate(zip(model_names, axs.T)):
            letter_labels = string.ascii_lowercase[num*3:(num+3)*3]
            ax_col[0].annotate('({})'.format(letter_labels[0]), (0,1), (1,-1), 
                               xycoords='axes fraction', textcoords='offset fontsize', 
                               va='top', ha='left')
            ax_col[1].annotate('({})'.format(letter_labels[1]), (0,1), (1,-1), 
                               xycoords='axes fraction', textcoords='offset fontsize', 
                               va='top', ha='left')
            ax_col[2].annotate('({})'.format(letter_labels[2]), (0,1), (1,-1), 
                               xycoords='axes fraction', textcoords='offset fontsize', 
                               va='top', ha='left')
            
            ax_col[0].scatter(self.index_ddoy(self.data.loc[self.data[basis_tag]!=0, basis_tag].index), 
                              self.data.loc[self.data[basis_tag]!=0, basis_tag].values,
                              # color=self.gpp('color','data'), edgecolors='black',
                              color='black', edgecolors='black',
                              marker=self.gpp('marker','data'), s=32,
                              zorder=3)
        
            ax_col[0].scatter(self.index_ddoy(self.models[model_name].loc[self.models[model_name][basis_tag]!=0, basis_tag].index), 
                              self.models[model_name].loc[self.models[model_name][basis_tag]!=0, basis_tag].values*2,
                              color=self.gpp('color',model_name), edgecolors='black',
                              marker=self.gpp('marker',model_name), s=32,
                              zorder=2)
            
            #   Plot Model and Data Sources, color-coded
            
            # ax_col[0].annotate('Model: {}'.format(model_name), 
            #                    (0, 1), xytext=(0.1, 0.2), va='bottom', ha='left',
            #                    xycoords='axes fraction', textcoords='offset fontsize',
            #                    bbox=label_bbox)
            # x_offset = len(model_name)+8+0.1  #  Offset for text
            # ax_col[0].annotate('Data: {}'.format(self.spacecraft_name),
            #                    (0, 1), xytext=(x_offset, 0.2), va='bottom', ha='left',
            #                    xycoords='axes fraction', textcoords='offset fontsize',
            #                    bbox=label_bbox)
            ax_col[0].set(ylim=[0.5,2.5], yticks=[1, 2],
                          yticklabels=['Data: {}'.format(self.spacecraft_name), 
                                       'Model: {}'.format(model_name)])
            
            label_bbox = dict(facecolor=self.gpp('color','data'), 
                                   edgecolor=self.gpp('color','data'), 
                                   pad=0.1, boxstyle='round')
            # ax_col[0].get_yticklabels()[0].set_bbox(label_bbox)
            
            label_bbox = dict(facecolor=self.gpp('color',model_name), 
                                   edgecolor=self.gpp('color',model_name), 
                                   pad=0.1, boxstyle='round')
            ax_col[0].get_yticklabels()[1].set_bbox(label_bbox)
            
            
            
            ax_col[1].plot(self.index_ddoy(self.data.index), self.data[metric_tag], 
                           # color=self.gpp('color','data'), linewidth=1)
                           color='black', linewidth=1)
            ax_col[2].plot(self.index_ddoy(self.data.index), self.data[metric_tag], 
                           # color=self.gpp('color','data'), linewidth=1)
                           color='black', linewidth=1)
            
            ax_col[1].plot(self.index_ddoy(self.models[model_name].index), self.models[model_name][metric_tag],
                           color=self.gpp('color',model_name))

            temp_shift = self._shift_Models(time_delta_column='empirical_time_delta')
            #   !!!!! Make shift return just models, that's all it changes...
            # ax_col[2].plot(temp_shift.index, temp_shift[(model_name, metric_tag)],
            #                color=self.gpp('color',model_name))        
            ax_col[2].plot(self.index_ddoy(self.models[model_name].index), temp_shift[(model_name, metric_tag)],
                           color=self.gpp('color',model_name))  
                                                 
            connections_basis = []
            connections_metric = []
            for i, (index, value) in enumerate(self.models[model_name].loc[self.models[model_name][basis_tag]!=0, basis_tag].items()):
                
                #  Find the time-coordinate for where each 'jump' got shifted to
                t_elapsed = (self.models[model_name].index - self.models[model_name].index[0]).total_seconds()/3600.
                x = t_elapsed - self.models[model_name].loc[:, 'empirical_time_delta'].values
                x_tie = (index - self.models[model_name].index[0]).total_seconds()/3600.
                shift_index_elapsed = np.interp(x_tie, x, t_elapsed)
                
                shift_index = index + dt.timedelta(hours=(shift_index_elapsed-x_tie))
                
                #   !!! Can't use built-in self.index_ddoy with an individual entry...
                #   
                index_ddoy = (index - dt.datetime(self.data_index[0].year,1,1)).total_seconds()/(24*60*60) + 1/24.
                shift_index_ddoy = (shift_index - dt.datetime(self.data_index[0].year,1,1)).total_seconds()/(24*60*60) + 1/24.
                
                #   This first plot shows the matched features, so only add
                #   the connection if both features exist                
                if shift_index in self.data.index:
                    if self.data.loc[shift_index, 'jumps'] == 1:
                        pair1 = (index_ddoy, 2)
                        pair2 = (shift_index_ddoy, 1)
                        
                        connections_basis.append([pair1, pair2])
                
                #   For the shifted abcissa (y) value, just take the unshifted y value
                #   Shifts only change x, not y, so the y value should not change
                #   And this saves having to interpolate
                pair3 = (index_ddoy, self.models[model_name].loc[index, metric_tag])
                pair4 = (shift_index_ddoy, self.models[model_name].loc[index, metric_tag])
                
                connections_metric.append([pair3, pair4])
                
                con = ConnectionPatch(xyA=pair3, xyB=pair4, 
                                     coordsA="data", coordsB="data",
                                     axesA=ax_col[1], axesB=ax_col[2],
                                     color="gray", linewidth=1.5, linestyle=":", alpha=0.8, zorder=5)
                ax_col[2].add_artist(con)
            
            lc = mc.LineCollection(connections_basis, 
                                    linewidths=1.5, linestyles="-", color='gray', linewidth=1.5, zorder=1)
            ax_col[0].add_collection(lc)
            
            ax_col[0].set_xlim(self.index_ddoy(self.data.index)[0], self.index_ddoy(self.data.index)[-1])
        
        axs[1,-1].annotate('Original', 
                           (1,0.5), xytext=(0.25,0), va='center', ha='left',
                           xycoords='axes fraction', textcoords='offset fontsize',
                           rotation='vertical')
        axs[2,-1].annotate('DTW Shifted',
                           (1,0.5), xytext=(0.25,0), va='center', ha='left',
                           xycoords='axes fraction', textcoords='offset fontsize',
                           rotation='vertical')
        
        #axs[0,0].locator_params(nbins=3, axis='x')
        #axs[0,0].xaxis.set_major_formatter(self.timeseries_formatter(axs[0,0].get_xticks()))
        axs[-1,-1].xaxis.set_major_locator(MultipleLocator(25))
        axs[-1,-1].xaxis.set_minor_locator(MultipleLocator(5))
        
        axs[-1,-1].yaxis.set_major_locator(MultipleLocator(250))
        axs[-1,-1].yaxis.set_minor_locator(MultipleLocator(50))

        fig.supxlabel('Day of Year {}'.format(self.data.index[0].year))
        fig.supylabel(r'Flow Speed ($u_{mag}$) [km/s]')
        
        for suffix in ['.png', '.eps', '.jpg']:
            fig.savefig(fullfilepath, dpi=300)
        plt.show()
        return
    
    def shift_Models(self, time_delta_column=None, time_delta_sigma_column=None, n_mc=10000):
        
        result = self._shift_Models(time_delta_column=time_delta_column, 
                                    time_delta_sigma_column=time_delta_sigma_column,
                                    n_mc=n_mc)
        self._primary_df = result
    
    def _shift_Models(self, time_delta_column=None, time_delta_sigma_column=None, n_mc=10000):
        """
        Shifts models by the specified column, expected to contain delta times in hours.
        Shifted models are only valid during the interval where data is present,
        and NaN outside this interval.
        
        This can also be used to propagate arrival time errors by supplying 
        time_delta_sigma with no time_delta (i.e., time_delta of zero)
        
        """
        import matplotlib.pyplot as plt
        import copy
        
        #   We're going to try to do this all with interpolation, rather than 
        #   needing to shift the model dataframe by a constant first
        
        shifted_primary_df = copy.deepcopy(self._primary_df)
        
        for model_name in self.model_names:
            try:
                time = ((self.models[model_name].index - self.models[model_name].index[0]).to_numpy('timedelta64[s]') / 3600.).astype('float64')
            except:
                breakpoint()
                
            #   If no time_delta_column is given, then set it to all zeros
            if time_delta_column == None:
                time_deltas = np.zeros(len(self.models[model_name]))
            else:
                time_deltas = self.models[model_name][time_delta_column].to_numpy('float64')
            
            #   If no time_delta_sigma_column is given
            #   We don't want to do the whole Monte Carlo with zeros
            if time_delta_sigma_column == None:
                print("Setting time_delta_sigmas to zero")
                time_delta_sigmas = np.zeros(len(self.models[model_name]))
            else:
                print("Using real time_delta_sigmas")
                time_delta_sigmas = self.models[model_name][time_delta_sigma_column].to_numpy('float64')
            
            #   Now we're shifting based on the ordinate, not the abcissa,
            #   so we have time - time_deltas, not +
            def shift_function(arr, col_name=''):
                #   Use the ORIGINAL time with this, not time + delta
                if (time_delta_sigmas == 0.).all():
                    arr_shifted = np.interp(time - time_deltas, time, arr)
                    #arr_shifted = np.interp(time, time+time_deltas, arr)
                    arr_shifted_pos_unc = arr_shifted * 0.
                    arr_shifted_neg_unc = arr_shifted * 0.
                else:
                    arr_perturb_list = []
                    r = np.random.default_rng()
                    for i in range(n_mc):
                        time_deltas_perturb = r.normal(time_deltas, time_delta_sigmas)
                        arr_perturb_list.append(np.interp(time - time_deltas_perturb, time, arr))
                        #arr_perturb_list.append(np.interp(time, time+time_deltas_perturb, arr))

                    arr_shifted = np.mean(arr_perturb_list, axis=0)
                    arr_shifted_sigma = np.std(arr_perturb_list, axis=0)
                    
                    arr_shifted = np.percentile(arr_perturb_list, 50, axis=0)
                    arr_shifted_pos_unc = np.percentile(arr_perturb_list, 84, axis=0) - arr_shifted
                    arr_shifted_neg_unc = arr_shifted - np.percentile(arr_perturb_list, 16, axis=0)
                    
                    # if col_name == 'u_mag':
                    #     fig, ax = plt.subplots()
                    #     ax.hist(np.array(arr_perturb_list)[:,0], range=[100, 900], bins=800)
                    #     ax.axvline(arr_shifted[0], color='black')
                    #     ax.axvline(arr_shifted[0] + arr_shifted_sigma[0], color='gray')
                    #     ax.axvline(arr_shifted[0] - arr_shifted_sigma[0], color='gray')
                output = (arr_shifted, arr_shifted_pos_unc, arr_shifted_neg_unc)
                return output
            
            # def test_shift_function(arr):
            #     #   Use the ORIGINAL time with this, not time + delta
            #     if (time_delta_sigmas == 0.).all():
            #         arr_shifted = np.interp(time - time_deltas, time, arr)
            #         arr_shifted_sigma = arr_shifted * 0.
            #     else:
            #         arr_shifted = np.interp(time - time_deltas, time, arr)
                    
            #         arr_shifted_plus = np.interp(time - (time_deltas + time_delta_sigmas), time, arr)
            #         arr_shifted_minus = np.interp(time - (time_deltas - time_delta_sigmas), time, arr)
                    
            #         (arr_shifted_plus - arr_shifted_minus)
            #     output = (arr_shifted, arr_shifted_sigma)
            #     return output
            
            #self.best_shift_functions[model_name] = shift_function
            
            for i, (col_name, col) in enumerate(self.models[model_name].items()):
                #   Skip columns of all NaNs and columns which don't appear
                #   in list of tracked variables
                #if len(col.dropna() > 0) and (col_name in self.variables):
                if (col_name in self.variables):
                    
                    col_shifted = shift_function(col.to_numpy('float64'), col_name=col_name)
                    
                    shifted_primary_df.loc[:, (model_name, col_name)] = col_shifted[0]
                    shifted_primary_df.loc[:, (model_name, col_name+'_pos_unc')] = col_shifted[1]
                    shifted_primary_df.loc[:, (model_name, col_name+'_neg_unc')] = col_shifted[2]
                    
        return shifted_primary_df
    
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
        
        # weights = dict.fromkeys(self.variables, 1.0/len(self.model_names))
        # print(weights)
        
        # if weights == None:
        #     for model in self.model_names:
        #         all_nans = ~self.model_dfs[model].isna().all()
        #         overlap_col = list(set(all_nans.index) & set(weights_df.columns))
        #         weights_df.loc[model] = all_nans[overlap_col].astype(int)
        #         weights_df = weights_df.div(weights_df.sum())
        
        # if dict, use dict
        # if list/array, assign to dict in order
        # if none, make dict with even weights
        
        #   Equal weights
        
        
        #   Equal weights, but ignoring NaNs and Zeros
        weights = np.zeros(np.shape(self.models[self.model_names[0]][self.variables]))
        for model_name in self.model_names:
            #   !!!! Add zero handling
            weights += self.models[model_name][self.variables].notna().astype('float64').values
        weights[np.where(weights > 0)] = 1.0/weights[np.where(weights > 0)]
        
        #   
        variables_unc = [v+'_pos_unc' for v in self.variables]
        variables_unc.extend([v+'_neg_unc' for v in self.variables])
        
        partials_list = []
        for model_name in self.model_names:
            df = self.models[model_name][self.variables].mul(weights, fill_value=0.)
            
            if set(variables_unc).issubset(set(self.models[model_name].columns)): #   If sigmas get added from start, get rid of this bit
                
                weights_for_unc = np.concatenate((weights, weights), axis=1)
                df[variables_unc] = self.models[model_name][variables_unc].mul(weights_for_unc, fill_value=0.)
                
            partials_list.append(df)
        
        #   Empty dataframe of correct dimensions for ensemble
        ensemble = partials_list[0].mul(0.)
        
        #   Sum weighted variables, and RSS weighted errors
        for partial in partials_list:
            ensemble.loc[:, self.variables] += partial.loc[:, self.variables].astype('float64')
            ensemble.loc[:, variables_unc] += (partial.loc[:, variables_unc].astype('float64'))**2.
            
        ensemble.loc[:, variables_unc] = np.sqrt(ensemble.loc[:, variables_unc])
        #   !!!! This doesn't add error properly
        # ensemble = reduce(lambda a,b: a.add(b, fill_value=0.), partials_list)

        self.addModel('ensemble', ensemble)
        
    # =============================================================================
    #   Plotting stuff
    # =============================================================================
    
    # def set_PresentationMode(self):
    #     self.plotpropsfigsize = (4, 3)
    #     self.plotparams.adjustments = {'left': 0.15, 'right':0.95, 
    #                                     'bottom':0.225, 'top':0.925, 
    #                                     'wspace':0.2, 'hspace':0.0}
    #     self.presentationmode = True
    
    def _plot_parameters(self, parameter):
        #  Check which parameter was specified and set up some plotting keywords
        match parameter:
            case ('u_mag' | 'flow speed'):
                tag = 'u_mag'
                ylabel = r'$u_{mag}$ [km/s]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'linear', 'ylim': (250, 600),
                           'yticks': np.arange(350,600+50,100)}
            case ('p_dyn' | 'pressure'):
                tag = 'p_dyn'
                ylabel = r'$p_{dyn}$ [nPa]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'log', 'ylim': (1e-3, 2e0),
                           'yticks': 10.**np.arange(-3,0+1,1)}
            case ('n_tot' | 'density'):
                tag = 'n_tot'
                ylabel = r'$n_{tot}$ [cm$^{-3}$]'
                plot_kw = {'ylabel': ylabel,
                           'yscale': 'log', 'ylim': (5e-3, 5e0),
                           'yticks': 10.**np.arange(-2, 0+1, 1)}
            case ('B_mag' | 'magnetic field'):
                tag = 'B_mag'
                ylabel = r'$B_{IMF}$ [nT]'
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
        
    def plot_SingleTimeseries(self, parameter, starttime, stoptime, fullfilepath=''):
        """
        Plot the time series of a single parameter from a single spacecraft,
        separated into 4/5 panels to show the comparison to each model.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        #import PlottingConstants
        
        tag, plot_kw = self._plot_parameters(parameter)
        
        # save_filestem = 'Timeseries_{}_{}_{}-{}'.format(self.spacecraft_name.replace(' ', ''),
        #                                                 tag,
        #                                                 starttime.strftime('%Y%m%d'),
        #                                                 stoptime.strftime('%Y%m%d'))
        
    
        fig, axs = plt.subplots(figsize=self.plotprops['figsize'], nrows=len(self.model_names), sharex=True, squeeze=False)
        plt.subplots_adjust(**self.plotprops['adjustments'])
        
        for indx, ax in enumerate(axs.flatten()):
            
            ax.plot(self.index_ddoy(self.data.index), 
                    self.data[tag],
                    color='black', alpha=1.0, linewidth=1.0,
                    label='Juno/JADE (Wilson+ 2018)')
            
            #ax.set(**plot_kw)
            ax.set_ylabel('')
            #  get_yticks() for log scale doesn't work as expected; workaround:
            yticks = ax.get_yticks()
            yticks = [yt for yt in yticks if ax.get_ylim()[0] <= yt <= ax.get_ylim()[1]]
            if indx != len(axs)-1: 
                if (yticks[0] in ax.get_ylim()) and yticks[-1] in ax.get_ylim():
                    ax.set(yticks=yticks[1:])
            #ax.grid(linestyle='-', linewidth=2, alpha=0.1, color='xkcd:black')
        print('Plotting models...')  
        for i, (ax, model_name) in enumerate(zip(axs.flatten(), self.model_names)):
            if tag in self.models[model_name].columns:
                ax.plot(self.index_ddoy(self.models[model_name].index),
                        self.models[model_name][tag], 
                        color=self.gpp('color',model_name), label=model_name, linewidth=2)
            
            model_label_box = dict(facecolor=self.gpp('color',model_name), 
                                   edgecolor=self.gpp('color',model_name), 
                                   pad=0.1, boxstyle='round')
           
            ax.text(0.01, 0.95, '({}) {}'.format(string.ascii_lowercase[i], model_name),
                    color='black',
                    horizontalalignment='left', verticalalignment='top',
                    #bbox = model_label_box,
                    transform = ax.transAxes)
            
            ax.set_xlim(self.index_ddoy(self.data.index)[0], self.index_ddoy(self.data.index)[-1])
            ax.xaxis.set_major_locator(MultipleLocator(25))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.set_yticks([0,1])
            ax.set_ylim([0,2])
        
        fig.text(0.5, 0.025, 'Day of Year {}'.format(self._primary_df.index[0].year),
                 ha='center', va='center')
        fig.text(0.025, 0.5, plot_kw['ylabel'], ha='center', va='center', rotation='vertical')
        
        # tick_interval = int((stoptime - starttime).days/10.)
        # subtick_interval = int(tick_interval/5.)
        
        # axs[0,0].set_xlim((starttime, stoptime))
        # axs[0,0].xaxis.set_major_formatter(self.timeseries_formatter(axs[0,0].get_xticks()))
        
        # axs[0,0].xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
        # if subtick_interval > 0:
        #     axs[0,0].xaxis.set_minor_locator(mdates.DayLocator(interval=subtick_interval))
        # axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%j'))
        
        # #  Trying to set up year labels, showing the year once at the start
        # #  and again every time it changes
        # ax_year = axs[-1].secondary_xaxis(-0.15)
        # ax_year.set_xticks([axs[-1].get_xticks()[0]])
        # ax_year.visible=False
        # ax_year.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.align_ylabels()
        #
        
        if fullfilepath != '':
            for suffix in ['.png', '.jpg']:
                plt.savefig(str(fullfilepath)+suffix, dpi=300, bbox_inches='tight')
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
                    color=self.gpp('color',model_name))
            
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
        
        fig, ax = TD.init_TaylorDiagram(np.nanstd(self.data['u_mag']), fig=fig)
        
        for model_name in self.model_names:
            
            ref = self.data['u_mag'].to_numpy(dtype='float64')
            tst = self.models[model_name]['u_mag'].loc[self.data_index].to_numpy(dtype='float64')
            
            fig, ax = TD.plot_TaylorDiagram(tst, ref, fig=fig, ax=ax, zorder=9, 
                                            s=36, c='black', marker=self.gpp('marker',model_name),
                                            label=model_name)
            
            # fig, ax = TD.plot_TaylorDiagram(tst, ref, fig=fig, ax=ax, zorder=10, 
            #                                 s=36, c=self.gpp('color',model_name), marker=self.gpp('marker',model_name),
            #                                 label=model_name + ' ' + str(shift) + ' h')
            # (r, sig), rmsd = TD.find_TaylorStatistics(tst, ref)
            # ax.scatter(np.arccos(r), sig, zorder=9, 
            #            s=36, c='black', marker=self.gpp('marker',model_name),
            #            label=model_name)
            
            best_shift_indx = np.argmax(self.model_shift_stats[model_name]['r'])
            shift = int((self.model_shift_stats[model_name].iloc[best_shift_indx])['shift'])
            
            shift_model_df = self.models[model_name].copy(deep=True)
            shift_model_df.index += dt.timedelta(hours=shift)
            shift_model_df = pd.concat([self.data, shift_model_df], axis=1,
                                        keys=[self.spacecraft_name, model_name])
            
            fig, ax = TD.plot_TaylorDiagram(shift_model_df[(model_name, 'u_mag')], 
                                            shift_model_df[(self.spacecraft_name, 'u_mag')], 
                                            fig=fig, ax=ax, zorder=9, 
                                            s=36, c=self.gpp('color',model_name), marker=self.gpp('marker',model_name),
                                            label=model_name + ' ' + str(shift) + ' h')
            
            # (r, sig), rmsd = TD.find_TaylorStatistics(shift_model_df[(model_name, 'u_mag')], 
            #                                                shift_model_df[(self.spacecraft_name, 'u_mag')]) #self.model_shift_stats[model_name].iloc[best_shift_indx][['shift', 'r', 'stddev', 'rmsd']]
            
            # ax.scatter(np.arccos(r), sig, zorder=10, 
            #             s=36, c=self.gpp('color',model_name), marker=self.gpp('marker',model_name),
            #             label=model_name + ' ' + str(shift) + ' h')
        
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        
    def plot_DynamicTimeWarping_Optimization(self, fig=None):
        import matplotlib.pyplot as plt
        import string
        
        if fig == None:
            fig = plt.figure(figsize=[6,4.5])
            
        gs = fig.add_gridspec(nrows=len(self.model_names), ncols=3, width_ratios=[1,1,1],
                              left=0.1, bottom=0.1, right=0.95, top=0.95,
                              wspace=0.0, hspace=0.1)
        
        axs0 = [fig.add_subplot(gs[y,0]) for y in range(len(self.model_names))]
        axs1 = [fig.add_subplot(gs[y,2]) for y in range(len(self.model_names))]
        
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
            ax0.plot(self.model_shift_stats[model_name]['shift'], self._dtw_optimization_equation(self.model_shift_stats[model_name]),
                     color=self.gpp('color',model_name), zorder=10)
            ax0_correl.plot(self.model_shift_stats[model_name]['shift'], self.model_shift_stats[model_name]['r'],
                            color='C1', zorder=1)
            ax0_width.plot(self.model_shift_stats[model_name]['shift'], (self.model_shift_stats[model_name]['width_68']/2.),
                           color='C3', zorder=1)

            #   Mark the maximum of the optimization function
            ax0.axvline(shift, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax0.annotate('({:.0f}, {:.3f}, {:.1f})'.format(shift, r, width/2.), 
                        (shift, r), xytext=(0.5, 0.5),
                        textcoords='offset fontsize', zorder=10)
            
            ax0.annotate(label1, (0,1), xytext=(0.5, -0.5),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               ha='left', va='top')
            
            dtw_opt_shifts = self.model_shifts[model_name]['{:.0f}'.format(shift)]
            histo = np.histogram(dtw_opt_shifts, range=[-96,96], bins=int(192/6))
            
            ax1.stairs(histo[0]/np.sum(histo[0]), histo[1],
                          color=self.gpp('color',model_name), linewidth=2)
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
        fig, ax = TD.init_TaylorDiagram(ref_std, fig=fig)

        for model_name in self.model_names:
            (r, sig), rmse = TD.find_TaylorStatistics(self.models[model_name]['u_mag'].to_numpy('float64'), 
                                                      self.data['u_mag'].to_numpy('float64'))
            ax.scatter(np.arccos(r), sig, 
                        marker=self.gpp('marker',model_name), s=24, c='black',
                        zorder=9,
                        label=model_name)
            
            #best_shift_indx = np.argmax(dtw_stats[model_name]['r'] * 6/(dtw_stats[model_name]['width_68']))
            # best_shift_indx = np.where(traj0.model_dtw_stats[model_name]['shift'] == best_shifts[model_name])[0]
            # r, sig = traj0.model_dtw_stats[model_name].iloc[best_shift_indx][['r', 'stddev']].values.flatten()
            r, sig = self.best_shifts[model_name][['r', 'stddev']].values.flatten()
            ax.scatter(np.arccos(r), sig, 
                        marker=self.gpp('marker',model_name), s=48, c=self.gpp('color',model_name),
                        zorder=10,
                        label='{} + DTW'.format(model_name))
            
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand', markerscale=1.0)
        ax.set_axisbelow(True)
        
    def timeseries_formatter(self, ticks):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        ticks = [mdates.num2date(t) for t in ticks]
    
        def date_formatter(tick, pos):
            tick = mdates.num2date(tick)
            tick_str = tick.strftime('%j')
           
            if (pos == 0):
                tick_str += '\n' + tick.strftime('%d %b')
                tick_str += '\n' + tick.strftime('%Y')
            
            if (pos != 0):
                try:
                    if (tick.month > ticks[pos-1].month):
                        tick_str += '\n' + tick.strftime('%d %b')
                    
                except:
                    print('FAILED')
                    print(tick.month)
                    print(pos-1)
                    print(ticks[pos-1])
                   
                if (tick.year > ticks[pos-1].year):
                    tick_str += '\n' + tick.strftime('%d %b')
                    tick_str += '\n' + tick.strftime('%Y')
           
            return tick_str
        return date_formatter
    
    def index_ddoy(self, index, start_ref=None):
        #   !!! If trajectory.data didn't subscript _primary_df, this could be
        #   and argument-less function...
        if start_ref == None:
            result = (index - dt.datetime(index[0].year,1,1)).total_seconds()/(24*60*60)
        else:
            result = (index - dt.datetime(start_ref.year,1,1)).total_seconds()/(24*60*60)
        return result