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
from tqdm import tqdm
import scipy

import plot_TaylorDiagram as TD

from matplotlib.ticker import MultipleLocator
import string
import MMESH_context as mmesh_c


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
    
    def __init__(self, config_fullfilepath=None, trajectories=[]):
        
        self.filepaths = {}
        
        self.domain = {}
        
        self.trajectories = {}
        self.trajectory_names = []
        self.trajectory_sources = {}
        self.model_names = []
        self.model_sources = {}
        
        self.cast_intervals = {}
        
        #   !!!! This should be removed, but the relex is here for back-compatibility
        self.spacecraft_names = self.trajectory_names
        
        if config_fullfilepath is not None:
            self.init_fromconfig(config_fullfilepath)
        else:
            self.spacecraft_names = []
            self.trajectories = {}   #  dict of Instances of Trajectory class
            self.cast_intervals = {}    #  ???? of Instances of Trajectory class
                                        #   i.e. hindcast, nowcast, forecase, simulcast
                                        #   all relative to data span (although cound be between data spans)
            
            #   The MultiTrajectory contains the superset of all trajectory models
            all_model_sources = {}
            for trajectory in trajectories:
                self.spacecraft_names.append(trajectory.spacecraft_name)
                self.trajectories[trajectory.trajectory_name] = trajectory
                #possible_models += trajectory.model_names
                all_model_sources.update(trajectory.model_sources)  #  overwritten by latest
            
            self.model_names = all_model_sources.keys()
            self.model_sources = all_model_sources
        
        self.offset_times = []  #   offset_times + (constant_offset) + np.linspace(0, total_hours, n_steps) give interpolation index
        self.optimized_shifts = []
        
    def init_fromconfig(self, config_fullfilepath):
        import tomllib
        from pathlib import Path
        
        with open(config_fullfilepath, "rb") as f:
            config_dict = tomllib.load(f)
        self.config = config_dict
        self.config_fullfilepath = config_fullfilepath
        
        #   Store path info as pathlib Path
        for path_name, path_spec in config_dict['paths'].items():
            self.filepaths[path_name] = Path(path_spec)
        
        #   Store domain info
        self.domain = config_dict['domain']
        
        #   Store data info
        for trajectory_name, val in config_dict['trajectories'].items():
            self.trajectories[trajectory_name] = Trajectory(name=trajectory_name)
            self.trajectories[trajectory_name].start = val['start']
            self.trajectories[trajectory_name].stop = val['stop']
            self.trajectories[trajectory_name].source = val['source']
            
            self.trajectory_names.append(trajectory_name)
            self.trajectory_sources[trajectory_name] = val['source']

        #   Store model info
        #   This doesn't go in the Trajectory proper right now, but instead in the MultiTrajectory directly
        #   The preferred way to add model info is through the model setter
        for model_name, model_info in config_dict['models'].items():
            self.model_names.append(model_name)
            self.model_sources[model_name] = model_info['source']
            
        #   Store the cast info
        for cast_interval_name, val in config_dict['cast_intervals'].items():
            self.cast_intervals[cast_interval_name] = Trajectory(name = cast_interval_name)
            self.cast_intervals[cast_interval_name].start = val['start']
            self.cast_intervals[cast_interval_name].stop = val['stop']
            self.cast_intervals[cast_interval_name].target = val['target']
        
        
    def init_fromtrajectory(self, trajectories):
        
        if not hasattr(self, 'trajectories'):
            self.trajectories = {}
        for trajectory in trajectories:
            self.trajectories[trajectory.trajectory_name] = trajectory
            
            
    
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
    
    def cast_Models(self, with_error=True, min_rel_scale = None):
        
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
                    #   The upper and lower confidence intervals ARE SYMMETRIC
                    #   As expected, these are normally distributed since 
                    #   negative temporal shifts are acceptable
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
            
            #   Second: shift the models according to the time_delta predictions    
            if with_error:
                cast_interval.shift_Models(time_delta_column='mlr_time_delta', time_delta_sigma_column='mlr_time_delta_sigma',
                                           force_distribution=True, compare_distributions=False, min_rel_scale = min_rel_scale)
            else:
                cast_interval.shift_Models(time_delta_column='mlr_time_delta')
            
    def ensemble(self, weights = None, as_skewnorm = True):   
        """
        Iteratively run Trajectory.ensemble()

        Returns
        -------
        None.

        """
        
        for trajectory_name, trajectory in self.cast_intervals.items():
            trajectory.ensemble(weights = weights,
                                as_skewnorm = as_skewnorm)
 
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
        
        self.distribution_function = scipy.stats.skewnorm
        
        self.trajectory_name = name
        self.start = None
        self.stop = None
        self.source = ''
        
        #   !!!! relex needs to be fixed
        self.spacecraft_name = self.source
        self.spacecraft_df = ''
        
        self.model_sources = {}
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
    
    @property
    def _secondary_df(self):
        return self._primary_df.query('@self.start <= index < @self.stop')
    
    @property
    def variables_unc(self):
        if self.distribution_function is not None:
            return [var + suffix for suffix in ['_a', '_loc', '_scale'] for var in self.variables]
        else:
            return [var+'_pos_unc' for var in self.variables] + [var+'_neg_unc' for var in self.variables]
    
    def _add_to_primary_df(self, label, df):
        
        #   Relabel the columns using MultiIndexing to add the label
        df.columns = pd.MultiIndex.from_product([[label], df.columns])
        
        if self._primary_df.empty:
            self._primary_df = df
        else:
            # new_keys = [*self._primary_df.columns.levels[0], label]
            self._primary_df = pd.concat([self._primary_df, df], axis=1, verify_integrity=True)
        
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
        
        #   Set the start and stop times based on the data, if not set
        if self.start is None:
            self.stop = self.data_index[0]
        if self.start is None:
            self.stop = self.data_index[-1]
    
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
        
    def addModel(self, model_name, model_df, model_source=''):
        self.model_names.append(model_name)
        #self.model_names = sorted(self.model_names)
        self.models = model_df
        if model_source != '':
            self.model_sources[model_name] = model_source
        
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
        model_names = ['huxt']
        
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
    
    def shift_Models(self, time_delta_column=None, time_delta_sigma_column=None, n_mc=10000, 
                     force_distribution=False, compare_distributions=False,
                     min_rel_scale = None):
        
        result = self._shift_Models(time_delta_column=time_delta_column, 
                                    time_delta_sigma_column=time_delta_sigma_column,
                                    n_mc=n_mc,
                                    force_distribution=force_distribution,
                                    compare_distributions=compare_distributions,
                                    min_rel_scale = min_rel_scale)
        self._primary_df = result
    
    def _shift_Models(self, time_delta_column=None, time_delta_sigma_column=None, n_mc=10000, 
                      force_distribution=False, compare_distributions=False,
                      min_rel_scale = 1e-11):
        """
        Shifts models by the specified column, expected to contain delta times in hours.
        Shifted models are only valid during the interval where data is present,
        and NaN outside this interval.
        
        This can also be used to propagate arrival time errors by supplying 
        time_delta_sigma with no time_delta (i.e., time_delta of zero)
        
        """
        import matplotlib.pyplot as plt
        import copy
        import time
        import scipy.stats as stats
        # import tqdm
        import multiprocessing as mp
        
        np.seterr(invalid='ignore')
        
        if min_rel_scale is None: min_rel_scale = 1e-11

        #   We're going to try to do this all with interpolation, rather than 
        #   needing to shift the model dataframe by a constant first
        
        shifted_primary_df = copy.deepcopy(self._primary_df)
        
        for model_name in self.model_names:
            
            #   temporarily drop all-nan rows from the model-specific df
            #   This prevents making needlessly large arrays
            # nonnan_row_index = ~self.models[model_name].isnull().all(axis=1)
            nonnan_model_df = self.models[model_name].dropna(how='all')
            
            try:
                #time_from_index = ((self.models[model_name].index - self.models[model_name].index[0]).to_numpy('timedelta64[s]') / 3600.).astype('float64')
                time_from_index = ((nonnan_model_df.index - nonnan_model_df.index[0]).to_numpy('timedelta64[s]') / 3600.).astype('float64')
            except:
                breakpoint()
                
            #   If no time_delta_column is given, then set it to all zeros
            if time_delta_column == None:
                # time_deltas = np.zeros(len(self.models[model_name]))
                time_deltas = np.zeros(len(nonnan_model_df))
            else:
                # time_deltas = self.models[model_name][time_delta_column].to_numpy('float64')
                time_deltas = nonnan_model_df[time_delta_column].to_numpy('float64')
            
            #   If no time_delta_sigma_column is given
            #   We don't want to do the whole Monte Carlo with zeros
            if time_delta_sigma_column == None:
                print("Setting time_delta_sigmas to zero")
                # time_delta_sigmas = np.zeros(len(self.models[model_name]))
                time_delta_sigmas = np.zeros(len(nonnan_model_df))
            else:
                print("Using real time_delta_sigmas")
                # time_delta_sigmas = self.models[model_name][time_delta_sigma_column].to_numpy('float64')
                time_delta_sigmas = nonnan_model_df[time_delta_sigma_column].to_numpy('float64')
            
            
            #if time_delta_sigma_column != None: breakpoint()
            #   Now we're shifting based on the ordinate, not the abcissa,
            #   so we have time_from_index - time_deltas, not +
            
            #   shift_function() gets redefined for every model, so the time_deltas above are the correct ones
            def shift_function(col, 
                               col_name='', model_name='', 
                               force_distribution=False, 
                               compare_distributions=False,
                               min_rel_scale = min_rel_scale):
                #   If the col is all NaNs, we can't do anything
                if ~np.isnan(col).all():
                
                    #   Use the ORIGINAL time_from_index with this, not time_from_index + delta
                    if (time_delta_sigmas == 0.).all():
                        col_shifted = np.interp(time_from_index - time_deltas, time_from_index, col)
                        #arr_shifted = np.interp(time_from_index, time_from_index+time_deltas, arr)
                        col_shifted_pos_unc = col_shifted * 0.
                        col_shifted_neg_unc = col_shifted * 0.
                        
                        output = (col_shifted, col_shifted_pos_unc, col_shifted_neg_unc)
                        
                        output = pd.DataFrame({'median': col_shifted,
                                               'pos_unc': col_shifted_pos_unc,
                                               'neg_unc': col_shifted_neg_unc})
                    else:
                        #   Randomly perturb the time deltas by their uncertainties (sigma) n_mc times
                        r = np.random.default_rng()
                        arr2d_perturb = r.normal(time_deltas, time_delta_sigmas, (n_mc, len(time_deltas)))
                        
                        #   For each of these random selections, sample the corresponding value of column
                        col2d_perturb = arr2d_perturb*0.
                        for i in range(n_mc):
                            #   !!!! Need to get rid of NaNs before doing this
                            
                            col2d_perturb[i,:] = np.interp(time_from_index - arr2d_perturb[i,:], time_from_index, col)
                        
                        col2d_perturb_originalsize = col2d_perturb
                        col2d_cast_indices = np.isfinite(time_deltas)
                        col2d_perturb = col2d_perturb[:, np.isfinite(time_deltas)]  #   This could filter the original df by self.start and self.stop
                        
                        def fit_dist_wmp(func, arr2d):
                            """
                            fit dist(ributions) w(ith) m(ulti)p(rocessing)
    
                            Parameters
                            ----------
                            func : function
                                scipy.stats.*.fit function, for any distribution
                            arr2d : n x n_mc np.array
                                2D array where n is iterated over
    
                            Returns
                            -------
                            results : n x m np.array
                                2D array of same length as arr2d, where m is the 
                                number of fitting parameters returned by func
    
                            """
                            
                            results = []
                            
                            #   Find out how many cores can be used
                            with mp.Pool(mp.cpu_count()) as pool:
                                #   Map the fitting function to each column in a generator
                                gen = pool.imap(func, [col for col in temp_col2d_perturbT])
                                #   Perform the fitting & print a progress bar
                                for entry in tqdm(gen, total=len(arr2d)):
                                    results.append(entry)
                                    
                            return np.array(results)
                                    
                        #   UNCERTAINTY CHARACTERIZATION !!!!
                        if force_distribution == True:
                            #+++++++++++++++++++++++++++++++++++++++++++++++
                            #   If all values in a column are the same, 
                            #       the fit will fail
                            #   If the fit fails in mp, the whole thing 
                            #       fails
                            #   So, filter out all same-value columns to 
                            #       treat separately
                            #-----------------------------------------------
                            usable_cols = np.array([~((row == row[0]) == True).all() for row in col2d_perturb.T])
                            
                            n_cols = len(col2d_perturb.T)
                            n_usable_cols = len(np.argwhere(usable_cols))
                            n_unusable_cols = n_cols - n_usable_cols
                            
                            temp_col2d_perturbT = col2d_perturb.T[usable_cols]
                            
                            uc_fit = np.zeros((n_cols, 3))
                            
                            #   Parallelized fitting
                            if n_usable_cols > 0:
                                
                                tqdm_desc = 'Characterizing uncertainties: {}|{}|{}'.format(self.trajectory_name, model_name, col_name)
                                mp_results = self._fit_DistributionToList_wmp(temp_col2d_perturbT, desc=tqdm_desc)
                                
                                #   Assign the parallelized results
                                uc_fit[usable_cols] = np.array(mp_results)
                                
                            #   Assign the unparallelized results
                            uc_fit[~usable_cols] = np.array([[0]*n_unusable_cols, np.mean(col2d_perturb.T[~usable_cols],1), [0]*n_unusable_cols]).T
                            
                        if compare_distributions == True:
                            #   Normal
                            norm_fits = fit_dist_wmp(stats.norm.fit, temp_col2d_perturbT)
                            norm_ks = [stats.kstest(col, stats.norm.cdf, args=args).statistic for col, args in zip(temp_col2d_perturbT, norm_fits)]
                            
                            #   Skew-Normal
                            skew_fits = fit_dist_wmp(stats.skewnorm.fit, temp_col2d_perturbT)
                            skew_ks = [stats.kstest(col, stats.skewnorm.cdf, args=args).statistic for col, args in zip(temp_col2d_perturbT, skew_fits)]
                            
                            # #   Beta
                            # #   Not currently supported, as it throws FitErrors
                            # beta_fits_list = fit_dist_wmp(stats.bera.fit, temp_col2d_perturbT)
                            # beta_ks = [stats.kstest(col, stats.beta.cdf, args=args) for col, args in zip(temp_col2d_perturbT, beta_fits_list)]
                            
                            #   Gamma
                            gamma_fits = fit_dist_wmp(stats.gamma.fit, temp_col2d_perturbT)
                            gamma_ks = [stats.kstest(col, stats.gamma.cdf, args=args).statistic for col, args in zip(temp_col2d_perturbT, gamma_fits)]
                            
                            #   Save these distributions + ks test
                            #breakpoint()
                            
                            # breakpoint()
                            # import matplotlib.pyplot as plt
                            fig, axs = plt.subplots(nrows = 3)
                            x0 = np.linspace(0.0, 0.1, 50)
                            x1 = np.linspace(0.0, 1.0, 50)
                            x2 = np.linspace(0.9, 1.0, 50)
                            
                            for name, dist in {'Normal':norm_ks, 'Skew-Normal':skew_ks, 'Gamma':gamma_ks}.items():
                                axs[0].hist(dist, bins=x0, label=name, linewidth=2, histtype='step')
                                axs[1].hist(dist, bins=x1, label=name, linewidth=2, histtype='step')
                                axs[2].hist(dist, bins=x2, label=name, linewidth=2, histtype='step')
                            for ax in axs:
                                ax.legend(loc='upper right')
                            
                            fig.suptitle('Mode: {}, Variable: {}'.format(model_name, col_name))
                            fig.supxlabel('K-S Statistic')
                            fig.supylabel('Number of Fits')
                            
                            plt.show()
                            
                        
                        #   Return to normally scheduled programming
                        #   Take the 16th, 50th, and 84th percentiles of the random samples as statistics                
                        # col_16p, col_50p, col_84p = np.percentile(col2d_perturb, [16,50,84], axis=0, overwrite_input=True)
                        
                        # col_shifted = col_50p
                        # col_shifted_pos_unc = col_84p - col_50p
                        # col_shifted_neg_unc = col_50p - col_16p
                        
                        #   Force a minimum for the relative scale;
                        #   By default, this 1e-11 (a billionth of 1%)
                        #   Helps prevent errors with overly-peaked distributions
                        #   !!!!! This could alternatively take the form of a small 
                        #   additional scale added in quadrature to each term, regardless of rel. scale
                        scale_rel_to_loc = uc_fit.T[2] / uc_fit.T[1]
                        lt_min_indx = scale_rel_to_loc < min_rel_scale
                        
                        if (lt_min_indx == True).any():
                            uc_fit.T[2][lt_min_indx] = uc_fit.T[1][lt_min_indx] * min_rel_scale
                        
                        # output = (col_shifted, col_shifted_pos_unc, col_shifted_neg_unc,
                        #           uc_fit.T[0], uc_fit.T[1], uc_fit.T[2])
                        uc_fit_withpadding = np.zeros((len(col), 3))
                        uc_fit_withpadding[col2d_cast_indices, :] = uc_fit
                        

                        
                        
                        output = pd.DataFrame({'a': uc_fit_withpadding.T[0],
                                               'loc': uc_fit_withpadding.T[1],
                                               'scale': uc_fit_withpadding.T[2],
                                               'median': [self.distribution_function.median(*params) for params in uc_fit_withpadding]})
                else:
                    output = pd.DataFrame({'a': [np.nan] * len(col),
                                           'loc': [np.nan] * len(col),
                                           'scale': [np.nan] * len(col),
                                           'median': [np.nan] * len(col)})
                        
                return output
            
            # def test_shift_function(arr):
            #     #   Use the ORIGINAL time_from_index with this, not time_from_index + delta
            #     if (time_delta_sigmas == 0.).all():
            #         arr_shifted = np.interp(time_from_index - time_deltas, time_from_index, arr)
            #         arr_shifted_sigma = arr_shifted * 0.
            #     else:
            #         arr_shifted = np.interp(time_from_index - time_deltas, time_from_index, arr)
                    
            #         arr_shifted_plus = np.interp(time_from_index - (time_deltas + time_delta_sigmas), time_from_index, arr)
            #         arr_shifted_minus = np.interp(time_from_index - (time_deltas - time_delta_sigmas), time_from_index, arr)
                    
            #         (arr_shifted_plus - arr_shifted_minus)
            #     output = (arr_shifted, arr_shifted_sigma)
            #     return output
            
            len_items = len(nonnan_model_df.columns)
            for col_name, col in nonnan_model_df.items():
                #   Skip columns of all NaNs and columns which don't appear
                #   in list of tracked variables
                #if len(col.dropna() > 0) and (col_name in self.variables):
                if (col_name in self.variables):
                    
                    col_shifted = shift_function(col.to_numpy('float64'), 
                                                 col_name=col_name, model_name=model_name,
                                                 force_distribution=force_distribution,
                                                 compare_distributions=compare_distributions)
                    
                    col_shifted_df = col_shifted.set_index(nonnan_model_df.index)
                    
                    c_name_mapper = dict()
                    c_names = list(col_shifted_df.columns)
                    for c_name in c_names:
                        if c_name == 'median':
                            c_name_mapper[c_name] = col_name
                        else:
                            c_name_mapper[c_name] = col_name + '_' + c_name
                    
                    col_shifted_df = col_shifted_df.rename(c_name_mapper, axis='columns')
                    
                    for c_name, c in col_shifted_df.items():
                        shifted_primary_df.loc[:, (model_name, c_name)] = c
                    # shifted_primary_df.loc[nonnan_row_index, (model_name, col_name)] = col_shifted[0]
                    # shifted_primary_df.loc[nonnan_row_index, (model_name, col_name+'_pos_unc')] = col_shifted[1]
                    # shifted_primary_df.loc[nonnan_row_index, (model_name, col_name+'_neg_unc')] = col_shifted[2]
                    
        return shifted_primary_df
                        
    def _fit_DistributionToList_wmp(self, list_of_data, desc=''):
        """
        fit dist(ributions) w(ith) m(ulti)p(rocessing)
    
        Parameters
        ----------
        func : function
            scipy.stats.*.fit function, for any distribution
        arr2d : n x n_mc np.array
            2D array where n is iterated over
    
        Returns
        -------
        results : n x m np.array
            2D array of same length as arr2d, where m is the 
            number of fitting parameters returned by func
    
        """
        import multiprocessing as mp
        
        results = []
        
        #   Find out how many cores can be used
        func = self.distribution_function.fit
        
        with mp.Pool(mp.cpu_count()) as pool:
            
            #   Map the fitting function to each entry in the list in a generator
            gen = pool.imap(func, [entry for entry in list_of_data])
            #   Perform the fitting & print a progress bar
            for entry in tqdm(gen, total=len(list_of_data), desc=desc):
                results.append(entry)
                
        return np.array(results)

    # def sample(self, model_names = [], variables = [], n_samples=1, mu=False):
    #     """
        

    #     Parameters
    #     ----------
    #     model_name : TYPE, optional
    #         DESCRIPTION. The default is [].
    #     column_ref : TYPE, optional
    #         DESCRIPTION. The default is [].
    #     n_samples : TYPE, optional
    #         DESCRIPTION. The default is 1.

    #     Returns
    #     -------
    #     None.

    #     """
    #     import scipy.stats as stats
        
    #     model_names = list(model_names)
    #     variables = list(variables)
        
    #     if len(model_names) == 0:
    #         model_names = self.model_names
    #     if len(variables) == 0:
    #         variables= self.variables
        
    #     for i_sample in range(n_samples): 
            
    #         #   Make an empty dataframe to hold the sample
    #         multiindex = pd.MultiIndex.from_product([self.model_names,
    #                                                  self.variables])
    #         sample_df = pd.DataFrame(index = self.models.index,
    #                                  columns = multiindex)
    #         for model_name in model_names:
    #             for variable in variables:
                    
    #                 #   For this variable, get the names of parameters describing the distribution
    #                 param_names = self.models[model_name].columns[self.models[model_name].columns.str.contains(variable)].to_list()
    #                 param_names.remove(variable)
                    
    #                 #   Now get the parameters themselves
    #                 params = self.models[model_name].loc[:, param_names]
                    
    #                 #
    #                 def apply_fn(row):
    #                     return stats.skewnorm.mean(*row.to_numpy())
                    
    #                 sample_df.loc[:, (model_name, variable)] = params.apply(apply_fn, axis='columns')
                    
    #         breakpoint()
                
    #     return
    
    def weights(self, how='default'):
        
        #   Create a new DataFrame with all model-variable combinations
        weights = self._primary_df.loc[:, pd.IndexSlice[:, self.variables]].copy(deep=True)
        #   And then empty it
        for col in weights.columns:
            weights.loc[:, col] = 1
            
        if how == 'default':
            #   By default, weights are 1 unless that corresponding model has NaNs, in which case its 0
            for model_name in self.model_names:
                for variable in self.variables:
                    isnan_mask = np.isnan(self.models[model_name][variable].to_numpy())
                    weights.loc[isnan_mask, (model_name, variable)] = 0
        
        return weights
    
    def ensemble(self, weights = None, as_skewnorm = True):
        """
        Produces an ensemble model from inputs.
        
        The stickiest part of this is considering the weights:
            Potentially, each model, parameters, and timestep could have a unique weight
            Model Weights: account for general relative performance of models
            Parameters Weights: account for better performance of individual parameters within a model
                i.e., Model 1 velocity may outperform Model 2, but Model 2 pressure may outperform Model 1
            Timestep Weights: Certain models may relatively outperform others at specific times. 
                Also allows inclusion of partial models, i.e., HUXt velocity but with 0 weight in e.g. pressure

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
        
        #   Ignores math warnings in numpy that clutter the terminal
        np.seterr(invalid='ignore')
        
        #   Initialize an empty ensmeble data frame
        ensemble = pd.DataFrame(index = self._primary_df.index)
        
        if weights == None:
            weights = self.weights()
        
        if as_skewnorm:
            
            for variable in self.variables:
                
                #   !!!!!!
                #   THIS SECTION NEEDS TO BE REWRITTEN
                #   x_pdf should vary per row
                #   So, x_pdf is a list of (potentially) unequal lengths... although can force m_pdf length too
                #   variable_2d_LIST is a LIST where each entry is the same len as corresponding entry in x_pdf
                #   Loop over the same way, multiplying pdf to build up, etc.
                #   Doing this should prevent overly-narrow distributions in the fake histograms
                
                #   Estimate the bounds of this variable from the minimum and maximum of the 95% confidence interval
                var_mins, var_maxs = [], []
                for model_name in self.model_names:
                    
                    dist_params = self._secondary_df[model_name].loc[:, [variable+'_a', variable+'_loc', variable+'_scale']].to_numpy()
                    
                    interval95 = np.array([np.percentile(self.distribution_function.rvs(*dist_param, size=1000), (2.5, 97.5)) if (~np.isnan(dist_param)).all() else [np.nan, np.nan] for dist_param in dist_params])
                    
                    # interval95 = np.array([self.distribution_function.interval(0.95, *dist_param) for dist_param in dist_params])
                    
                    # #   The skew-normal distribution can cause errors in the interval function, causing the right hand bound to go negative
                    # #   These are not physical-- simply NaN them
                    # if (interval95[:,1] < 0).any():
                    #     interval95[interval95[:,1] < 0, 1] = np.nan
                    
                    var_mins.append(interval95.T[0])
                    var_maxs.append(interval95.T[1])
                
                #   np.perceniles prevents single large or small numbers form biasing these
                # var_min, var_max = np.nanpercentile(dynamic_range, (2.5, 97.5))
                
                #   All models are going to be on the same interpolated axis, 
                #   so make sure the min and max are sufficient for all expected values
                x_min_list = np.nanmin(var_mins, 0)
                x_max_list = np.nanmax(var_maxs, 0)
                
                #   n x m numpy array, where:
                #   n = length of _primary_df
                #   m = length of values sampled over, arbitrary
                m_pdf = 1000
                # x_pdf = np.linspace(var_min, var_max, m_pdf)
                x_pdf_list = [np.linspace(l, r, m_pdf) for l, r in zip(x_min_list, x_max_list)]
                
                variable_list = [np.zeros(m_pdf) + 1 for i in range(len(self._secondary_df))]
                
                for model_name in self.model_names:
                
                    variable_params_withpadding_df = self.models[model_name].filter(like=variable, axis='columns')
                    variable_params_withpadding_df.drop(variable, axis='columns', inplace=True)
                    variable_params_df = variable_params_withpadding_df.query('@self.start <= index < @self.stop')
                    
                    variable_model_weights = weights.query('@self.start <= index < @self.stop').loc[:, (model_name, variable)].to_numpy()
                    
                    #   If all weights are zero here, skip the pdf generation
                    if ~(variable_model_weights == 0.).all():
                        for i in range(len(self._secondary_df)):
                            #
                            pdf = self.distribution_function.pdf(x_pdf_list[i], *variable_params_df.iloc[i].to_numpy())
                            
                            #   Multiply the resulting pdf by the correct weights
                            pdf *= variable_model_weights[i]
                            
                            #   Where the pdf is NaN, it shouldn't change the values of variable_list
                            isnan_mask = np.isnan(pdf)
                            if (isnan_mask == True).any():
                                pdf[isnan_mask] = 1
                            
                            #   THIS SHOULDNT HAPPEN
                            if (pdf == 0).all() == True:
                                breakpoint()
                            
                            variable_list[i] *= pdf
                            
                variable_poorly_constrained_mask = [(v == 0).all() for v in variable_list]
                
                if (np.array(variable_poorly_constrained_mask) == True).any():
                    #   Simulate a broad pdf, centered at the mean and 
                    #   with non-zero density at all extrema
    
                    breakpoint()
                
                #   The scale factor rescales the pdf such that the sum is ~m_pdf
                #   This should yield histograms in the next step with reasonable samples
                # scale_factor = 1/np.sum(variable_2d_arr, axis=1) * m_pdf
                scale_factor_list = [1/np.sum(v) * m_pdf for v in variable_list]
                
                #   This multiplies the correct dimensions by scale_factor
                # simulated_histogram2d = np.int64(variable_2d_arr * scale_factor[:, np.newaxis])
                simulated_histogram_list = [np.int64(v * sf) for v, sf in zip(variable_list, scale_factor_list)]
                
                try:
                    simulated_data = [np.repeat(x, h) for x, h in zip(x_pdf_list, simulated_histogram_list)]
                except:
                    breakpoint()
                
                try:
                    fit_list = self._fit_DistributionToList_wmp(simulated_data)
                except:
                    breakpoint()
                
                #   Calculate the median of the pdf and set it as the titular value
                ensemble.loc[self._secondary_df.index, variable] = [self.distribution_function.median(*f) for f in fit_list]
                
                #   Report the a (skew), loc, and shape parameters as well
                ensemble.loc[self._secondary_df.index, variable + '_a'] = fit_list.T[0]
                ensemble.loc[self._secondary_df.index, variable + '_loc'] = fit_list.T[1]
                ensemble.loc[self._secondary_df.index, variable + '_scale'] = fit_list.T[2]
            
        else:
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
            breakpoint()
            #   Sum weighted variables, and RSS weighted errors
            for partial in partials_list:
                ensemble.loc[:, self.variables] += partial.loc[:, self.variables].astype('float64')
                ensemble.loc[:, variables_unc] += (partial.loc[:, variables_unc].astype('float64'))**2.
                
            ensemble.loc[:, variables_unc] = np.sqrt(ensemble.loc[:, variables_unc])
            #   !!!! This doesn't add error properly
            # ensemble = reduce(lambda a,b: a.add(b, fill_value=0.), partials_list)
        
        
        self.addModel('ensemble', ensemble, model_source='MME')
    
    # =============================================================================
    #     Statistical convenience functions   
    # =============================================================================
    def _parse_stats_inputs(self, model_names, variables):
        model_names = self.model_names if model_names is None else model_names
        model_names = [model_names] if isinstance(model_names, str) else model_names
        
        variables = self.variables if variables is None else variables
        variables = [variables] if isinstance(variables, str) else variables
        
        return model_names, variables
    
    def _scipy_stats_method_apply(self, method, model_names, variables):
        from pandas import IndexSlice as idx
        
        #   This falsely 
        pd.options.mode.chained_assignment = None  # default='warn'
        
        model_names, variables = self._parse_stats_inputs(model_names, variables)
        
        #   Create a df to hold the output and set all to 0
        output_df = self._primary_df.loc[:, idx[model_names, variables]]
        for col in output_df.columns:
            output_df.loc[:, col] = np.nan
        
        for model_name in model_names:
            for variable in variables:
                #   This makes sure that all the distribution params are ordered in the expected way
                dist_param_names = [variable + suffix for suffix in ['_a', '_loc', '_scale']]
                df = self.models[model_name].loc[:, dist_param_names]
                
                #   Only take the defined bits
                defined_indx = [~np.isnan(row).all() for _, row in df.iterrows()]
                dist_params = df.loc[defined_indx, :].to_numpy()
                    
                statistics = [method(*dist_param) for dist_param in dist_params]
                statistics = np.array(statistics).flatten()
                
                output_df.loc[defined_indx, idx[model_name, variable]] = statistics
        
        return output_df
    
    def median(self, model_names=None, variables=None):
        
        method = self.distribution_function.median
        result = self._scipy_stats_method_apply(method, model_names, variables)
        
        return result
    
    def ci(self, percentile=50, model_names=None, variables=None):
        
        #   Percent Point Function needs an extra argument, 
        #   so define a new function
        def method(*args):
            return self.distribution_function.ppf(percentile/100, *args)
        result = self._scipy_stats_method_apply(method, model_names, variables)
        
        return result
    
    def mean(self, model_names=None, variables=None):
        
        method = self.distribution_function.mean
        result = self._scipy_stats_method_apply(method, model_names, variables)
        
        return result
    
    def sigma(self, model_names=None, variables=None):
        
        method = self.distribution_function.std
        result = self._scipy_stats_method_apply(method, model_names, variables)
        
        return result
    
    def nsigma(self, n=1, model_names=None, variables=None):
        
        mu = self.mean(model_names, variables)
        sigma = self.sigma(model_names, variables)
        
        return mu + n * sigma
        
    def sample(self, model_names=None, variables=None, size=1):
        
        def method(*args):
            return self.distribution_function.rvs(*args, size=1)
        
        #   Consider looping over this to sample more?
        result = []
        for i in tqdm(range(size)):
            result.append(self._scipy_stats_method_apply(method, model_names, variables))
    
        if size == 1:
            result = result[0]
        
        return result
    
    def taylor_statistics(self, model_names=None, variables=None, with_error=True):
        import plot_TaylorDiagram as TD
        
        model_names, variables = self._parse_stats_inputs(model_names, variables)
        
        if with_error:
            index = ['r_mu', 'r_sigma', 'sig_mu', 'sig_sigma', 'rmsd_mu', 'rmsd_sigma']
            n_mc = 100
        else:
            index = ['r', 'sig', 'rmsd']
            n_mc = 1
        
        output_cols = pd.MultiIndex.from_product([model_names, variables])
        output_df = pd.DataFrame(index=index,
                                 columns=output_cols)
        
        #   Sample the df, making sure its a list
        sampled_dfs = self.sample(size = n_mc)
        if not with_error:
            sampled_dfs = [sampled_dfs]
        
        for variable in variables:
            
            #   Drop all NaN rows, *unless* the whole column is NaN
            #   This ensures the corr. coeffs. can be compared between models
            usable_cols = self.model_names
            for col in usable_cols:
                if self.models[col][variable].isnull().all():
                    usable_cols.remove(col)
            
            #   Get a df only looking at variable
            isolated_variable_dfs = [df.xs(variable, axis='columns', level=1) for df in sampled_dfs]
            
            isolated_variable_dfs = [df.dropna(axis = 'index', how = 'any', subset = usable_cols) for df in isolated_variable_dfs]
            
            for model_name in model_names:
                
                isolated_model_dfs = [df[model_name] for df in isolated_variable_dfs]
                
                r_list, sig_list, rmsd_list = [], [], []
                for i, df in enumerate(isolated_model_dfs):
                    
                    #   Squeeze forces this to be a series, so can be directly compared to spacecraft data
                    (r, sig), rmsd = TD.find_TaylorStatistics(test_data = df.squeeze(),
                                                              ref_data = self.data.loc[df.index, variable])
                    
                    r_list.append(r)
                    sig_list.append(sig)
                    rmsd_list.append(rmsd)
                        
                        
                if with_error:
                    
                    output_df.loc['r_mu', (model_name, variable)] = np.mean(r_list)
                    output_df.loc['r_sigma',  (model_name, variable)] = np.std(r_list)
                    output_df.loc['sig_mu', (model_name, variable)] = np.mean(sig_list)
                    output_df.loc['sig_sigma', (model_name, variable)] = np.std(sig_list)
                    output_df.loc['rmsd_mu', (model_name, variable)] = np.mean(rmsd_list)
                    output_df.loc['rmsd_sigma', (model_name, variable)] = np.std(rmsd_list)
                
                else:
                    
                    output_df.loc['r', (model_name, variable)] = r_list[0]
                    output_df.loc['sig', (model_name, variable)] = sig_list[0]
                    output_df.loc['rmsd', (model_name, variable)] = rmsd_list[0]
            
            
                
        breakpoint()
        
        return output_df
    
        
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