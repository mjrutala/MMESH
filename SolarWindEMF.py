#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:38:04 2023

@author: mrutala
"""

import datetime as dt
import logging


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

class SolarWindEM:
    
    def __init__(self):
        
        #self.startdate = startdate
        #self.stopdate = stopdate
        
        self.spacecraft_names = []
        self.spacecraft_dict = {}
        # self.spacecraft_dictionary_template = {'spacecraft_name': '',
        #                                        'spacecraft_data': 0.,
        #                                        'models_present': [],
        #                                        'model_outputs': []}
    
    def addData(self, spacecraft_name, spacecraft_df):
        
        self.spacecraft_names.append(spacecraft_name)
        
        
        self.spacecraft_dict[spacecraft_name] = {'spacecraft_data': spacecraft_df,
                                                         'models': {}}
        
    def addModel(self, model_name, model_df, spacecraft_name = 'Previous'):
        
        #  Parse spacecraft_name
        if (spacecraft_name == 'Previous') and (len(self.spacecraft_dictionaries) > 0):
            spacecraft_name = self.spacecraft_dictionaries.keys()[-1]
        elif spacecraft_name == 'Previous':
            logging.warning('No spacecraft loaded yet! Either specify a spacecraft, or load spacecraft data first.')
               
        self.spacecraft_dict[spacecraft_name]['models'][model_name] = model_df


    def baseline(self):
        
        for key, value in self.spacecraft_dict.items():
            print()
            
            
    def warp(self, basis_tag):
        
        for spacecraft_name, d in self.spacecraft_dict.items():
            
            for model_name, model_output in d['models'].items():
                
                print()
                
                


#  If the nested dictionaries used in SolarWindEM become a headache
#  I could always spin off a new class to hold the data and models
#  This would be kinda nice, because it could have built in plotting routines and the like
# class DataModel:
#     def __init__(self):

    
#  There should be a model_output class, which has methods to warp it
import pandas as pd

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
        self.model_dtw_stats = {}
        self.model_dtw_times = {}
        
        #self.trajectory = pd.Dataframe(index=index)
        
        self._data = None
        self._master_df = pd.DataFrame()
        
    @property
    def trajectory(self):
        return self._trajectory
    
    @trajectory.setter
    def trajectory(self, df):
        self._trajectory = df.upper()

    @property
    def data(self):
        return self._master_df[self.spacecraft_name]
    
    @data.setter
    def data(self, df):
        
        #  Find common columns between the supplied DataFrame and internals
        int_columns = list(set(df.columns).intersection(self.variables))
        df = df[int_columns]
        
        if self._master_df.empty:
            df.columns = pd.MultiIndex.from_product([[self.spacecraft_name], df.columns])
            self._master_df = df
        else:
            new_keys = [self._master_df.columns.levels[0], self.spacecraft_name]
            self._master_df = pd.concat([self._master_df, df], axis=1,
                                        keys=new_keys)
    
    def addData(self, spacecraft_name, spacecraft_df):
        self.spacecraft_name = spacecraft_name
        self.data = spacecraft_df
    
    @property
    def models(self):
        return self._master_df[self.model_names]
    
    @models.setter
    def models(self, df):
        int_columns = list(set(df.columns).intersection(self.variables))
        df = df[int_columns]
        
        if self._master_df.empty:
            new_keys = [self.model_names[-1]]
        else:

            new_keys = [*self._master_df.columns.levels[0], self.model_names[-1]]
            df.columns = pd.MultiIndex.from_product([[self.model_names[-1]], df.columns])

        self._master_df = pd.concat([self._master_df, df], axis=1)
    
    def addModel(self, model_name, model_df):
        self.model_names.append(model_name)
        self.models = model_df

    def baseline(self):
        import scipy
        import numpy as np
        import pandas as pd
        
        models_stats = {}
        for model_name, model_df in self.model_dfs.items():
            
            model_stats = {}
            for m_tag in self.metric_tags:
                
                #  Find overlap between data and this model, ignoring NaNs
                i1 = self.spacecraft_df.dropna(subset=[m_tag]).index
                i2 = model_df.dropna(subset=[m_tag]).index
                intersection_indx = i1.intersection(i2) 
                
                if len(intersection_indx) > 2:
                    #  Calculate r (corr. coeff.) and both standard deviations
                    r, pvalue = scipy.stats.pearsonr(self.spacecraft_df[m_tag].loc[intersection_indx],
                                             model_df[m_tag].loc[intersection_indx])
                    model_stddev = np.std(model_df[m_tag].loc[intersection_indx])
                    spacecraft_stddev = np.std(self.spacecraft_df[m_tag].loc[intersection_indx])
                else:
                    r = 0
                    model_stddev = 0
                    spacecraft_stddev = 0
                
                model_stats[m_tag] = (spacecraft_stddev, model_stddev, r)
                
            models_stats[model_name] = model_stats
        
        return (models_stats)
    
    def shift(self, basis_tag, metric_tag, shifts=[0]):
        
        return
        
    def warp(self, basis_tag, metric_tag, shifts=[0]):
        import sys
        import DTW_Application as dtwa
        import SWData_Analysis as swda
        import numpy as np
        import pandas as pd
        
        #shifts = [0]  #  Needs to be input
        sigma_cutoff = 3  #  Needs to be input
        smoothing_time_spacecraft = 2.0  #  Needs to be input
        smoothing_time_model = 2.0  #  Needs to be input
        
        resolution_width_spacecraft = 0.0  #  Input?
        resolution_width_model = 0.0  #  Input?
        
        self.spacecraft_df = swda.find_Jumps(self.spacecraft_df, 'u_mag', 
                                             sigma_cutoff, 
                                             smoothing_time_spacecraft, 
                                             resolution_width=resolution_width_spacecraft)
        
        for model_name, model_df in self.model_dfs.items():
            
            model_df = swda.find_Jumps(model_df, 'u_mag', 
                                       sigma_cutoff, 
                                       smoothing_time_model, 
                                       resolution_width=resolution_width_model)
            
            stats = []
            time_deltas = []
            for shift in shifts:
                stat, time_delta = dtwa.find_SolarWindDTW(self.spacecraft_df, model_df, shift, 
                                                          basis_tag, metric_tag, 
                                                          total_slope_limit=96.0,
                                                          intermediate_plots=True,
                                                          model_name = model_name,
                                                          spacecraft_name = self.spacecraft_name)
                time_delta = time_delta.rename(columns={'time_lag': str(shift)})
                
                stats.append(pd.DataFrame.from_dict(stat))
                time_deltas.append(time_delta)
            
            stats_df = pd.concat(stats, axis='index', ignore_index=True)
            times_df = pd.concat(time_deltas, axis='columns')
            self.model_dtw_stats[model_name] = stats_df
            self.model_dtw_times[model_name] = times_df
            
        return
    
    def ensemble(self, weights = None):
        import pandas as pd
        
        weights_df = pd.DataFrame(columns = self.em_parameters)
        
        if weights == None:
            for model in self.model_names:
                all_nans = ~self.model_dfs[model].isna().all()
                overlap_col = list(set(all_nans.index) & set(weights_df.columns))
                weights_df.loc[model] = all_nans[overlap_col].astype(int)
                weights_df = weights_df.div(weights_df.sum())
        
        # if dict, use dict
        # if list/array, assign to dict in order
        # if none, make dict with even weights
        
        self.ensemble = pd.DataFrame(columns = self.em_parameters)
        
        for model, model_df in self.model_dfs.items(): print(len(model_df))
        
        for tag in self.em_parameters:
            for model, model_df in self.model_dfs.items():
                
                self.ensemble[tag] = weights_df[tag][model] * model_df[tag]
        
        print(weights_df.div(weights_df.sum()))
        
        
    # =============================================================================
    #   Plotting stuff
    # =============================================================================
    def _plot_parameters(self, parameter):
        import numpy as np
        
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
