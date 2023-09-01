#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:38:04 2023

@author: mrutala
"""

import datetime as dt
import logging

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
class SolarWindData:
    
    def __init__(self):
        
        self.metric_tags = ['u_mag', 'p_dyn', 'n_tot', 'B_mag']
        
        self.spacecraft_name = ''
        self.spacecraft_df = ''
        
        self.model_names = []
        self.model_dfs = {}
        
    def addData(self, spacecraft_name, spacecraft_df):
        
        self.spacecraft_name = spacecraft_name
        self.spacecraft_df = spacecraft_df
        
    def addModel(self, model_name, model_df):
        
        self.model_names.append(model_name)
        self.model_dfs[model_name] = model_df

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
        
    def warp(self, basis_tag, metric_tag):
        import sys
        import DTW_Application as dtwa
        shifts = [0]
        
        for model_name, model_df in self.model_dfs.items():
            for shift in shifts:
                test = dtwa.find_SolarWindDTW(self.spacecraft_df, model_df, shift, 
                                         basis_tag, metric_tag, 
                                         total_slope_limit=96.0,
                                         intermediate_plots=True)
                return(test)
