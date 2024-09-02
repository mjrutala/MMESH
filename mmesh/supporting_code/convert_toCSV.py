#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:04:15 2024

@author: mrutala
"""
import glob
import datetime as dt
import pandas as pd
from tqdm import tqdm
import numpy as np

def convert_toCSV_CCMCENLIL(target_dir='', new_dir=''):
    """
    Create new .csv files with datetimes from CCMC ENLIL printouts

    Parameters
    ----------
    targetdir : TYPE, optional
        DESCRIPTION. The default is ''.
    newdir : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    """
    
    filenames = [g.split('/')[-1] for g in glob.glob(target_dir + '*.*')]
    
    column_headers = ['time', 'R', 'Lat', 'Lon', 
                      'u_r', 'u_t', 'u_n', 
                      'B_r', 'B_t', 'B_n', 
                      'n_proton', 'T_proton', 'E_r', 'E_t', 'E_n', 
                      'u_mag', 'B_mag', 'p_dyn', 'BP']
    
    for filename in tqdm(sorted(filenames)):
        
        #  In these files, the start date is hidden in a comment, so read the first few lines
        header = []
        with open(target_dir + filename) as f:
            for indx, line in enumerate(f):
                header.append(line.rstrip())
                if indx == 6:
                    break
        for line in header:
            if 'Start Date' in line:
                file_starttime = dt.datetime.strptime(line.split(': ')[1], '%Y/%m/%d %H:%M:%S')
        
        file_data = pd.read_table(target_dir + filename, 
                                  names=column_headers, 
                                  comment='#', sep='\s+')
        
        file_data['datetime'] = [file_starttime + dt.timedelta(days=time) for time in file_data['time']]
        
        #   Write this file back out to a CSV
        file_data.to_csv(new_dir + filename[0:-4]+'.csv', sep=',', index=False)
    
    return

def convert_toCSV_TaoToAMDA(target_dir='', new_dir=''):
    """
    Convert AMDA-formatted Tao+ .csv to different format

    Parameters
    ----------
    target_dir : TYPE, optional
        DESCRIPTION. The default is ''.
    new_dir : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    """
    
    filenames = [g.split('/')[-1] for g in glob.glob(target_dir + '*.*')]
    
    column_headers = ['iyear', 'imonth', 'iday', 'stime', 'n_proton', 'T_proton', 'u_r', 'u_t', 'B_t', 'p_dyn_proton', 'angle_EST', 'data_quality']
    new_column_headers = ['datetime', 'n_proton', 'u_r', 'u_t', 'T_proton', 'p_dyn_proton', 'B_t', 'B_r', 'angle_EST']
    
    for filename in tqdm(sorted(filenames)):
        
        file_data = pd.read_table(target_dir + filename, 
                                  names=column_headers, 
                                  comment='#', sep='\s+',
                                  parse_dates = {'datetime': [0,1,2,3]})
        file_data['B_r'] = np.NaN
        file_data = file_data[new_column_headers]
        
        if file_data.loc[:, 'B_r'].isnull().all():
            file_data.loc[:, 'B_r'] = 0.0
            
        #   These files don't have p_dyn written correctly, so re-calculate
        file_data['p_dyn_proton'] = (1.67e-27) * file_data['n_proton'] * (1e6) * (file_data['u_r']**2 + file_data['u_t']**2) * (1e3)**2 * 1e9
    
        #   Write this file back out to a CSV
        newfile = new_dir + filename[0:-4]+'.txt'

        
        with open(newfile, 'w') as f:
            f.write('#')
         
        file_data.to_csv(new_dir + filename[0:-4]+'.txt', sep=' ', index=False, mode='a', date_format='%Y-%m-%dT%H:%M:%S.%f')
    
    return