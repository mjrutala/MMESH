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
                                  comment='#', delim_whitespace=True)
        
        file_data['datetime'] = [file_starttime + dt.timedelta(days=time) for time in file_data['time']]
        
        #   Write this file back out to a CSV
        file_data.to_csv(new_dir + filename[0:-4]+'.csv', sep=',', index=False)
    
    return

def convert_toCSV_TaoFromAMDA():
    
    return