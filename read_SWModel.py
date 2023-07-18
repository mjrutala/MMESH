# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:09:36 2023

@author: Matt Rutala
"""

"""
target = string, startdate & finaldata = datetime
"""
import pandas as pd
import numpy as np
import datetime as dt
import copy
import logging

from pathlib import Path

m_p = 1.67e-27
kg_per_amu = 1.66e-27 # 

default_df = pd.DataFrame(columns=['u_mag', 'n_tot', 'p_dyn', 'B_mag', 
                                   'u_r', 'u_t', 'u_n', 
                                   'n_proton', 'n_alpha',
                                   'p_dyn_proton', 'p_dyn_alpha',
                                   'T_proton', 'T_alpha',
                                   'B_r', 'B_t', 'B_n', 'B_pol'])

def Tao(target, starttime, finaltime, basedir=''):
    
    #  basedir is expected to be the folder /SolarWindEM
    target = target.lower().replace(' ', '')
    fullpath = Path(basedir) / 'models' / 'tao' / target
    
    match target:
        # case 'mars':
        #     return(None)
        # case 'jupiter':
        #     return(None)
        # case 'saturn':
        #     return(None)
        case 'voyager1':
            filenames = ['swmhd_tao_voyager1_1978.txt',
                         'swmhd_tao_voyager1_1979.txt']    
        case 'voyager2':
            filenames = ['swmhd_tao_voyager2_1978.txt',
                         'swmhd_tao_voyager2_1979.txt',
                         'swmhd_tao_voyager2_1980.txt']
        # case 'galileo':
        #     return(None)
        case 'ulysses':
            filenames = ['swmhd_tao_ulysses_1991.txt',
                         'swmhd_tao_ulysses_1992.txt',
                         'swmhd_tao_ulysses_1998.txt',
                         'swmhd_tao_ulysses_1999.txt',
                         'swmhd_tao_ulysses_2003.txt',
                         'swmhd_tao_ulysses_2004.txt']
        case 'newhorizons':
            filenames = ['swmhd_tao_newhorizons_2006-2007.txt']
        case 'juno':
            filenames = ['fromamda_tao_juno_2015.txt',
                         'fromamda_tao_juno_2016.txt']
        case _:
            logging.warning('This version of the Tao SWMHD reader'
                            'does not support ' + target + ' observations.')
            return(default_df)
        
    #  This should put files in chronological order, useful for checking overalaps in the data
    filenames.sort()
    
    data = default_df
    
    for filename in filenames:
        #  Files downloaded from AMDA have a different order of columns than
        #  those directly from Chihiro Tao
        if 'fromamda' in filename:
            column_headers =  ['datetime', 
                               'n_proton', 'u_r', 'u_t', 'T_proton', 
                               'p_dyn_proton', 'B_t', 'B_r', 'angle_EST']
            def column_to_datetime(df):
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
                return(df)
            
        else:
            column_headers = ['iyear', 'imonth', 'iday', 'stime',
                              'n_proton', 'T_proton', 'u_r', 'u_t',
                              'B_t', 'p_dyn_proton',
                              'angle_EST', 'data_quality']
            def column_to_datetime(df):
                df['datetime'] = [dt.datetime(row['iyear'], 
                                              row['imonth'], 
                                              row['iday'], 
                                              dt.datetime.strptime(row['stime'], '%H:%M:%S').hour)
                                  for indx, row in df.iterrows()]
                df.drop(columns=['iyear', 'imonth', 'iday', 'stime'], inplace=True)
                return(df)
            
        file_data = pd.read_table(fullpath / filename, 
                             names=column_headers, 
                             comment='#', delim_whitespace=True)
        file_data = column_to_datetime(file_data) 
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < finaltime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
        
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2)
    data['B_mag'] = np.sqrt(data['B_r'].replace(np.nan, 0)**2 + data['B_t']**2)
    
    #  If the pressure isn't calculated in the file, do it yourself
    if (data['p_dyn_proton'] == 0.0).all():
        data['p_dyn_proton'] = data['n_proton'] * m_p * (1e6) * data['u_mag']**2 * (1e3)**2 * (1e9)
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    #data = data.set_index('datetime')
    data = data.reindex(columns=default_df.columns)
    
    return(data)

def SWMFOH(target, starttime, finaltime, basedir=''):
    
    #  basedir is expected to be the folder /SolarWindEM
    model_path = 'models/swmf-oh/'
    full_path = basedir + model_path + target.lower() + '/'
    
    match target.lower():
        case 'jupiter':
            filenames = ['vogt_swmf-oh_jupiter_2014.txt', 
                         'vogt_swmf-oh_jupiter_2015.txt', 
                         'vogt_swmf-oh_jupiter_2016.txt',
                         'vogt_swmf-oh_jupiter_2017.txt',
                         'vogt_swmf-oh_jupiter_2018.txt',
                         'vogt_swmf-oh_jupiter_2019.txt',
                         'vogt_swmf-oh_jupiter_2020.txt',
                         'vogt_swmf-oh_jupiter_2021.txt',
                         'vogt_swmf-oh_jupiter_2022.txt']
        case 'juno':
            filenames = ['vogt_swmf-oh_juno_2014.txt', 
                         'vogt_swmf-oh_juno_2015.txt', 
                         'vogt_swmf-oh_juno_2016.txt']
        # case 'galileo':
        #     filenames = []
        # case 'cassini':
        #     filenames = []
        case 'mars':
            filenames = ['vogt_swmf-oh_mars_2014.txt', 
                         'vogt_swmf-oh_mars_2015.txt', 
                         'vogt_swmf-oh_mars_2016.txt',
                         'vogt_swmf-oh_mars_2017.txt',
                         'vogt_swmf-oh_mars_2018.txt',
                         'vogt_swmf-oh_mars_2019.txt',
                         'vogt_swmf-oh_mars_2020.txt',
                         'vogt_swmf-oh_mars_2021.txt',
                         'vogt_swmf-oh_mars_2022.txt']
        case _:
            logging.warning(['This version of the SWMF-OH reader',
                             ' does not support ' + target + ' observations.'])
            return(default_df)
        
    column_headers = ['year', 'month', 'day', 'hour', 
                      'X', 'Y', 'Z', 
                      'n_proton', 
                      'u_r', 'u_t', 'u_n', 
                      'B_r', 'B_t', 'B_n', 
                      'p_thermal', 
                      'J_r', 'J_t', 'J_n']
    
    data = default_df
    for filename in filenames:
        
        file_data = pd.read_table(full_path + filename, \
                             names=column_headers, \
                             header=0, delim_whitespace=True)
        file_data['datetime'] = pd.to_datetime(file_data[['year', 'month', 'day', 'hour']])
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < finaltime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
    
    data = data.drop(columns=['year', 'month', 'day', 'hour'])
    
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2 + data['u_n']**2)
    data['B_mag'] = np.sqrt(data['B_r']**2 + data['B_t']**2 + data['B_n']**2)
    data['p_dyn_proton'] = data['n_proton'] * m_p * (1e6) * (data['u_mag'] * 1e3)**2 * (1e9)
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    data = data.set_index('datetime')
    return(data)

def MSWIM2D(target, starttime, finaltime, basedir=''):
    
    model_path = 'models/MSWIM2D/'
    full_path = basedir + model_path + target.lower() + '/'
    
    match target.lower():
        case 'jupiter':
            filenames = ['umich_mswim2d_jupiter_2016.txt']
        case 'juno':
            filenames = ['umich_mswim2d_juno_2015.txt',
                         'umich_mswim2d_juno_2016.txt']
        # case 'galileo':
        #     filenames = []
        # case 'cassini':
        #     filenames = []
        case _:
            logging.warning('This version of the MSWIM2D reader '
                            'does not support ' + target + ' observations.')
            return(default_df)
        
    column_headers = ['datetime', 'hour', 'r', 'phi', 'rho_proton', 'u_x', 'u_y', 'u_z', 'B_x', 'B_y', 'B_z', 'T_i']
    
    data = default_df
    for filename in filenames:
        
        file_data = pd.read_table(full_path + filename, \
                             names=column_headers, \
                             skiprows=16, delim_whitespace=True)
        file_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%dT%H:%M:%S')
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < finaltime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)

    data['rho_proton'] = data['rho_proton'] * kg_per_amu  # kg / cm^3
    data['n_proton'] = data['rho_proton'] / m_p  # / cm^3
    data['u_mag'] = np.sqrt(data['u_x']**2 + data['u_y']**2 + data['u_z']**2) # km/s
    data['B_mag'] = np.sqrt(data['B_x']**2 + data['B_y']**2 + data['B_z']**2) # nT
    data['p_dyn_proton'] = (data['rho_proton'] * 1e6) * (data['u_mag'] * 1e3)**2 * (1e9) # nPa
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    data = data.set_index('datetime')
    return(data)

def HUXt(target, starttime, finaltime, basedir=''):
    
    model_path = 'models/HUXt/'
    full_path = basedir + model_path + target.lower() + '/'
    
    match target.lower():
        case 'jupiter':
            filenames = ['Jupiter_2016_Owens_HuxT.csv', 'Jupiter_2020_Owens_HuxT.csv']
        case _:
            logging.warning('This version of the HUXt reader'
                            ' does not support ' + target + ' observations.')
            return(default_df)
    
    #  r [R_S], lon [radians], Umag [km/s], 
    column_headers = ['datetime', 'r', 'lon', 'u_mag', 'B_pol']
    
    data = default_df  
    for filename in filenames:
        
        file_data = pd.read_csv(full_path + filename, 
                             names=column_headers, 
                             header=0)
        file_data['datetime'] = pd.to_datetime(file_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < finaltime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
    
    return(data)

def Heliocast(target, starttime, finaltime, data_path=''):
    import pandas as pd
    import numpy as np
    m_p = 1.67e-27 # proton mass
    
    match target.lower():
        case 'jupiter':
            filenames = []
        case 'juno':
            filenames = ['Heliocast_JunoCruise.csv']
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = []
    
    column_headers = ['datetime','Ur','Up','Ut','n','Bmag','Br','Bp','Bt']
    
    data = pd.DataFrame(columns=column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_csv(data_path + filename, \
                             names=column_headers, \
                             header=0)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    data['rho'] = data['n'] * m_p
    data['Umag'] = np.sqrt(data['Ur']**2 + data['Ut']**2 + data['Up']**2)
    data['Pdyn'] = (data['rho'] * 1e6) * (data['Umag'] * 1e3)**2 * 1e9
    
    return(data.reset_index(drop=True))

def ENLIL(target, starttime, finaltime, basedir=''):
    
    target = target.lower().replace(' ', '')
    #  basedir is expected to be the folder /SolarWindEM
    model_path = 'models/enlil/'
    full_path = basedir + model_path + target + '/'
    
    match target:
        case 'earth':
            filenames = ['ccmc_enlil_earth_20160509.txt',
                         'ccmc_enlil_earth_20160606.txt']
        case 'mars':
            filenames = ['ccmc_enlil_mars_20160509.txt',
                         'ccmc_enlil_mars_20160606.txt']
        case 'jupiter':
            filenames = ['ccmc_enlil_jupiter_20160509.txt', 
                         'ccmc_enlil_jupiter_20160606.txt']
        case 'stereoa':
            filename = ['ccmc_enlil_stereoa_20160509.txt',
                        'ccmc_enlil_stereoa_20160606.txt']
        case 'stereob':
            filename = ['ccmc_enlil_stereob_20160509.txt',
                        'ccmc_enlil_stereob_20160606.txt']
        case 'juno':
            filenames = ['ccmc_enlil_juno_20160509.txt',
                         'ccmc_enlil_juno_20160606.txt']
        # case 'galileo':
        #     filenames = []
        # case 'cassini':
        #     filenames = []
        case _:
            logging.warning('This version of the ENLIL reader'
                            ' does not support ' + target + ' observations.')
            return(default_df)
        
    #  NB switched labels from lon/lat to t/n-- should be the same?
    column_headers = ['time', 'R', 'Lat', 'Lon', 
                      'u_r', 'u_t', 'u_n', 
                      'B_r', 'B_t', 'B_n', 
                      'n_proton', 'T_proton', 'E_r', 'E_t', 'E_n', 
                      'u_mag', 'B_mag', 'p_dyn', 'BP']
    data = pd.DataFrame(columns = column_headers + ['datetime'])
    
    for filename in filenames:
        
        #  In these files, the start date is hidden in a comment, so read the first few lines
        header = []
        with open(full_path + filename) as f:
            for indx, line in enumerate(f):
                header.append(line.rstrip())
                if indx == 6:
                    break
        for line in header:
            if 'Start Date' in line:
                file_starttime = dt.datetime.strptime(line.split(': ')[1], '%Y/%m/%d %H:%M:%S')
        
        file_data = pd.read_table(full_path + filename, 
                             names=column_headers, 
                             comment='#', delim_whitespace=True)
        
        file_data['datetime'] = [file_starttime + dt.timedelta(days=time) for time in temp_data['time']]
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < finaltime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
        
        #  !!! Need to comment this mess and clean it up
        if len(data) > 0:
            if sub_data.index[0] < data.index[-1]:
                data_indx = np.where(data.index > sub_data.index[0])[0]
                sub_data_indx = np.where(sub_data.index < data.index[-1])[0]
                
                overlap_from_data = data.iloc[data_indx]
                overlap_from_sub_data = sub_data.iloc[sub_data_indx]
                
                data.drop(data.iloc[data_indx].index, axis=0, inplace=True)
                sub_data = sub_data.drop(labels=sub_data.iloc[sub_data_indx].index, axis='index')
                
                overlap_from_data = overlap_from_data.reindex(index=overlap_from_sub_data.index, method='nearest')
                overlap_concat = pd.concat([overlap_from_data, overlap_from_sub_data])
                
                temp = overlap_concat.groupby(overlap_concat.index).mean()
                
                sub_data = pd.concat([temp, sub_data])
                
        data = pd.concat([data, sub_data])
    
    data = data.drop(columns=['time'])
    
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2 + data['u_n']**2)
    data['B_mag'] = np.sqrt(data['B_r']**2 + data['B_t']**2 + data['B_n']**2)
    data['p_dyn_proton'] = data['n_proton'] * m_p * (1e6) * (data['u_mag'] * 1e3)**2 * (1e9)
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    #data = data.set_index('datetime')
    return(data)

def choose(model, target, starttime, stoptime, basedir=''):
    
    match model.lower():
        case 'tao':
            result = Tao(target, starttime, stoptime, basedir=basedir)
        case 'huxt':
            result = HUXt(target, starttime, stoptime, basedir=basedir)
        case 'mswim2d':
            result = MSWIM2D(target, starttime, stoptime, basedir=basedir)
        case 'swmf-oh':
            result = SWMFOH(target, starttime, stoptime, basedir=basedir)
        case 'enlil':
            result = ENLIL(target, starttime, stoptime, basedir=basedir)
        case _:
            result = None
            raise Exception("Model '" + model.lower() + "' is not currently supported.") 
    return(result)

def SWModel_Parameter_Concatenator(running_dataframe, current_dataframe):
    
    #  All groupby().mean() calls ignore NaNs, but it would be nice to make
    #  this explicit
    
    r_df = running_dataframe
    c_df = current_dataframe
    
    if r_df.index.has_duplicates:
        r_df = r_df.groupby(r_df.index).mean()
    if c_df.index.has_duplicates:
        c_df = c_df.groupby(c_df.index).mean()
    
    #  If both the running and current dataframes are populated
    # And if the first timestamp in current is later than the last timestamp in running
    if (len(r_df) > 0) and (len(c_df) > 0) and (c_df.index[0] <= r_df.index[-1]):
        #  Then there is an overlap
        
        #  Find where the running dataframe has dates later or equal to the first in the current dataframe
        #  And where the current dataframe has points earlier than the last in the running data dataframe
        r_geq_c = r_df[r_df.index >= c_df.index[0]]
        c_leq_r = c_df[c_df.index <= r_df.index[-1]]
        
        #  Drop the overlap region from the running and current dataframes
        r_df = r_df.drop(r_geq_c.index, axis='index')
        c_df = c_df.drop(c_leq_r.index, axis='index')
        
        #  Reindex the overlapping running dataframe to the overlapping current
        #  This follows a general attempt to prefer more recent data
        r_geq_c = r_geq_c.reindex(index=c_leq_r.index, method='nearest')
        
        #  Concatenate the overlapping dataframes, which now have identical indices,
        #  and average them together by index
        overlap_df = pd.concat([r_geq_c, c_leq_r])
        overlap_df = overlap_df.groupby(overlap_df.index).mean()
        
        output_df = pd.concat([r_df, overlap_df, c_df])
    
    else:
        output_df = pd.concat([r_df, c_df])        
            

    return(output_df)