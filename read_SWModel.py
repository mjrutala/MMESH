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

from pathlib import Path

m_p = 1.67e-27

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
        case 'mars':
            return(None)
        case 'jupiter':
            return(None)
        case 'saturn':
            return(None)
        case 'voyager1':
            filenames = ['swmhd_tao_voyager1_1978.txt',
                         'swmhd_tao_voyager1_1979.txt']    
        case 'voyager2':
            filenames = ['swmhd_tao_voyager2_1978.txt',
                         'swmhd_tao_voyager2_1979.txt',
                         'swmhd_tao_voyager2_1980.txt']
        case 'galileo':
            return(None)
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
    
    data = default_df
    
    for filename in filenames:
        #  Files downloaded from AMDA have a different order of columns than
        #  those directly from Chihiro Tao
        if 'fromamda' in filename:
            column_headers =  ['datetime', 
                               'n_proton', 'u_r', 'u_t', 'T_proton', 
                               'p_dyn_proton', 'B_t', 'B_r', 'angle_EST']
            def column_to_datetime(df):
                df['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
                return(df)
            
        else:
            column_headers = ['iyear', 'imonth', 'iday', 'stime',
                              'n_proton', 'T_proton', 'u_r', 'u_t',
                              'B_t', 'p_dyn',
                              'angle_EST', 'data_quality']
            def column_to_datetime(df):
                df['datetime'] = [dt.datetime(row['iyear'], 
                                              row['imonth'], 
                                              row['iday'], 
                                              dt.datetime.strptime(row['stime'], '%H:%M:%S').hour)
                                  for indx, row in df.iterrows()]
                df.drop(columns=['iyear', 'imonth', 'iday', 'stime'], inplace=True)
                return(df)
            
        partial_data = pd.read_table(fullpath / filename, 
                             names=column_headers, 
                             comment='#', delim_whitespace=True)
        partial_data = column_to_datetime(partial_data)
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        sub_data = partial_data.loc[(partial_data['datetime'] >= starttime) &
                                    (partial_data['datetime'] < finaltime)]
        
        #  Keep all the segments requested
        data = pd.concat([data, sub_data], ignore_index=True)
 
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2)
    data['B_mag'] = np.sqrt(data['B_r'].replace(np.nan, 0)**2 + data['B_t']**2)
    data['p_dyn'] = data['p_dyn_proton']
    
    data = data.set_index('datetime')
    data = data.reindex(columns=default_df.columns)
    
    return(data)

def SWMFOH(target, starttime, finaltime, basedir=''):
    
    import pandas as pd
    import numpy as np
    m_p = 1.67e-27 # proton mass
    
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
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = []
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
        
    column_headers = ['year', 'month', 'day', 'hour', 'X', 'Y', 'Z', 'n_proton', 'u_r', 'u_t', 'u_n', 'B_r', 'B_t', 'B_n', 'p_thermal', 'J_r', 'J_t', 'J_n']
    
    data = pd.DataFrame(columns = column_headers + ['datetime'])
    
    for filename in filenames:
        
        temp_data = pd.read_table(full_path + filename, \
                             names=column_headers, \
                             header=0, delim_whitespace=True)
        temp_data['datetime'] = pd.to_datetime(temp_data[['year', 'month', 'day', 'hour']])
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    data = data.drop(columns=['year', 'month', 'day', 'hour'])
    
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2 + data['u_n']**2)
    data['B_mag'] = np.sqrt(data['B_r']**2 + data['B_t']**2 + data['B_n']**2)
    data['p_dyn_proton'] = data['n_proton'] * m_p * (1e6) * (data['u_mag'] * 1e3)**2 * (1e9)
    data['p_dyn'] = data['p_dyn_proton']
    
    data = data.set_index('datetime')
    return(data)

def MSWIM2D(target, starttime, finaltime, basedir=''):
    
    import pandas as pd
    import numpy as np
    kg_per_amu = 1.66e-27 # 
    m_p = 1.67e-27
    
    model_path = 'models/MSWIM2D/'
    full_path = basedir + model_path + target.lower() + '/'
    
    match target.lower():
        case 'jupiter':
            filenames = ['umich_mswim2d_jupiter_2016.txt']
        case 'juno':
            filenames = ['umich_mswim2d_juno_2015.txt',
                         'umich_mswim2d_juno_2016.txt']
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = []
        
    column_headers = ['datetime', 'hour', 'r', 'phi', 'rho_proton', 'u_x', 'u_y', 'u_z', 'B_x', 'B_y', 'B_z', 'T_i']
    
    data = pd.DataFrame(columns = column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_table(full_path + filename, \
                             names=column_headers, \
                             skiprows=16, delim_whitespace=True)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%dT%H:%M:%S')
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    data['rho_proton'] = data['rho_proton'] * kg_per_amu  # kg / cm^3
    data['n_proton'] = data['rho_proton'] / m_p  # / cm^3
    data['u_mag'] = np.sqrt(data['u_x']**2 + data['u_y']**2 + data['u_z']**2) # km/s
    data['B_mag'] = np.sqrt(data['B_x']**2 + data['B_y']**2 + data['B_z']**2) # nT
    data['p_dyn'] = (data['rho_proton'] * 1e6) * (data['u_mag'] * 1e3)**2 * (1e9) # nPa
    data['p_dyn_proton'] = data['p_dyn']
    
    data = data.set_index('datetime')
    return(data)

def HUXt(target, starttime, finaltime, basedir=''):
        
    import pandas as pd
    import numpy as np
    
    model_path = 'models/HUXt/'
    full_path = basedir + model_path + target.lower() + '/'
    
    match target.lower():
        case 'jupiter':
            filenames = ['Jupiter_2016_Owens_HuxT.csv', 'Jupiter_2020_Owens_HuxT.csv']
        case 'juno':
            filenames = []
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = []
    
    #  r [R_S], lon [radians], Umag [km/s], 
    column_headers = ['datetime', 'r', 'lon', 'u_mag', 'B_pol']
    
    data = pd.DataFrame(columns = column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_csv(full_path + filename, 
                             names=column_headers, 
                             header=0)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    data = data.set_index('datetime')
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
        case _:
            result = None
            raise Exception("Model '" + model.lower() + "' is not currently supported.") 
    return(result)