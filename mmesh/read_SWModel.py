# -*- coding: utf-8 -*-
'''
Created on Tue Apr 11 10:09:36 2023

@author: Matt Rutala
'''

'''
target = string, startdate & finaldata = datetime
'''
import pandas as pd
import numpy as np
import datetime as dt
import copy
import logging
import glob

from pathlib import Path

from read_SWData import make_DerezzedData

m_p = 1.67e-27
kg_per_amu = 1.66e-27 # 

default_df = pd.DataFrame(columns=['u_mag', 'n_tot', 'p_dyn', 'B_mag', 
                                   'u_r', 'u_t', 'u_n', 
                                   'n_proton', 'n_alpha',
                                   'p_dyn_proton', 'p_dyn_alpha',
                                   'T_proton', 'T_alpha',
                                   'B_r', 'B_t', 'B_n', 'B_pol'])

def Tao(target, starttime, stoptime, basedir='', resolution=None):
    
    #  basedir is expected to be the folder /SolarWindEM
    model_path = '/Users/mrutala/Data/SolarWind_Models/Tao/'
    full_path = basedir + model_path + target.replace(' ', '').lower() + '/'
    
    filenames = [g.split('/')[-1] for g in glob.glob(full_path + '*.txt')]
    if filenames == []:
        logging.warning('This version of the ENLIL reader'
                        ' does not support ' + target + ' observations.')
        return(default_df)
    # match target:
    #     # case 'mars':
    #     #     return(None)
    #     # case 'jupiter':
    #     #     return(None)
    #     # case 'saturn':
    #     #     return(None)
    #     case 'voyager1':
    #         filenames = ['swmhd_tao_voyager1_1978.txt',
    #                      'swmhd_tao_voyager1_1979.txt']    
    #     case 'voyager2':
    #         filenames = ['swmhd_tao_voyager2_1978.txt',
    #                      'swmhd_tao_voyager2_1979.txt',
    #                      'swmhd_tao_voyager2_1980.txt']
    #     case 'pioneer10':
    #         filenames = ['swmhd_tao_pioneer10_1973.txt',
    #                      'swmhd_tao_pioneer10_1974.txt']
    #     case 'pioneer11':
    #         filenames = ['swmhd_tao_pioneer11_1974.txt',
    #                      'swmhd_tao_pioneer10_1975.txt',
    #                      'swmhd_tao_pioneer10_1976.txt',
    #                      'swmhd_tao_pioneer10_1977.txt',
    #                      'swmhd_tao_pioneer10_1978.txt']
    #     case 'ulysses':
    #         filenames = ['swmhd_tao_ulysses_1991.txt',
    #                      'swmhd_tao_ulysses_1992.txt',
    #                      'swmhd_tao_ulysses_1997.txt',
    #                      'swmhd_tao_ulysses_1998.txt',
    #                      'swmhd_tao_ulysses_1999.txt',
    #                      'swmhd_tao_ulysses_2003.txt',
    #                      'swmhd_tao_ulysses_2004.txt']
    #     case 'newhorizons':
    #         filenames = ['swmhd_tao_newhorizons_2006-2007.txt']
    #     case 'juno':
    #         filenames = ['fromamda_tao_juno_2015.txt',
    #                      'fromamda_tao_juno_2016.txt']
    #     case _:
    #         logging.warning('This version of the Tao SWMHD reader'
    #                         'does not support ' + target + ' observations.')
    #         return(default_df)
        
    #  This should put files in chronological order, useful for checking overalaps in the data
    filenames.sort()
    
    data = default_df
    
    for filename in sorted(filenames):
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
            
        file_data = pd.read_table(Path(full_path) / filename, 
                             names=column_headers, 
                             comment='#', delim_whitespace=True)
        file_data = column_to_datetime(file_data) 
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < stoptime)]
        
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
    
    data = make_DerezzedData(data, resolution=resolution)
    
    return(data)

def SWMFOH(target, starttime, stoptime, basedir='', resolution=None):
    
    #  basedir is expected to be the folder /SolarWindEM
    model_path = '/Users/mrutala/Data/SolarWind_Models/swmf-oh/'
    full_path = basedir + model_path + target.lower() + '/'
    
    filenames = [g.split('/')[-1] for g in glob.glob(full_path + '*')]
    if filenames == []:
        logging.warning('This version of the ENLIL reader'
                        ' does not support ' + target + ' observations.')
        return(default_df)
    # match target.lower():
    #     case 'jupiter':
    #         filenames = ['vogt_swmf-oh_jupiter_2014.txt', 
    #                      'vogt_swmf-oh_jupiter_2015.txt', 
    #                      'vogt_swmf-oh_jupiter_2016.txt',
    #                      'vogt_swmf-oh_jupiter_2017.txt',
    #                      'vogt_swmf-oh_jupiter_2018.txt',
    #                      'vogt_swmf-oh_jupiter_2019.txt',
    #                      'vogt_swmf-oh_jupiter_2020.txt',
    #                      'vogt_swmf-oh_jupiter_2021.txt',
    #                      'vogt_swmf-oh_jupiter_2022.txt']
    #     case 'juno':
    #         filenames = ['vogt_swmf-oh_juno_2014.txt', 
    #                      'vogt_swmf-oh_juno_2015.txt', 
    #                      'vogt_swmf-oh_juno_2016.txt']
    #     # case 'galileo':
    #     #     filenames = []
    #     # case 'cassini':
    #     #     filenames = []
    #     case 'mars':
    #         filenames = ['vogt_swmf-oh_mars_2014.txt', 
    #                      'vogt_swmf-oh_mars_2015.txt', 
    #                      'vogt_swmf-oh_mars_2016.txt',
    #                      'vogt_swmf-oh_mars_2017.txt',
    #                      'vogt_swmf-oh_mars_2018.txt',
    #                      'vogt_swmf-oh_mars_2019.txt',
    #                      'vogt_swmf-oh_mars_2020.txt',
    #                      'vogt_swmf-oh_mars_2021.txt',
    #                      'vogt_swmf-oh_mars_2022.txt']
    #     case _:
    #         logging.warning(['This version of the SWMF-OH reader',
    #                          ' does not support ' + target + ' observations.'])
    #         return(default_df)
        
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
                                    (file_data.index < stoptime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
    
    data = data.drop(columns=['year', 'month', 'day', 'hour'])
    
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2 + data['u_n']**2)
    data['B_mag'] = np.sqrt(data['B_r']**2 + data['B_t']**2 + data['B_n']**2)
    data['p_dyn_proton'] = data['n_proton'] * m_p * (1e6) * (data['u_mag'] * 1e3)**2 * (1e9)
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    #data = data.set_index('datetime')
    
    data = make_DerezzedData(data, resolution=resolution)
    
    return(data)

def MSWIM2D(target, starttime, stoptime, basedir='', resolution=None):
    
    model_path = '/Users/mrutala/Data/SolarWind_Models/MSWIM2D/'
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
                                    (file_data.index < stoptime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
    data['rho_proton'] = data['rho_proton'] * kg_per_amu  # kg / cm^3
    data['n_proton'] = data['rho_proton'] / m_p  # / cm^3
    data['u_mag'] = np.sqrt(data['u_x']**2 + data['u_y']**2 + data['u_z']**2) # km/s
    data['B_mag'] = np.sqrt(data['B_x']**2 + data['B_y']**2 + data['B_z']**2) # nT
    data['p_dyn_proton'] = (data['rho_proton'] * 1e6) * (data['u_mag'] * 1e3)**2 * (1e9) # nPa
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    #data = data.set_index('datetime')
    data = make_DerezzedData(data, resolution=resolution)
    
    return(data)

def HUXt(target, starttime, stoptime, basedir='', resolution=None):
    
    model_path = '/Users/mrutala/Data/SolarWind_Models/HUXt/'
    full_path = basedir + model_path + target.replace(' ', '').lower() + '/'
    
    filenames = [g.split('/')[-1] for g in glob.glob(full_path + '*')]
    if filenames == []:
        logging.warning('This version of the ENLIL reader'
                        ' does not support ' + target + ' observations.')
        return(default_df)
    # match target.replace(' ','').lower():
    #     case 'jupiter':
    #         filenames = ['Jupiter_2016_Owens_HuxT.csv', 'Jupiter_2020_Owens_HuxT.csv']
    #     case 'juno':
    #         filenames = ['juno_2015001-2016001_HUXt.csv', 'juno_2016001-2017001_HUXt.csv']
    #     case 'ulysses':
    #         filenames = ['ulysses_1991001-1992001_HUXt.csv', 'ulysses_1992001-1993001_HUXt.csv',
    #                      'ulysses_1997001-1998001_HUXt.csv', 'ulysses_1998001-1999001_HUXt.csv',
    #                      'ulysses_1999001-2000001_HUXt.csv', 'ulysses_2003001-2004001_HUXt.csv',
    #                      'ulysses_2004001-2005001_HUXt.csv']
    #     case 'voyager1':
    #         filenames = ['voyager1_1978001-1979001_HUXt.csv', 'voyager1_1979001-1980001_HUXt.csv']
    #     case 'voyager2':
    #         filenames = ['voyager2_1978001-1979001_HUXt.csv', 'voyager2_1979001-1980001_HUXt.csv',
    #                      'voyager2_1980001-1981001_HUXt.csv']
    #     case 'pioneer10':
    #         filenames = ['pioneer10_1973001-1974001_HUXt.csv', 'pioneer10_1974001-1975001_HUXt.csv']
    #     case 'pioneer11':
    #         filenames = ['pioneer11_1974001-1975001_HUXt.csv', 'pioneer11_1975001-1976001_HUXt.csv',
    #                      'pioneer11_1976001-1977001_HUXt.csv', 'pioneer11_1977001-1978001_HUXt.csv',
    #                      'pioneer11_1978001-1979001_HUXt.csv', 'pioneer11_1979001-1980001_HUXt.csv']
    #     case _:
    #         logging.warning('This version of the HUXt reader'
    #                         ' does not support ' + target + ' observations.')
    #         return(default_df)
    
    #  r [R_S], lon [radians], Umag [km/s], 
    column_headers = ['datetime', 'r', 'lon', 'u_mag', 'B_pol']
    
    data = default_df  
    for filename in sorted(filenames):
        
        file_data = pd.read_csv(full_path + filename, 
                             names=column_headers, 
                             header=0)
        try:
            file_data['datetime'] = pd.to_datetime(file_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            file_data['datetime'] = pd.to_datetime(file_data['datetime'], format='%Y-%m-%d %H:%M:%S')
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < stoptime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
    
    # !!!! This wwould be increasing resolution, not decreasing it... does it work?
    data = make_DerezzedData(data, resolution=resolution)
    
    return(data)

def Heliocast(target, starttime, stoptime, data_path='', resolution=None):
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
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < stoptime)]
        
        data = pd.concat([data, sub_data])
    
    data['rho'] = data['n'] * m_p
    data['Umag'] = np.sqrt(data['Ur']**2 + data['Ut']**2 + data['Up']**2)
    data['Pdyn'] = (data['rho'] * 1e6) * (data['Umag'] * 1e3)**2 * 1e9
    
    data = make_DerezzedData(data, resolution=resolution)
    
    return(data.reset_index(drop=True))

def ENLIL(target, starttime, stoptime, basedir='', resolution=None):
    
    target = target.lower().replace(' ', '')
    #  basedir is expected to be the folder /SolarWindEM
    model_path = '/Users/mrutala/Data/SolarWind_Models/enlil/'
    full_path = basedir + model_path + target + '/'
    
    filenames = [g.split('/')[-1] for g in glob.glob(full_path + '*')]
    if filenames == []:
        logging.warning('This version of the ENLIL reader'
                        ' does not support ' + target + ' observations.')
        return(default_df)
    # match target:
    #     case 'earth':
    #         filenames = [g.split('/')[-1] for g in glob.glob(full_path + '*')]

    #     case 'mars':
    #         filenames = ['ccmc_enlil_mars_20160509.txt',
    #                      'ccmc_enlil_mars_20160606.txt']
    #     case 'jupiter':
    #         filenames = ['ccmc_enlil_jupiter_20160509.txt', 
    #                      'ccmc_enlil_jupiter_20160606.txt']
    #     case 'stereoa':
    #         filename = ['ccmc_enlil_stereoa_20160509.txt',
    #                     'ccmc_enlil_stereoa_20160606.txt']
    #     case 'stereob':
    #         filename = ['ccmc_enlil_stereob_20160509.txt',
    #                     'ccmc_enlil_stereob_20160606.txt']
    #     case 'juno':
    #         filenames = ['ccmc_enlil_juno_20160509.txt',
    #                      'ccmc_enlil_juno_20160606.txt']
    #     case 'ulysses':
    #         filenames = [g.split('/')[-1] for g in glob.glob(full_path + '*')]
            
    #     case _:
    #         logging.warning('This version of the ENLIL reader'
    #                         ' does not support ' + target + ' observations.')
    #         return(default_df)
        
    #  NB switched labels from lon/lat to t/n-- should be the same?
    column_headers = ['time', 'R', 'Lat', 'Lon', 
                      'u_r', 'u_t', 'u_n', 
                      'B_r', 'B_t', 'B_n', 
                      'n_proton', 'T_proton', 'E_r', 'E_t', 'E_n', 
                      'u_mag', 'B_mag', 'p_dyn', 'BP']
    data = pd.DataFrame(columns = column_headers + ['datetime'])
    
    for filename in sorted(filenames):
        
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
        
        file_data['datetime'] = [file_starttime + dt.timedelta(days=time) for time in file_data['time']]
        file_data = file_data.set_index('datetime')
        
        #  Ditch unrequested data now so you don't need to hold it all in memory
        span_data = file_data.loc[(file_data.index >= starttime) &
                                    (file_data.index < stoptime)]
        
        data = SWModel_Parameter_Concatenator(data, span_data)
    
    data = data.drop(columns=['time', 'datetime'])
    
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2 + data['u_n']**2)
    data['B_mag'] = np.sqrt(data['B_r']**2 + data['B_t']**2 + data['B_n']**2)
    data['p_dyn_proton'] = data['n_proton'] * m_p * (1e6) * (data['u_mag'] * 1e3)**2 * (1e9)
    data['p_dyn'] = data['p_dyn_proton']
    data['n_tot'] = data['n_proton']
    
    data = make_DerezzedData(data, resolution=resolution)
    
    return(data)

def choose(model, target, starttime, stoptime, basedir='', resolution=None):
    
    match model.lower():
        case 'tao':
            result = Tao(target, starttime, stoptime, 
                         basedir=basedir, resolution=resolution)
        case 'huxt':
            result = HUXt(target, starttime, stoptime, 
                          basedir=basedir, resolution=resolution)
        case 'mswim2d':
            result = MSWIM2D(target, starttime, stoptime, 
                             basedir=basedir, resolution=resolution)
        case 'swmf-oh':
            result = SWMFOH(target, starttime, stoptime, 
                            basedir=basedir, resolution=resolution)
        case 'enlil':
            result = ENLIL(target, starttime, stoptime, 
                           basedir=basedir, resolution=resolution)
        case _:
            result = None
            raise Exception('Model '' + model.lower() + '' is not currently supported.') 
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

def runHUXt(target, metakernel_paths, starttime, stoptime, basedir=''):
    '''
    We're going to run HUXt at relatively high temporal (and potentially 
    spatial) resolution, so it's essential that we run it over as little a 
    volume of space as possible.
    
    To this end, we need to:
        1) Get the helioposition of the spacecraft (r, lon, lat)
        2) If the spacecraft covers too large an area, split the time domain
            into several small sections
        3) Run HUXt over all the necessary sparial sections
        4) Combine the HUXt dataframes back together
        5) Return this dataframe (and save it?)

    Parameters
    ----------
    target : TYPE
        DESCRIPTION.
    starttime : TYPE
        DESCRIPTION.
    stoptime : TYPE
        DESCRIPTION.
    basedir : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    '''
    
    #!!!! Get spacecraft longitudes during this time span
    #  Or split up the time span dynamically so each HUXt run only needs to be ~2-4ish degrees lon.
    #  Then look into how sampling in 3D works (i.e., with latitude) -- hpopefully theres another easy sampler like sc_series...
    #  Then move this to read_model? Maybe with a renaming as well (Model class? models.py? ...)
    
    import sys
    sys.path.append('/Users/mrutala/projects/HUXt-DIAS/code/')
    
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import astropy.units as u
    
    import spacecraftdata as SpacecraftData
    import huxt as H
    import huxt_analysis as HA
    import huxt_inputs as Hin
    
    lon_res = (360/128.)
    lon_valid = np.arange(0, 360+lon_res, lon_res)
    lon_limit = 2.*lon_res #  degrees, 128 bins by default
    
    reference_frame = 'SUN_INERTIAL' # HGI
    observer = 'SUN'
    
    #spacecraft = SpacecraftData.SpacecraftData(target)
    #spacecraft.make_timeseries(starttime, stoptime, dt.timedelta(hours=1))
    #spacecraft.find_state(reference_frame, observer, keep_kernels=True)  #  Keep kernels to find Earth's position later
    #spacecraft.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, 
    #                       coord3_range=np.array(lat_range)*np.pi/180., 
    #                       transform='reclat')
    
    for metakernel_path in metakernel_paths: 
        spice.furnsh(metakernel_path)
    times = np.arange(starttime, stoptime, dt.timedelta(hours=1)).astype(dt.datetime)
    etimes = spice.datetime2et(times)
    pos_xyz, lt = spice.spkpos(target, etimes, reference_frame, 'None', observer)
    
    r, lon, lat = zip(*[spice.reclat(pos) for pos in pos_xyz])
    lon = np.array(lon) * (180/np.pi)
    
    #positions = spacecraft.data[['x_pos', 'y_pos', 'z_pos']].to_numpy('float64')
    #r, lon, lat = zip(*[spice.reclat(list(pos)) for pos in positions])
    #spacecraft.data['r_pos'] = np.array(r)
    #spacecraft.data['lon_pos'] = np.array(lon) * (180/np.pi)
    # spacecraft.data['lat_pos'] = np.array(lat) * (180/np.pi)
    
    earth_pos, ltime = spice.spkpos('EARTH', etimes, reference_frame, 'NONE', observer)
    earth_r, earth_lon, earth_lat = zip(*[spice.reclat(pos) for pos in earth_pos])
    earth_lon = np.array(earth_lon) * (180/np.pi)
    #spacecraft.data['earth_lon'] = np.array(earth_lon) * (180/np.pi)
    
    # =========================================================================
    #   Find the the indices to split up the position into managable chunks
    #   N.B. this won't work for non-monotonic longitude arrays
    # =========================================================================
    #lon_range = spacecraft.data['lon_pos'][-1] - spacecraft.data['lon_pos'][0]
    lon = (lon + 360.) % 360.
    earth_lon = (earth_lon + 360.) % 360.
    lon_range = lon[-1] - lon[0]
    if lon_range < 0:
        lon_range += 360.
    
    lon_segments = np.ceil(lon_range/lon_limit)
    
    indx_list = []
    for i in range(int(lon_segments)):
        i_step = lon[0] + ((lon_range/lon_segments) * i)
        indx_list.append(np.argmin(abs(lon - i_step)))
        
    start_indx = indx_list
    stop_indx = indx_list[1:] + [len(etimes)]
    
    output_list = []
    lon_start_arr = np.zeros(len(etimes))
    lon_stop_arr = np.zeros(len(etimes))
    for start, stop in zip(start_indx, stop_indx):
        run_start = times[start]
        run_stop = times[stop-1] + dt.timedelta(days=1)  #  HUXt ends runs a little early, seemingly
        run_time = (run_stop - run_start).days * u.day
        
        #  Find the bounding longitudes for this duration
        lon_start = (np.min(lon[start:stop]) - earth_lon[start] + 360) % 360.
        lon_stop = (np.max(lon[start:stop]) - earth_lon[start] + 360) % 360.
        
        #  Fix the starting and stopping longitudes to the longitude values the model actually usesFi
        lon_start = lon_valid[np.where(lon_valid <= lon_start)[0][-1]] % 360.
        lon_stop = lon_valid[np.where(lon_valid >= lon_stop)[0][0]] % 360.
        
        lon_start_arr[start:stop] = (lon_start + earth_lon[start] + 360) % 360.
        lon_stop_arr[start:stop] = (lon_stop + earth_lon[start] + 360) % 360.
        
        r_min = 215 * u.solRad
        r_max = (10. * u.AU).to(u.solRad)
        
        #download and process the OMNI data
        time, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(run_start, run_stop)
        
        #set up the model, with (optional) time-dependent bpol boundary conditions
        model = Hin.set_time_dependent_boundary(vcarr, time, run_start, run_time, 
                                                r_min=r_min, r_max=r_max, dt_scale=1.0, latitude=0*u.deg,
                                                bgrid_Carr = bcarr, lon_start=lon_start*u.deg, lon_stop=lon_stop*u.deg, frame='sidereal')

        model.solve([])
        
        HA.plot(model, 1*u.day)
        plt.show()
        
        spacecraft_df = HA.get_observer_timeseries(model, observer=target)
        spacecraft_df['lon'] = (spacecraft_df['lon']*(180/np.pi) + earth_lon[start] + 360) % 360.
        
        output_list.append(spacecraft_df)
        del model
        
        print('HUXt run for {} - {} complete.'.format(run_start.strftime('%d-%m-%Y'), run_stop.strftime('%d-%m-%Y')))
    
    spice.kclear()
    
    output_concat = pd.concat(output_list, axis=0, ignore_index=True)
    output_concat = output_concat.set_index(output_concat['time'])
    output_concat = output_concat.drop(labels='time', axis='columns')
    output_concat = output_concat.sort_index(axis='index')
    result = output_concat.resample('60Min').mean()
    result = result.loc[(result.index >= starttime) & (result.index < stoptime)]
    
    #
    indx = np.where((lon_stop_arr - result['lon']) < 0)[0]
    if len(indx) > 0:
        print('Spacecraft longitude is greater than the longitude stop value')
        print('Mean residuals [deg.]: ' + str(np.mean(lon_stop_arr[indx] - result['lon'][indx].to_numpy('float64'))))
        print('If this is small, it is likely due to the dataframe resampling.')
        
    indx = np.where((result['lon'] - lon_start_arr) < 0)[0]
    if len(indx) > 0:
        print('Spacecraft longitude is less than the longitude start value')
        print('Mean residuals: [deg.] ' + str(np.mean(result['lon'][indx].to_numpy('float64') - lon_start_arr[indx])))
        print('If this is small, it is likely due to the dataframe resampling.')
    #
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, ax = plt.subplots(nrows=1, figsize=(8,5))      
        ax.scatter(result['r']*u.solRad.to(u.AU), result['lon'], s=4)
        ax.plot(result['r']*u.solRad.to(u.AU), lon_start_arr, color='black', linestyle='--')
        ax.plot(result['r']*u.solRad.to(u.AU), lon_stop_arr, color='black', linestyle=':')
        plt.show()
    
        
    #  Finally, reset the heliolongitude in the output dataframe, since the 
    #  input longitudes are meaningless (Earth's lon. is reset to 0 each time)
    # result['lon'] = spacecraft.data['lon_pos']
    
    filestem = '{}_{}-{}_HUXt'.format(target.replace(' ', '').lower(), 
                                      starttime.strftime('%Y%j'), 
                                      stoptime.strftime('%Y%j'))

    result.to_csv(basedir + filestem + '.csv')    
    
    return result