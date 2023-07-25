# -*- coding: utf-8 -*-
"""
DESC:   Programs for reading spacecraft data.
        Supported: Ulysses.

MJR 2023-06-07
"""
# =============================================================================
# Importing modules
# =============================================================================
import pandas as pd
import datetime as dt
import numpy as np
import os
import astropy.units as u
import logging

rswd_log = logging.getLogger('read_SWData')
rswd_log.setLevel(logging.INFO) # Eventually, a /verbose kw
 
from pathlib import Path

# =============================================================================
# 
# =============================================================================
def get_fromCDAWeb(spacecraft, basedir='',
                   plasma=False, mag=False, merged=True):
    import subprocess
    import os
    import glob
    
    import pandas as pd
    
    baseurl = 'https://cdaweb.gsfc.nasa.gov/pub/data/'
    
    basedir = Path(basedir)
    #
    path_info = pd.DataFrame(columns=['plasma', 'mag', 'merged'], 
                             index=['url', 'savedir', 'filename_pattern'])
    
    #  Construct os-agnostic destination paths 
    # =============================================================================
    #     
    # =============================================================================
    match spacecraft.lower():
        case 'pioneer 10':
            spacecraft_family = 'pioneer'
            spacecraft_dir = basedir / spacecraft_family / spacecraft.lower().replace(' ','')
            #plasma_instrument = ''
            #mag_instrument = ''
            
            # #  Plasma data
            # plasma_url = '/pioneer/pioneer10/plasma/version31/v31_hour_ascii/'
            # #plasma_url = '/pioneer/pioneer10/plasma/version31/v31_summary_ascii/'
            # plasma_savedir = 
            # plasma_filename_pattern = 'p10plasm_????.asc'
            
            # #  Mag data
            # mag_url = '/pioneer/pioneer10/mag/ip_1min_ascii/'
            # mag_savedir = 
            # mag_filename_pattern = 'm?????.asc'
            
            #  Hourly merged (plasma + mag) data
            path_info['merged']['url'] = 'pioneer/pioneer10/merged/coho1hr_magplasma_ascii/'
            path_info['merged']['savedir'] = spacecraft_dir / 'merged/hourly/'
            path_info['merged']['filename_pattern'] = 'p10_????.asc' #  'p10_%Y.asc'
            
        case 'pioneer 11':
            spacecraft_family = 'pioneer'
            spacecraft_dir = basedir / spacecraft_family / spacecraft.lower().replace(' ','')
            #  Hourly merged (plasma + mag) data
            path_info['merged']['url'] = 'pioneer/pioneer11/merged/coho1hr_magplasma_ascii/'
            path_info['merged']['savedir'] = spacecraft_dir / 'merged/hourly/'
            path_info['merged']['filename_pattern'] = 'p11_????.asc' # 'p11_%Y.asc'
            
        case _:
            result = None
            return(result)
    
    # =============================================================================
    #     
    # =============================================================================
    commandname = ['wget']
    
    if not plasma: path_info.drop('plasma', axis=1, inplace=True)
    if not mag: path_info.drop('mag', axis=1, inplace=True)
    if not merged: path_info.drop('merged', axis=1, inplace=True)
    
    for column in path_info:
        
        if not os.path.exists(path_info[column]['savedir']):
            os.makedirs(path_info[column]['savedir'])
        
        #  A bunch of flags with this command call:
        #  -r: recursive
        #  -R *.html: reject .html files
        #  -np: don't ascend to the parent directory and go looking for files
        #  --directory-prefix: where to copy the files to
        #  -nH: no host directory (i.e. 'https://...')
        #  -nd: copy all files to this directory (could be --cut-dirs=#...)
        #  -s-show_progress: give a loading bar showing download time
        #  -q: quiet, don't show any other info
        flags = ['-r', '-R', '*.html', '-np', 
                  '--directory-prefix=' + str(path_info[column]['savedir']), 
                  '-nH', '-nd', '--show-progress', '-q']
        filepattern_flags = ['--accept=' + path_info[column]['filename_pattern']]
        host_url = [baseurl + path_info[column]['url']]
    
        commandline = commandname + flags + filepattern_flags + host_url
        
        subprocess.run(commandline)
        
    return()

# =============================================================================
#  
# =============================================================================
def make_DerezzedData(df_tuple, resolution=None):
    
    result_list = []
    match resolution:
        case None:
            #  Do not resample
            result_list.append(*df_tuple)
        
        case str():
            #  As long as the input is a string, use resample
            for df in df_tuple:
                resample = df.resample(resolution).mean()
                result_list.append(resample)
            
        case (float() | int()):
            #  Find the t_delta at percentile given by resolution
            #  The find the largest such t_delta among all DataFrames
            #  And scale to that
            percentiles = []
            for df in df_tuple:
                if len(df) > 0:
                    df['t_delta'] = (df.index.to_series().shift(-1) - 
                                    df.index.to_series()).dt.total_seconds()
                    percentiles.append(np.nanpercentile(df['t_delta'], resolution))
                
            resolution = np.max(percentiles)
            
            for df in df_tuple:
                if len(df) > 0:
                    df = df.resample('{:.0f}s'.format(resolution)).mean()
                    df = df.drop(['t_delta'], axis='columns')
                    result_list.append(df)

    return(tuple(result_list))      
            
# =============================================================================
# 
# =============================================================================
def Juno_published(starttime, stoptime, basedir='', resolution=None, combined=False):
    
    spacecraft_dir = 'Juno/'
    filepath_ions = basedir + spacecraft_dir + 'plasma/published/Wilson2018/'
    filepath_mag = basedir + spacecraft_dir + 'mag/published/AMDA/'
    
    # =========================================================================
    #     
    # =========================================================================
    def read_Juno_Wilson2018(starttime, stoptime):
        fullfilename = filepath_ions + 'Wilson2018_juno_cruise_jade_2016.csv'
        
        spacecraft_data = pd.read_csv(fullfilename)
        
        output_columns = ['u_mag', 'u_mag_err', 
                          'n_proton', 'n_proton_err', 
                          'n_alpha', 'n_alpha_err',
                          'T_proton', 'T_proton_err', 
                          'T_alpha', 'T_alpha_err',
                          'p_dyn_alpha', 'p_dyn_alpha_err', 
                          'p_dyn_proton', 'p_dyn_proton_err']
        output_spacecraft_data = pd.DataFrame(columns=output_columns)
        
        #  Data quality is a boolean to determine whether to use (1) or not use (0) the respective measurement
        dataquality = spacecraft_data['USED_IN_PAPER']
        dataquality_index = np.where(dataquality == 1)[0]
        
        #  No conversion necessary, km/s is good
        output_spacecraft_data['u_mag'] = spacecraft_data['V_KMPS']
        output_spacecraft_data['u_mag_err'] = spacecraft_data['V_KMPS_UNCERTAINTY']
        
        #  No conversion necessary, cm^-3 is good
        output_spacecraft_data['n_proton'] = spacecraft_data['N_PROTONS_CC']
        output_spacecraft_data['n_proton_err'] = spacecraft_data['N_PROTONS_CC_UNCERTAINTY']  
        # Alphas may contribute significantly during pressure/density increases
        output_spacecraft_data['n_alpha'] = spacecraft_data['N_ALPHAS_CC']
        output_spacecraft_data['n_alpha_err'] = spacecraft_data['N_ALPHAS_CC_UNCERTAINTY']
        
        #  No conversion necessary, eV is good
        output_spacecraft_data['T_proton'] = spacecraft_data['T_PROTONS_EV']
        output_spacecraft_data['T_proton_err'] = spacecraft_data['T_PROTONS_EV_UNCERTAINTY']
        #
        output_spacecraft_data['T_alpha'] = spacecraft_data['T_ALPHAS_EV']
        output_spacecraft_data['T_alpha_err'] = spacecraft_data['T_ALPHAS_EV_UNCERTAINTY']
        
        #  No conversion necessary, pressure in nPa is good
        output_spacecraft_data['p_dyn_proton'] = spacecraft_data['RAM_PRESSURE_PROTONS_NPA']
        output_spacecraft_data['p_dyn_proton_err'] = spacecraft_data['RAM_PRESSURE_PROTONS_NPA_UNCERTAINTY']
        #
        output_spacecraft_data['p_dyn_alpha'] = spacecraft_data['RAM_PRESSURE_ALPHAS_NPA']
        output_spacecraft_data['p_dyn_alpha_err'] = spacecraft_data['RAM_PRESSURE_ALPHAS_NPA_UNCERTAINTY']
        
        output_spacecraft_data['p_dyn'] = output_spacecraft_data['p_dyn_proton'] #!!!
        output_spacecraft_data['n_tot'] = output_spacecraft_data['n_proton']
        
        #  Finally, set the index to be datetimes
        output_spacecraft_data.index = [dt.datetime.strptime(t, '%Y-%jT%H:%M:%S.%f') for t in spacecraft_data['UTC']]
        
        #  Only pass on data with dataquality == 1, and reset the indices to reflect this
        output_spacecraft_data = output_spacecraft_data.iloc[dataquality_index]
        
        #  Within selected time range
        time_index = np.where((output_spacecraft_data.index >= starttime) 
                      & (output_spacecraft_data.index < stoptime))
        output_spacecraft_data = output_spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        output_spacecraft_data = output_spacecraft_data[~output_spacecraft_data.index.duplicated(keep='last')]
        
        #  Find the time between the nth observation and the n+1th
        #output_spacecraft_data['t_delta'] = (output_spacecraft_data.index.to_series().shift(-1) - 
        #                                                 output_spacecraft_data.index.to_series()).dt.total_seconds()
        
        return(output_spacecraft_data)
    
    # =========================================================================
    #     
    # =========================================================================
    def read_Juno_AMDAMAG(starttime, stoptime):
        
        filename_template = 'amda_juno_cruise_fgm_%Y.txt'
        
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template) for t in filename_datetimes_in_range]   # !!! Repeated
        
        input_columns = ['datestr',
                         'B_r', 'B_t', 'B_n', 'B_mag']
        
        filelist = [filepath_mag + filename for filename in filenames_in_range if os.path.exists(filepath_mag + filename)]
        filelist.sort()
        if len(filelist) == 0:
            rswd_log.info('No files found in')
            return(pd.DataFrame(columns=input_columns))
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, skiprows=73, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        #  Set index to the datetime
        spacecraft_data.index = [dt.datetime.strptime(
                                 row['datestr'], '%Y-%m-%dT%H:%M:%S.%f') 
                                 for indx, row in spacecraft_data.iterrows()]
        spacecraft_data = spacecraft_data.drop(['datestr'], axis=1)
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        #  Find the time between the nth observation and the n+1th
        #spacecraft_data['t_delta'] = (spacecraft_data.index.to_series().shift(-1) - 
        #                              spacecraft_data.index.to_series()).dt.total_seconds()
        return(spacecraft_data)
    # =========================================================================
    #         
    # =========================================================================
    plasma_data = read_Juno_Wilson2018(starttime, stoptime)
    mag_data = read_Juno_AMDAMAG(starttime, stoptime)
    all_data = (plasma_data, mag_data,)
    
    all_data = make_DerezzedData(all_data, resolution=resolution)
    
    data = pd.concat(all_data, axis=1)
    
    if combined == True: data = data.dropna(subset=['u_mag', 'p_dyn', 'n_tot', 'B_mag'])
    
    return(data)
    
# =============================================================================
# 
# =============================================================================
def Ulysses(starttime, stoptime, basedir='', resolution=None):
    #  Keep basefilepath flexible such that it can point to a static path or a 
    #  relative path
    #  Default to current working directory
    
    #  
    spacecraft_dir = 'Ulysses/'
    filepath_ions = basedir + spacecraft_dir + 'plasma/swoops/ion/hour/'
    filepath_mag = basedir + spacecraft_dir + 'mag/interplanetary/hour/'
    
    def read_Ulysses_plasma(starttime, finalttime):
        #  List only the relevant files by generating filenames
        filename_template_ion = 'hourav_%Y.asc'
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template_ion) for t in filename_datetimes_in_range]   # !!! Repeated
        
        filelist = [filepath_ions + filename for filename in filenames_in_range if os.path.exists(filepath_ions + filename)]
        filelist.sort()
        
        
        #  Read all the data into one dataframe
        input_columns = ['iyr', 'idoy', 'ihr', 'imin', 'isec',
                         'rau', 'hlat', 'hlong', 
                         'n_proton', 'n_alpha', 
                         'T_large', 'T_small', 
                         'u_r', 'u_t', 'u_n']
        if len(filelist) == 0:
            rswd_log.info('No files found in')
            return(pd.DataFrame(columns=input_columns))
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, header=0, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        
        datetime_generator = ('{:02n}-{:03n}T{:02n}:{:02n}:{:02n}'
                              .format(*row[['iyr', 'idoy', 'ihr', 'imin', 'isec']])
                              for indx, row in spacecraft_data.iterrows())
            
        spacecraft_data.index = [dt.datetime.strptime(
                                 gen, '%y-%jT%H:%M:%S') 
                                 for gen in datetime_generator]
        
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'ihr', 'imin', 'isec'], axis=1)
        
        spacecraft_data['u_mag'] = np.sqrt(np.sum(spacecraft_data[['u_r', 'u_t', 'u_n']]**2., axis=1))
        #spacecraft_data['rss'] = np.sqrt(spacecraft_data.u_r**2 + spacecraft_data.u_t**2 + spacecraft_data.u_n**2)
        
        
        defaultunits_to_nPa = ((1.0 * u.M_p / u.cm**3) * (u.km**2 / u.s**2)).to(u.nPa).value
        
        spacecraft_data['p_dyn_proton'] = spacecraft_data['n_proton'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn_alpha'] = 4. * spacecraft_data['n_alpha'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn'] = spacecraft_data['p_dyn_proton'] #  !!!
        spacecraft_data['n_tot'] = spacecraft_data['n_proton']
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        return(spacecraft_data)
    
    def read_Ulysses_MAG(starttime, stoptime):
        filename_template = 'ulymaghr_%Y.asc'
        
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template) for t in filename_datetimes_in_range]   # !!! Repeated
        
        filelist = [filepath_mag + filename for filename in filenames_in_range if os.path.exists(filepath_mag + filename)]
        filelist.sort()
        
        input_columns = ['iyr', 'iddoy', 'idhr',  #  decimal doy, decimal hour
                         'B_r', 'B_t', 'B_n', 'B_mag',  
                         'nBVectors']
        
        if len(filelist) == 0:
            rswd_log.info('No files found in')
            return(pd.DataFrame(columns=input_columns))
    
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, header=0, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        spacecraft_data['idoy'] = [int(iddoy) for iddoy in spacecraft_data['iddoy']]
        spacecraft_data['ihr'] = [int(idhr) for idhr in spacecraft_data['idhr']]
        spacecraft_data['imin'] = [int((idhr % 1) * 60.) for idhr in spacecraft_data['idhr']]
        spacecraft_data['isec'] = [int((((idhr % 1) * 60.) % 1) * 60.) for idhr in spacecraft_data['idhr']]
        datetime_generator = ('{:02n}-{:03n}T{:02n}:{:02n}:{:02n}'
                              .format(*row[['iyr', 'idoy', 'ihr', 'imin', 'isec']])
                              for indx, row in spacecraft_data.iterrows())
        spacecraft_data.index = [dt.datetime.strptime(
                                 gen, '%y-%jT%H:%M:%S') 
                                 for gen in datetime_generator]
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'iddoy', 'ihr', 'idhr', 'imin', 'isec'], axis=1)
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        return(spacecraft_data)
    
    # =============================================================================
    #     
    # =============================================================================
    plasma_data = read_Ulysses_plasma(starttime, stoptime)
    mag_data = read_Ulysses_MAG(starttime, stoptime)
    all_data = (plasma_data, mag_data,)
    
    all_data = make_DerezzedData(all_data, resolution=resolution)
    
    data = pd.concat(all_data, axis=1)
    
    return(data)

def Voyager(starttime, stoptime, spacecraft_number=1, basedir=None, resolution=None):
    #  Keep basefilepath flexible such that it can point to a static path or a 
    #  relative path
    #  Default to current working directory
    if basedir == None:
        basedir = ''
    spacecraft_number = str(spacecraft_number)
        
    #  
    spacecraft_dir = 'Voyager/Voyager' + spacecraft_number + '/'
    filepath_ions = basedir + spacecraft_dir + 'plasma/hour/'
    filepath_mag = basedir + spacecraft_dir + 'mag/hour/'
    
    def read_Voyager_plasma(starttime, finalttime):
        #  List only the relevant files by generating filenames
        filename_template_ion = 'v'+spacecraft_number+'_hour_%Y.asc'
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template_ion) for t in filename_datetimes_in_range]   # !!! Repeated
        
        filelist = [filepath_ions + filename for filename in filenames_in_range if os.path.exists(filepath_ions + filename)]
        filelist.sort()
        
        #  Read all the data into one dataframe
        input_columns = ['iyr', 'idoy', 'ihr',
                         'u_mag_proton', 'n_proton', 'v_kin_proton', 
                         'u_r', 'u_t', 'u_n']
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        #  Voyager 1 1977 plasma data has one line with an extra 0 at the end...
        #  Since 0 marks missing data, not sure what to make of it
        #  so 'on_bad_lines='skip'' skips that line completely
        data_generator = (pd.read_csv(f, delim_whitespace=True, header=0, names=input_columns, on_bad_lines='skip') 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        
        datetime_generator = ('{:02n}-{:03n}T{:02n}:00:00'
                              .format(*row[['iyr', 'idoy', 'ihr']])
                              for indx, row in spacecraft_data.iterrows())
            
        spacecraft_data.index = [dt.datetime.strptime(
                                 gen, '%Y-%jT%H:%M:%S') 
                                 for gen in datetime_generator]
        
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'ihr'], axis=1)
        
        spacecraft_data['u_mag'] = spacecraft_data['u_mag_proton']
        spacecraft_data['T_proton'] = 0.0052 * spacecraft_data['v_kin_proton']**2
        
        defaultunits_to_nPa = ((u.M_p / u.cm**3) * (u.km**2 / u.s**2)).to(u.nPa)
                                
        
        spacecraft_data['p_dyn_proton'] = spacecraft_data['n_proton'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn'] = spacecraft_data['p_dyn_proton'] #  !!!
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        #  Find the time between the nth observation and the n+1th
        spacecraft_data['t_delta'] = (spacecraft_data.index.to_series().shift(-1) - 
                                      spacecraft_data.index.to_series()).dt.total_seconds()
        
        #output_columns = ['n_proton', 'T_proton', 'u_mag', 'u_r', 'u_t', 'u_n', 'p_dyn_proton', 'p_dyn_alpha']
        #output_columns_units = ['cm^{-3}', 'cm^{-3}', 'K', 'K', 'km/s', 'km/s', 'km/s', 'nPa', 'nPa']
        #spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        #output_columns_eqns = ['', '', '', '', '', '', '', '', '', '', '0.5*n_proton*u_mag**2', '0.5*n_alpha*u_mag**2']
        #spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    def read_Voyager_MAG(starttime, stoptime):
        filename_template = 'vy'+spacecraft_number+'mag_1h.asc'
         
        filenames_in_range = [filename_template]  # !!! Repeated
    
        filelist = [filepath_mag + filename for filename in filenames_in_range if os.path.exists(filepath_mag + filename)]
        filelist.sort()
        
        input_columns = ['sc#', 'iyr', 'idoy', 'ihr',
                         'x_ihg', 'y_ihg', 'z_ihg', 'r_ihg',
                         'B_mag_avg', 'B_mag',  
                         'lat_ihg', 'lon_ihg']
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, header=0, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        datetime_generator = ('{:02n}-{:03n}T{:02n}:00:00'
                              .format(*row[['iyr', 'idoy', 'ihr']])
                              for indx, row in spacecraft_data.iterrows())
        spacecraft_data.index = [dt.datetime.strptime(
                                 gen, '%y-%jT%H:%M:%S') 
                                 for gen in datetime_generator]
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'ihr'], axis=1)
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        spacecraft_data['B_r'] = spacecraft_data['B_mag'] \
                                 * np.cos(spacecraft_data['lat_ihg']*np.pi/180.) \
                                 * np.cos(spacecraft_data['lon_ihg']*np.pi/180.)
        spacecraft_data['B_t'] = spacecraft_data['B_mag'] \
                                 * np.cos(spacecraft_data['lat_ihg']*np.pi/180.) \
                                 * np.sin(spacecraft_data['lon_ihg']*np.pi/180.)
        spacecraft_data['B_n'] = spacecraft_data['B_mag'] \
                                 * np.sin(spacecraft_data['lat_ihg']*np.pi/180.)
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        #  Find the time between the nth observation and the n+1th
        spacecraft_data['t_delta'] = (spacecraft_data.index.to_series().shift(-1) - 
                                      spacecraft_data.index.to_series()).dt.total_seconds()
        # output_columns = ['B_r', 'B_t', 'B_n', 'B_mag']
        # output_columns_units = ['nT', 'nT', 'nT', 'nT']
        # spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        # output_columns_eqns = ['', '', '', '']
        # spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    # =============================================================================
    #     
    # =============================================================================
    plasma_data = read_Voyager_plasma(starttime, stoptime)
    mag_data = read_Voyager_MAG(starttime, stoptime)
    
    match resolution:
        case None:
            #  Do not resample
            pass
        case str():
            #  As long as the input is a string, use resample
            plasma_data = plasma_data.resample(resolution).mean()
            mag_data = mag_data.resample(resolution).mean()
        case _:
            #  Resample to the largest common t_delta in either plasma or mag
            plasma_res = np.nanpercentile(plasma_data['t_delta'], resolution)
            mag_res = np.nanpercentile(mag_data['t_delta'], resolution)
            if plasma_res > mag_res:
                resolution = plasma_res
            else:
                resolution = mag_res
            plasma_data = plasma_data.resample('{:.0f}s'.format(resolution)).mean()
            mag_data = mag_data.resample('{:.0f}s'.format(resolution)).mean()
        
    plasma_data.drop(['t_delta'], axis=1, inplace=True)
    mag_data.drop(['t_delta'], axis=1, inplace=True)
    #data = pd.concat([plasma_data, mag_data], axis=1)
    
    #  manually concatenate the attributes
    #return(plasma_data, mag_data)
    data = pd.concat([plasma_data, mag_data], axis=1)
    #data.attrs['units'] = {**plasma_data.attrs['units'], **mag_data.attrs['units']}
    #data.attrs['equations'] = {**plasma_data.attrs['equations'], **mag_data.attrs['equations']}
        
    return(data)

def Pioneer(starttime, stoptime, spacecraft_number=10, basedir='', resolution=None):
    #  Keep basefilepath flexible such that it can point to a static path or a 
    #  relative path
    #  Default to current working directory
    
    basedir = Path(basedir)
    
    spacecraft_number = str(spacecraft_number).strip()
    spacecraft_dir = 'pioneer/pioneer' + spacecraft_number + '/' 
    
    filepath_ions =  basedir / spacecraft_dir / 'plasma/'
    filepath_mag = basedir / spacecraft_dir / 'mag/'
    filepath_merged = basedir / spacecraft_dir / 'merged/hourly/'
    
    def read_Pioneer_plasma(starttime, finalttime):
        #  List only the relevant files by generating filenames
        filename_template_ion = 'p'+spacecraft_numer+'_%Y.asc'
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template_ion) for t in filename_datetimes_in_range]   # !!! Repeated
        
        filelist = [filepath_ions + filename for filename in filenames_in_range if os.path.exists(filepath_ions + filename)]
        filelist.sort()
        
        #  Read all the data into one dataframe
        input_columns = ['iyr', 'idoy', 'ihr', 'imin', 'isec',
                         'rau', 'hlat', 'hlong', 
                         'n_proton', 'n_alpha', 
                         'T_large', 'T_small', 
                         'u_r', 'u_t', 'u_n']
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, header=0, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        
        datetime_generator = ('{:02n}-{:03n}T{:02n}:{:02n}:{:02n}'
                              .format(*row[['iyr', 'idoy', 'ihr', 'imin', 'isec']])
                              for indx, row in spacecraft_data.iterrows())
            
        spacecraft_data.index = [dt.datetime.strptime(
                                 gen, '%y-%jT%H:%M:%S') 
                                 for gen in datetime_generator]
        
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'ihr', 'imin', 'isec'], axis=1)
        
        spacecraft_data['u_mag'] = np.sqrt(np.sum(spacecraft_data[['u_r', 'u_t', 'u_n']]**2., axis=1))
        #spacecraft_data['rss'] = np.sqrt(spacecraft_data.u_r**2 + spacecraft_data.u_t**2 + spacecraft_data.u_n**2)
        
        
        defaultunits_to_nPa = ((1.0 * u.M_p / u.cm**3) * (u.km**2 / u.s**2)).to(u.nPa).value
        
        spacecraft_data['p_dyn_proton'] = spacecraft_data['n_proton'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn_alpha'] = 4. * spacecraft_data['n_alpha'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn'] = spacecraft_data['p_dyn_proton'] #  !!!
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        #  Find the time between the nth observation and the n+1th
        spacecraft_data['t_delta'] = (spacecraft_data.index.to_series().shift(-1) - 
                                      spacecraft_data.index.to_series()).dt.total_seconds()
        
        # output_columns = ['rau', 'hlat', 'hlong', 'n_proton', 'n_alpha', 'T_large', 'T_small', 'u_r', 'u_t', 'u_n', 'p_dyn_proton', 'p_dyn_alpha']
        # output_columns_units = ['AU', 'deg', 'deg', 'cm^{-3}', 'cm^{-3}', 'K', 'K', 'km/s', 'km/s', 'km/s', 'nPa', 'nPa']
        # spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        # output_columns_eqns = ['', '', '', '', '', '', '', '', '', '', '0.5*n_proton*u_mag**2', '0.5*n_alpha*u_mag**2']
        # spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    def read_Pioneer_MAG(starttime, stoptime):
        filename_template = 'ulymaghr_%Y.asc'
        
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template) for t in filename_datetimes_in_range]   # !!! Repeated
        
        filelist = [filepath_mag + filename for filename in filenames_in_range if os.path.exists(filepath_mag + filename)]
        filelist.sort()
        
        input_columns = ['iyr', 'iddoy', 'idhr',  #  decimal doy, decimal hour
                         'B_r', 'B_t', 'B_n', 'B_mag',  
                         'nBVectors']
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, header=0, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        spacecraft_data['idoy'] = [int(iddoy) for iddoy in spacecraft_data['iddoy']]
        spacecraft_data['ihr'] = [int(idhr) for idhr in spacecraft_data['idhr']]
        spacecraft_data['imin'] = [int((idhr % 1) * 60.) for idhr in spacecraft_data['idhr']]
        spacecraft_data['isec'] = [int((((idhr % 1) * 60.) % 1) * 60.) for idhr in spacecraft_data['idhr']]
        datetime_generator = ('{:02n}-{:03n}T{:02n}:{:02n}:{:02n}'
                              .format(*row[['iyr', 'idoy', 'ihr', 'imin', 'isec']])
                              for indx, row in spacecraft_data.iterrows())
        spacecraft_data.index = [dt.datetime.strptime(
                                 gen, '%y-%jT%H:%M:%S') 
                                 for gen in datetime_generator]
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'iddoy', 'ihr', 'idhr', 'imin', 'isec'], axis=1)
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        #  Find the time between the nth observation and the n+1th
        spacecraft_data['t_delta'] = (spacecraft_data.index.to_series().shift(-1) - 
                                      spacecraft_data.index.to_series()).dt.total_seconds()
        
        # output_columns = ['B_r', 'B_t', 'B_n', 'B_mag', 'nBVectors']
        # output_columns_units = ['nT', 'nT', 'nT', 'nT', '#']
        # spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        # output_columns_eqns = ['', '', '', '', '']
        # spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    def read_Pioneer_merged(starttime, finalttime):
        #  List only the relevant files by generating filenames
        filename_template_ion = 'p'+spacecraft_number+'_%Y.asc'
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template_ion) for t in filename_datetimes_in_range]   # !!! Repeated
        
        filelist = [filepath_merged / filename for filename in filenames_in_range if os.path.exists(filepath_merged / filename)]
        filelist.sort()
        
        #  Read all the data into one dataframe
        input_columns = ['iyr', 'idoy', 'ihr',
                         'rau', 'hgi_lat', 'hgi_lon', 
                         'B_r', 'B_t', 'B_n', 'B_mag',
                         'u_mag_proton', 'u_theta_proton', 'u_phi_proton',
                         'n_proton', 
                         'T_proton',
                         'proton_flux_low', 'proton_flux_mid', 'proton_flux_high']
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, names=input_columns,
                                      na_values=['999.99', '9999.9', '999.9999',  '9999999.', '9.999999e+06']) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        datetime_generator = ('{:02n}-{:03n}T{:02n}:00:00'
                              .format(*row[['iyr', 'idoy', 'ihr']])
                              for indx, row in spacecraft_data.iterrows())
            
        spacecraft_data.index = [dt.datetime.strptime(
                                 date, '%Y-%jT%H:%M:%S') 
                                 for date in datetime_generator]
        
        spacecraft_data = spacecraft_data.drop(['iyr', 'idoy', 'ihr'], axis=1)
        
        spacecraft_data['u_mag'] = spacecraft_data['u_mag_proton'] # !!!!
        
        defaultunits_to_nPa = ((1.0 * u.M_p / u.cm**3) * (u.km**2 / u.s**2)).to(u.nPa).value
        
        spacecraft_data['p_dyn_proton'] = spacecraft_data['n_proton'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn'] = spacecraft_data['p_dyn_proton'] #  !!!
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        
        #  Find the time between the nth observation and the n+1th
        spacecraft_data['t_delta'] = (spacecraft_data.index.to_series().shift(-1) - 
                                      spacecraft_data.index.to_series()).dt.total_seconds()
        
        return(spacecraft_data)
    # =============================================================================
    #     
    # =============================================================================
    
    merge_data = read_Pioneer_merged(starttime, stoptime)
    
    match resolution:
        case None:
            #  Do not resample
            pass
        case str():
            #  As long as the input is a string, use resample
            merge_data = merge_data.resample(resolution).mean()
            #plasma_data = plasma_data.resample(resolution).mean()
            #mag_data = mag_data.resample(resolution).mean()
        case _:
            #  Resample to the largest common t_delta in either plasma or mag
            merge_res = np.nanpercentile(merge_data['t_delta'], resolution)
            #plasma_res = np.nanpercentile(plasma_data['t_delta'], resolution)
            #mag_res = np.nanpercentile(mag_data['t_delta'], resolution)
            #if plasma_res > mag_res:
            #    resolution = plasma_res
            #else:
            #    resolution = mag_res
            resolution = merge_res
            plasma_data = plasma_data.resample('{:.0f}s'.format(resolution)).mean()
            mag_data = mag_data.resample('{:.0f}s'.format(resolution)).mean()
    
    merge_data.drop(['t_delta'], axis=1, inplace=True)
    #plasma_data.drop(['t_delta'], axis=1, inplace=True)
    #mag_data.drop(['t_delta'], axis=1, inplace=True)
    
    data = pd.concat([merge_data], axis=1)
    
    return(data)

# =============================================================================
# A simple function which allows you to specify the spacecraft name as an argument
# Which then calls the relrevant functions
# =============================================================================
def read(spacecraft, starttime, stoptime, basedir=None, resolution=None, combined=False):
    
    match spacecraft.lower():
        case 'ulysses':
            result = Ulysses(starttime, stoptime, 
                             basedir=basedir, resolution=resolution)
        case 'juno':
            result = Juno_published(starttime, stoptime, 
                                    basedir=basedir, resolution=resolution, combined=combined)
        case 'voyager 1':
            result = Voyager(starttime, stoptime, spacecraft_number='1', 
                             basedir=basedir, resolution=resolution)
        case 'voyager 2':
            result = Voyager(starttime, stoptime, spacecraft_number='2', 
                             basedir=basedir, resolution=resolution)
        case 'pioneer 10':
            result = Pioneer(starttime, stoptime, spacecraft_number='10', 
                             basedir=basedir, resolution=resolution)
        case 'pioneer 11':
            result = Pioneer(starttime, stoptime, spacecraft_number='11', 
                             basedir=basedir, resolution=resolution)
        case _:
            result = None
    return(result)
            
        
