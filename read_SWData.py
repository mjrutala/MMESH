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
 
# =============================================================================
# 
# =============================================================================
def Juno_published(starttime, stoptime, basedir=''):
    
    spacecraft_dir = 'Juno/'
    filepath_ions = basedir + spacecraft_dir + 'plasma/published/Wilson2018/'
    filepath_mag = basedir + spacecraft_dir + 'mag/published/AMDA/'
    
    # =========================================================================
    #     
    # =========================================================================
    def read_Juno_Wilson2018():
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
        
        return(output_spacecraft_data)
    
    # =========================================================================
    #     
    # =========================================================================
    def read_Juno_AMDAMAG():
        
        filename_template = 'amda_juno_cruise_fgm_%Y.txt'
        
        in_range = np.arange(starttime.year, stoptime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template) for t in filename_datetimes_in_range]   # !!! Repeated
    
        filelist = [filepath_mag + filename for filename in filenames_in_range if os.path.exists(filepath_mag + filename)]
        filelist.sort()
        
        input_columns = ['datestr',
                         'B_r', 'B_t', 'B_n', 'B_mag']
        
        #  Make a generator to read csv data and concatenate into a single dataframe
        data_generator = (pd.read_csv(f, delim_whitespace=True, skiprows=73, names=input_columns) 
                                     for f in filelist)
        spacecraft_data = pd.concat(data_generator, ignore_index=True)
        
        #  Set index to the datetime
        spacecraft_data.index = [dt.datetime.strptime(
                                 row['datestr'], '%Y-%m-%dT%H:%M:%S.%f') 
                                 for indx, row in spacecraft_data.iterrows()]
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < stoptime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        #  Check for duplicates in the datetime index
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='last')]
        return(spacecraft_data)
    # =========================================================================
    #         
    # =========================================================================
    plasma_data = read_Juno_Wilson2018()
    mag_data = read_Juno_AMDAMAG()
    
    data = pd.concat([plasma_data, mag_data], axis=1)
    
    return(data)
    
# =============================================================================
# 
# =============================================================================
def Ulysses(starttime, stoptime, basedir=''):
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
        
        output_columns = ['rau', 'hlat', 'hlong', 'n_proton', 'n_alpha', 'T_large', 'T_small', 'u_r', 'u_t', 'u_n', 'p_dyn_proton', 'p_dyn_alpha']
        output_columns_units = ['AU', 'deg', 'deg', 'cm^{-3}', 'cm^{-3}', 'K', 'K', 'km/s', 'km/s', 'km/s', 'nPa', 'nPa']
        spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        output_columns_eqns = ['', '', '', '', '', '', '', '', '', '', '0.5*n_proton*u_mag**2', '0.5*n_alpha*u_mag**2']
        spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
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
        
        output_columns = ['B_r', 'B_t', 'B_n', 'B_mag', 'nBVectors']
        output_columns_units = ['nT', 'nT', 'nT', 'nT', '#']
        spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        output_columns_eqns = ['', '', '', '', '']
        spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    # =============================================================================
    #     
    # =============================================================================
    plasma_data = read_Ulysses_plasma(starttime, stoptime)
    mag_data = read_Ulysses_MAG(starttime, stoptime)
    
    #  manually concatenate the attributes
    
    data = pd.concat([plasma_data, mag_data], axis=1)
    data.attrs['units'] = {**plasma_data.attrs['units'], **mag_data.attrs['units']}
    data.attrs['equations'] = {**plasma_data.attrs['equations'], **mag_data.attrs['equations']}
        
    return(data)

def Voyager(starttime, stoptime, spacecraft_number=1, basedir=None):
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
        
        output_columns = ['n_proton', 'T_proton', 'u_mag', 'u_r', 'u_t', 'u_n', 'p_dyn_proton', 'p_dyn_alpha']
        output_columns_units = ['cm^{-3}', 'cm^{-3}', 'K', 'K', 'km/s', 'km/s', 'km/s', 'nPa', 'nPa']
        spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        output_columns_eqns = ['', '', '', '', '', '', '', '', '', '', '0.5*n_proton*u_mag**2', '0.5*n_alpha*u_mag**2']
        spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
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
        
        output_columns = ['B_r', 'B_t', 'B_n', 'B_mag']
        output_columns_units = ['nT', 'nT', 'nT', 'nT']
        spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        output_columns_eqns = ['', '', '', '']
        spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    # =============================================================================
    #     
    # =============================================================================
    plasma_data = read_Voyager_plasma(starttime, stoptime)
    mag_data = read_Voyager_MAG(starttime, stoptime)
    
    #  manually concatenate the attributes
    #return(plasma_data, mag_data)
    data = pd.concat([plasma_data, mag_data], axis=1)
    #data.attrs['units'] = {**plasma_data.attrs['units'], **mag_data.attrs['units']}
    #data.attrs['equations'] = {**plasma_data.attrs['equations'], **mag_data.attrs['equations']}
        
    return(data)
# =============================================================================
# A simple function which allows you to specify the spacecraft name as an argument
# Which then calls the relrevant functions
# =============================================================================
def read(spacecraft, starttime, stoptime, basedir=None):
    
    match spacecraft.lower():
        case 'ulysses':
            result = Ulysses(starttime, stoptime, basedir=basedir)
        case 'juno':
            result = Juno_published(starttime, stoptime, basedir=basedir)
        case 'voyager 1':
            result = Voyager(starttime, stoptime, spacecraft_number='1', basedir=basedir)
        case 'voyager 2':
            result = Voyager(starttime, stoptime, spacecraft_number='2', basedir=basedir)
        case _:
            result = None
    return(result)
            
        
