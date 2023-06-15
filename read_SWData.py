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
def Juno_Wilson2018(starttime, finaltime, filepath=None):
    
    import pandas as pd
    import datetime as dt
    import numpy as np
    
    if filepath == None:
        filepath = '/Users/mrutala/projects/SolarWindProp/Wilson_JunoJADE_Cruise2016.csv'
    
    spacecraft_data = pd.read_csv(filepath)
    
    output_columns = ['Umag', 'Umag_err', 
                      'nproton', 'nproton_err', 
                      'nalpha', 'nalpha_err',
                      'Tproton', 'Tproton_err', 
                      'Talpha', 'Talpha_err',
                      'pdynalpha', 'pdynalpha_err', 
                      'pdynproton', 'pdynproton_err']
    output_spacecraft_data = pd.DataFrame(columns=output_columns)
    
    #  Data quality is a boolean to determine whether to use (1) or not use (0) the respective measurement
    dataquality = spacecraft_data['USED_IN_PAPER']
    dataquality_index = np.where(dataquality == 1)[0]
    
    #  No conversion necessary, km/s is good
    output_spacecraft_data['Umag'] = spacecraft_data['V_KMPS']
    output_spacecraft_data['Umag_err'] = spacecraft_data['V_KMPS_UNCERTAINTY']
    
    #  No conversion necessary, cm^-3 is good
    output_spacecraft_data['nproton'] = spacecraft_data['N_PROTONS_CC']
    output_spacecraft_data['nproton_err'] = spacecraft_data['N_PROTONS_CC_UNCERTAINTY']  
    # Alphas may contribute significantly during pressure/density increases
    output_spacecraft_data['nalpha'] = spacecraft_data['N_ALPHAS_CC']
    output_spacecraft_data['nalpha_err'] = spacecraft_data['N_ALPHAS_CC_UNCERTAINTY']
    
    #  No conversion necessary, eV is good
    output_spacecraft_data['Tproton'] = spacecraft_data['T_PROTONS_EV']
    output_spacecraft_data['Tproton_err'] = spacecraft_data['T_PROTONS_EV_UNCERTAINTY']
    #
    output_spacecraft_data['Talphas'] = spacecraft_data['T_ALPHAS_EV']
    output_spacecraft_data['Talphas_err'] = spacecraft_data['T_ALPHAS_EV_UNCERTAINTY']
    
    #  No conversion necessary, pressure in nPa is good
    output_spacecraft_data['pdynproton'] = spacecraft_data['RAM_PRESSURE_PROTONS_NPA']
    output_spacecraft_data['pdynproton_err'] = spacecraft_data['RAM_PRESSURE_PROTONS_NPA_UNCERTAINTY']
    #
    output_spacecraft_data['pdynalpha'] = spacecraft_data['RAM_PRESSURE_ALPHAS_NPA']
    output_spacecraft_data['pdynalpha_err'] = spacecraft_data['RAM_PRESSURE_ALPHAS_NPA_UNCERTAINTY']
    
    #  Finally, set the index to be datetimes
    output_spacecraft_data.index = [dt.datetime.strptime(t, '%Y-%jT%H:%M:%S.%f') for t in spacecraft_data['UTC']]
    
    #  Only pass on data with dataquality == 1, and reset the indices to reflect this
    output_spacecraft_data = output_spacecraft_data.iloc[dataquality_index]
    
    #  Within selected time range
    time_index = np.where((output_spacecraft_data.index >= starttime) 
                  & (output_spacecraft_data.index < finaltime))
    output_spacecraft_data = output_spacecraft_data.iloc[time_index]
    
    # # Adding MAG total B magnitude
    #(date_mag,b) = juno_b_mag_from_amda('juno_fgm_b_mag_cruise_2016.txt')
    
    return(output_spacecraft_data)

# =============================================================================
# 
# =============================================================================
def Ulysses(starttime, finaltime, basedir=None):
    #  Keep basefilepath flexible such that it can point to a static path or a 
    #  relative path
    #  Default to current working directory
    if basedir == None:
        basedir = ''
    
    #  
    spacecraft_dir = 'Ulysses/'
    filepath_ions = basedir + spacecraft_dir + 'plasma/swoops/ion/hour/'
    filepath_mag = basedir + spacecraft_dir + 'mag/interplanetary/hour/'
    
    def read_Ulysses_plasma(starttime, finalttime):
        #  List only the relevant files by generating filenames
        filename_template_ion = 'hourav_%Y.asc'
        in_range = np.arange(starttime.year, finaltime.year+1, 1)                                       # !!! Repeated
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
        
        spacecraft_data['p_dyn_proton'] = 0.5 * spacecraft_data['n_proton'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        spacecraft_data['p_dyn_alpha'] = 0.5 * 4. * spacecraft_data['n_alpha'] * spacecraft_data['u_mag']**2. * defaultunits_to_nPa
        
        #  Within selected time range
        time_index = np.where((spacecraft_data.index >= starttime) 
                      & (spacecraft_data.index < finaltime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        output_columns = ['rau', 'hlat', 'hlong', 'n_proton', 'n_alpha', 'T_large', 'T_small', 'u_r', 'u_t', 'u_n', 'p_dyn_proton', 'p_dyn_alpha']
        output_columns_units = ['AU', 'deg', 'deg', 'cm^{-3}', 'cm^{-3}', 'K', 'K', 'km/s', 'km/s', 'km/s', 'nPa', 'nPa']
        spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        output_columns_eqns = ['', '', '', '', '', '', '', '', '', '', '0.5*n_proton*u_mag**2', '0.5*n_alpha*u_mag**2']
        spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    def read_Ulysses_MAG(starttime, finaltime):
        filename_template = 'ulymaghr_%Y.asc'
        
        in_range = np.arange(starttime.year, finaltime.year+1, 1)                                       # !!! Repeated
        filename_datetimes_in_range = [dt.datetime(iyr, 1, 1) for iyr in in_range]                      # !!! Repeated
        filenames_in_range = [t.strftime(filename_template) for t in filename_datetimes_in_range]   # !!! Repeated
    
        filelist = [filepath_mag + filename for filename in filenames_in_range if os.path.exists(filepath_mag + filename)]
        filelist.sort()
        
        input_columns = ['iyr', 'iddoy', 'idhr',  #  decimal doy, decimal hour
                         'Br', 'Bt', 'Bn', 'Bmag',  
                         'nVectors']
        
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
                      & (spacecraft_data.index < finaltime))
        spacecraft_data = spacecraft_data.iloc[time_index]
        
        output_columns = ['Br', 'Bt', 'Bn', 'Bmag', 'nVectors']
        output_columns_units = ['nT', 'nT', 'nT', 'nT', '#']
        spacecraft_data.attrs['units'] = dict(zip(output_columns, output_columns_units))
        
        output_columns_eqns = ['', '', '', '', '']
        spacecraft_data.attrs['equations'] = dict(zip(output_columns, output_columns_eqns))
        
        return(spacecraft_data)
    
    # =============================================================================
    #     
    # =============================================================================
    plasma_data = read_Ulysses_plasma(starttime, finaltime)
    mag_data = read_Ulysses_MAG(starttime, finaltime)
    
    #  manually concatenate the attributes
    
    data = pd.concat([plasma_data, mag_data], axis=1)
    data.attrs['units'] = {**plasma_data.attrs['units'], **mag_data.attrs['units']}
    data.attrs['equations'] = {**plasma_data.attrs['equations'], **mag_data.attrs['equations']}
        
    return(data)

# =============================================================================
# A simple function which allows you to specify the spacecraft name as an argument
# Which then calls the relrevant functions
# =============================================================================
def read(spacecraft, starttime, finaltime, basedir=None):
    
    match spacecraft.lower():
        case 'ulysses':
            result = Ulysses(starttime, finaltime, basedir=basedir)
        case 'juno':
            result = Juno_Wilson2018(starttime, finaltime)
            
    return(result)
            
        
