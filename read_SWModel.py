# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:09:36 2023

@author: Matt Rutala
"""

"""
target = string, startdate & finaldata = datetime
"""
def Tao(target, starttime, finaltime, basedir=''):
    
    #from astropy.io import ascii
    
    #import datetime as datetime
    import pandas as pd
    import numpy as np
    m_p = 1.67e-27
    
    #  basedir is expected to be the folder /SolarWindEM
    model_path = 'models/tao/'
    full_path = basedir + model_path + target.lower() + '/'
    
    match target.lower():
        case 'jupiter':
            filenames = ['jupiter_tao_model_2016_v2.txt', 
                         'jupiter_tao_model_2020_v1.txt']
        case 'juno':
            filenames = ['amdaoutput_tao_juno_2015.txt',
                         'amdaoutput_tao_juno_2016.txt']
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = ['Cassini_2000_jupflyby_tao_model_v1.txt']
    
    column_headers = ['datetime', 'n_proton', 'u_r', 'u_t', 'T_proton', 'p_dyn_proton', 'B_t', 'B_r', 'angle_JSE']
    
    data = pd.DataFrame(columns=column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_table(full_path + filename, 
                             names=column_headers, 
                             comment='#', delim_whitespace=True)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
        
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
        
    #data['rho'] = ((data['Pdyn'] * 1e-9) / (data['Umag']**2 * 1e6)) * (1e-6) 
    #data['n'] = data['rho'] / m_p
    data['u_mag'] = np.sqrt(data['u_r']**2 + data['u_t']**2)
    data['B_mag'] = np.sqrt(data['B_r']**2 + data['B_t']**2)
    data['p_dyn'] = data['p_dyn_proton']
    
    data = data.set_index('datetime')
    
    return(data)

def TaoSW_x01(target, starttime, finaltime, smooth=1, delta=False, data_path=''):
    
    #import datetime as datetime
    import pandas as pd
    import numpy as np
    m_p = 1.67e-27
    
    match target.lower():
        case 'jupiter':
            filenames = ['jupiter_tao_model_2016_v2.txt', 'jupiter_tao_model_2020_v1.txt']
        case 'juno':
            filenames = ['jupiter_tao_model_juno_2016_v1.txt']
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = ['Cassini_2000_jupflyby_tao_model_v1.txt']
    
    column_headers = ['datetime', 'Umag', 'Pdyn', 'delta_angle', 'Bt', 'Br']
    
    data = pd.DataFrame(columns=column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_table(data_path + filename, \
                             names=column_headers, \
                             comment='#', delim_whitespace=True)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
        
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
        
    data['rho'] = ((data['Pdyn'] * 1e-9) / (data['Umag']**2 * 1e6)) * (1e-6) 
    data['n'] = data['rho'] / m_p
    data['Bmag'] = np.sqrt(data['Br']**2 + data['Bt']**2)

    
    return(data.reset_index(drop=True))

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