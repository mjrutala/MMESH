# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:09:36 2023

@author: Matt Rutala
"""

"""
target = string, startdate & finaldata = datetime
"""
def TaoSW(target, starttime, finaltime, data_path=''):
    
    #from astropy.io import ascii
    
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

def VogtSW(target, starttime, finaltime, data_path=''):
    
    import pandas as pd
    import numpy as np
    m_p = 1.67e-27 # proton mass
    
    match target.lower():
        case 'jupiter':
            filenames = ['sw_jupiter_rtn_vogt_2016_v2.txt', 'sw_jupiter_rtn_2020_Vogt.txt']
        case 'juno':
            filenames = ['sw_juno_rtn_vogt_2016_v2.txt']
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = []
        
    column_headers = ['year', 'month', 'day', 'hour', 'X', 'Y', 'Z', 'n', 'Ur', 'Ut', 'Un', 'Br', 'Bt', 'Bn', 'pthermal', 'Jr', 'Jt', 'Jn']
    
    data = pd.DataFrame(columns = column_headers + ['datetime'])
    
    for filename in filenames:
        
        temp_data = pd.read_table(data_path + filename, \
                             names=column_headers, \
                             header=0, delim_whitespace=True)
        temp_data['datetime'] = pd.to_datetime(temp_data[['year', 'month', 'day', 'hour']])
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    data = data.drop(columns=['year', 'month', 'day', 'hour'])
    
    data['rho'] = data['n'] * m_p
    data['Umag'] = np.sqrt(data['Ur']**2 + data['Ut']**2 + data['Un']**2)
    data['Bmag'] = np.sqrt(data['Br']**2 + data['Bt']**2 + data['Bn']**2)
    data['Pdyn'] = data['rho'] * (1e6) * (data['Umag'] * 1e3)**2 * (1e9)
    
    return(data.reset_index(drop=True))

def MSWIM2DSW(target, starttime, finaltime, data_path=''):
    
    import pandas as pd
    import numpy as np
    kg_per_amu = 1.66e-27 # 
    m_p = 1.67e-27
    
    match target.lower():
        case 'jupiter':
            filenames = ['mswim2d_interp_output_2016.txt']
        case 'juno':
            filenames = []
        case 'galileo':
            filenames = []
        case 'cassini':
            filenames = []
        
    column_headers = ['datetime', 'hour', 'r', 'phi', 'rho', 'Ux', 'Uy', 'Uz', 'Bx', 'By', 'Bz', 'Ti']
    
    data = pd.DataFrame(columns = column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_table(data_path + filename, \
                             names=column_headers, \
                             header=15, delim_whitespace=True)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    data['rho'] = data['rho'] * kg_per_amu  # kg / cm^3
    data['n'] = data['rho'] / m_p  # / cm^3
    data['Umag'] = np.sqrt(data['Ux']**2 + data['Uy']**2 + data['Uz']**2) # km/s
    data['Bmag'] = np.sqrt(data['Bx']**2 + data['By']**2 + data['Bz']**2) # nT
    data['Pdyn'] = (data['rho'] * 1e6) * (data['Umag'] * 1e3)**2 * (1e9) # nPa
    
    return(data.reset_index(drop=True))

def HUXt(target, starttime, finaltime, data_path=''):
        
    import pandas as pd
    import numpy as np
    
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
    column_headers = ['datetime', 'r', 'lon', 'Umag', 'Bpol']
    
    data = pd.DataFrame(columns = column_headers)
    
    for filename in filenames:
        
        temp_data = pd.read_csv(data_path + filename, \
                             names=column_headers, \
                             header=0)
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f%')
        
        sub_data = temp_data.loc[(temp_data['datetime'] >= starttime) & (temp_data['datetime'] < finaltime)]
        
        data = pd.concat([data, sub_data])
    
    return(data.reset_index(drop=True))

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
