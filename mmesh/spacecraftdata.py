#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:35:56 2023

@author: mrutala
"""
import datetime as dt
import pandas as pd
import numpy as np
import logging

import sys, os
from pathlib import Path
#p = Path(os.getcwd()).parent / 'separated_code/'
p = Path('/Users/mrutala/projects/MMESH/mmesh/').parent / 'separated_code/'
sys.path.append(str(p))

import read_SWData

scd_log = logging.getLogger('SpacecraftData')

class SpacecraftData:
    
    import datetime as dt
    import pandas as pd
    
    #  Physical constants
    #  probably should be replaced with astropy call or something
    au_to_km = 1.496e8
    rj_to_km = 71492.
    
    def __init__(self, name, 
                 basedir = '/Users/mrutala/'):
        import spiceypy as spice
        import pandas as pd
        
        basedir_spice = basedir+'SPICE/'
        basedir_data = basedir+'Data/'
        
        self.name = name
        self.basedir = basedir
        self.SPICE_METAKERNEL_PLANETARY = basedir_spice+'generic/metakernel_planetary.txt'
        #  Unfortunately, spyder/pyflakes doesn't know that it's using python
        #  3.10+, so it'll show this match statement as an error-- it works!
        match self.name.lower():
            case 'juno':        
                self.SPICE_ID = -61
                self.SPICE_METAKERNEL = basedir_spice+'juno/metakernel_juno.txt'
                #self.basedir_data = '/Users/mrutala/Data/juno/'
            case 'ulysses':     
                self.SPICE_ID = -55
                self.SPICE_METAKERNEL = basedir_spice+'ulysses/metakernel_ulysses.txt'
                #self.basedir_data = '/Users/mrutala/Data/ulysses/'
            case 'galileo':     
                self.SPICE_ID = -77
                self.SPICE_METAKERNEL = basedir_spice+'galileo/metakernel_galileo.txt'
                #self.basedir_data = '/Users/mrutala/Data/galileo/'
            case 'pioneer 10':  
                self.SPICE_ID = -23
                self.SPICE_METAKERNEL = basedir_spice+'pioneer10/metakernel_pioneer10.txt'
                #self.basedir_data = '/Users/mrutala/Data/pioneer/'
            case 'pioneer 11':  
                self.SPICE_ID = -24
                self.SPICE_METAKERNEL = basedir_spice+'pioneer11/metakernel_pioneer11.txt'
                #self.basedir_data = '/Users/mrutala/Data/pioneer/'
            case 'pioneer 12':  
                self.SPICE_ID = -12
                self.SPICE_METAKERNEL = basedir_spice+'pioneer/metakernel_pioneer.txt'
                #self.basedir_data = '/Users/mrutala/Data/pioneer/'
            case 'voyager 1':   
                self.SPICE_ID = -31
                self.SPICE_METAKERNEL = basedir_spice+'voyager/metakernel_voyager1.txt'
                #self.basedir_data = '/Users/mrutala/Data/voyager/'
            case 'voyager 2':   
                self.SPICE_ID = -32
                self.SPICE_METAKERNEL = basedir_spice+'voyager/metakernel_voyager2.txt'
                #self.basedir_data = '/Users/mrutala/Data/voyager/'
             
        data_column_names = ['x_pos', 'y_pos', 'z_pos',
                             'x_vel', 'y_vel', 'z_vel',
                             'l_time',
                             'u_r', 'u_r_err', 'u_n', 'u_n_err', 'u_t', 'u_t_err', 
                             'u_mag', 'u_mag_err',
                             'n_proton', 'n_proton_err', 'n_alpha', 'n_alpha_err',
                             'n_tot', 'n_tot_err', 
                             'p_dyn_proton', 'p_dyn_proton_err', 'p_dyn_alpha', 'p_dyn_alpha_err', 
                             'p_dyn', 'p_dyn_err',
                             'T_proton', 'T_proton_err', 'T_alpha', 'T_alpha_err',
                             'B_r', 'B_r_err', 'B_n', 'B_n_err', 'B_t', 'B_t_err', 
                             'B_mag', 'B_mag_err']
        self.data = pd.DataFrame(columns=data_column_names)
        
        self.starttime = None
        self.stoptime = None
        
    def find_MetaKernelCoverage(self, keep_kernels=False):
        #import datetime as dt
        #import numpy as np
        import spiceypy as spice
        
        import spiceypy_metakernelcontext as spice_mkc
        
        
        spice.furnsh(self.SPICE_METAKERNEL)
        starttime, stoptime = spice_mkc.kernelrange(self.SPICE_ID, kw_verbose=False)
        self.starttime = starttime
        self.stoptime = stoptime
        
        #  !!!! Hacky fix for a weird bug where the stoptime for Pioneers 10 and 11
        #  is apparently earlier than what the spk kernels dictate...
        if 'pioneer' in self.name.lower():
            self.stoptime -= dt.timedelta(hours=1)
            
        if not keep_kernels: spice.kclear()
    
    def make_timeseries(self, starttime=None, stoptime=None, timedelta=dt.timedelta(days=1),
                        returning = False):
        import numpy as np
        
        if starttime == None:
            if self.starttime == None:
                raise ValueError('No start time found.')
        else: self.starttime = starttime
        
        if stoptime == None:
            if self.stoptime == None:
                raise ValueError('No stop tme found.')
        else: self.stoptime = stoptime
        
        datetimes = np.arange(self.starttime, self.stoptime, timedelta).astype(dt.datetime)
        
        if self.data.empty:
            self.data = pd.concat([self.data, pd.DataFrame(index=datetimes)])
        else:
            print('Spacecraft DataFrame already filled. Returning instead...')
            returning = True
        #  In principle, this function could reindex the dataframe if it already exists
        #  But it would be unwieldy to pass a bunch of method terms, i.e. mean(), into it
        #  So I'm just going to return the datetimes if data exists...
        if returning:
            return pd.Index(datetimes)
        else:
            return
    
    def find_state(self, reference_frame, observer, keep_kernels=False):
        #import datetime as dt
        import numpy as np
        import spiceypy as spice
        import pandas as pd
        
        # Add reference frame and observer to self
        self.reference_frame = reference_frame
        self.observer = observer
        
        #  Essential solar system and timekeeping kernels
        spice.furnsh(self.SPICE_METAKERNEL_PLANETARY)
        
        #  Custom helio frames
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/fk/SolarFrames.tf')
        
        spice.furnsh(self.SPICE_METAKERNEL)
        
        datetimes_str = [time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in self.data.index]
        if len(datetimes_str) > 0:
            ets = spice.str2et(datetimes_str)
            sc_state, sc_lt = spice.spkezr(str(self.SPICE_ID), ets, reference_frame, 'NONE', observer)
            sc_state_arr = np.array(sc_state)
            sc_state_dataframe = pd.DataFrame(data=sc_state_arr, 
                                              index=self.data.index,
                                              columns=['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel'])
            sc_state_dataframe['l_time'] = sc_lt
            if not keep_kernels: spice.kclear()
            
            self.data.update(sc_state_dataframe)
            #self.data = pd.concat([self.data, sc_state_dataframe.reindex(columns=self.data.columns)], 
            #                      axis=0)
            
    def find_StateToEarth(self, keep_kernels=False):
        #import datetime as dt
        import numpy as np
        import spiceypy as spice
        import pandas as pd
        
        #  Essential solar system and timekeeping kernels
        spice.furnsh(self.SPICE_METAKERNEL_PLANETARY)
        
        #  Custom helio frames
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/fk/SolarFrames.tf')
        
        spice.furnsh(self.SPICE_METAKERNEL)
        
        datetimes_str = [time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in self.data.index]
        if len(datetimes_str) > 0:
            ets = spice.str2et(datetimes_str)
            
            sc_pos, sc_lt = spice.spkpos(str(self.SPICE_ID), ets, 'SUN_INERTIAL', 'NONE', 'SUN')
            sc_pos_rlonlat = np.array([spice.reclat(pos) for pos in sc_pos])
            
            eth_pos, eth_lt = spice.spkpos('EARTH BARYCENTER', ets, 'SUN_INERTIAL', 'NONE', 'SUN')
            eth_pos_rlonlat = np.array([spice.reclat(pos) for pos in eth_pos])
            
            del_r_TSE = sc_pos_rlonlat[:,0] - eth_pos_rlonlat[:,0]
            del_lon_TSE = (sc_pos_rlonlat[:,1] - eth_pos_rlonlat[:,1])*180/np.pi
            del_lat_TSE = (sc_pos_rlonlat[:,2] - eth_pos_rlonlat[:,2])*180/np.pi
            
            if not keep_kernels: spice.kclear()
            
            df = pd.DataFrame.from_dict({'del_r'  : del_r_TSE,
                                         'del_lon': np.abs(np.abs(del_lon_TSE + 180) % 360 - 180),
                                         'del_lat': np.abs(np.abs(del_lat_TSE + 180) % 360 - 180)})
            
            
            df.index = self.data.index
            
            return df
        
    def read_processeddata(self, starttime=None, stoptime=None, 
                           everything=False, strict=True, resolution=None, combined=False):
        import read_SWData
        import spiceypy as spice
        
        #  everything keyword reads in all available data, overwriting starttime and stoptime
        if everything == True:
            self.find_lifetime()
        else:
            if starttime != None: self.starttime = starttime
            if stoptime != None: self.stoptime = stoptime
        
        processed_data = read_SWData.read(self.name, 
                                          self.starttime, self.stoptime, 
                                          basedir=self.basedir + 'Data/',
                                          resolution=resolution, combined=combined)
        
        if len(processed_data.dropna(axis='index')) == 0:
            fmt = '%Y-%m-%d %H:%M:%S.%f'
            scd_log.warning('No data found between {} and {}. '.format(starttime.strftime(fmt), stoptime.strftime(fmt)) +
                            'If you expect data in this range, please check again') #+
                            #'the data files in {}'.format(self.basesdir))
            return
        
        #  If self.data exists, add to it, else, make it
        if strict:
            processed_data = processed_data.reindex(columns=self.data.columns)
            # try:
            #     processed_data = processed_data.reindex(columns=self.data.columns)
            # except ValueError:
            #     print('VALUE ERROR!')
            #     print(processed_data.columns)
            #     print(self.data.columns)
            #     print(processed_data.index.has_duplicates)
            
        #self.data = pd.concat([self.data, processed_data], 
        #                      axis=0)
        #self.data = processed_data.combine_first(self.data)
        desired_column_order = list(self.data.columns)
        unmatched_columns = list(set(processed_data.columns) - set(self.data.columns))
        unmatched_columns.sort()
        
        self.data = self.data.combine_first(processed_data)
        
        #  Put the dataframe back in order
        self.data = self.data[(desired_column_order + unmatched_columns)]
        
        self.data_type = 'processed'
        #self.datetimes = processed_data.index.to_pydatetime()
    
    def find_subset(self, coord1_range=None, coord2_range=None, coord3_range=None, transform=None):
        import numpy as np
        import spiceypy as spice
        import spacecraftdata as SpacecraftData
        
        #  Need coordinates to do this
        if len(self.data.dropna(axis='index', subset=['x_pos', 'y_pos', 'z_pos'])) == 0:
            scd_log.warning('No trajectory information available; ' 
                            'trajectory subsetting is not possible.')
            return
        
        match transform.lower():
            case None: spice_transform = lambda x, y, z : (x, y, z)
            case 'reclat': spice_transform = spice.reclat
            case 'recsph': spice_transform = spice.recsph
            
        #  Convert
        spacecraft_coords = [spice_transform(row[['x_pos', 'y_pos', 'z_pos']].to_numpy(dtype='float64')) for index, row in self.data.iterrows()]
        spacecraft_coords = np.array(spacecraft_coords)
        
        if np.all(coord1_range == None): 
            coord1_range = (np.min(spacecraft_coords[:,0])-1, np.max(spacecraft_coords[:,0])+1)
        if np.all(coord2_range == None): 
            coord2_range = (np.min(spacecraft_coords[:,1])-1, np.max(spacecraft_coords[:,1])+1)
        if np.all(coord3_range == None): 
            coord3_range = (np.min(spacecraft_coords[:,2])-1, np.max(spacecraft_coords[:,2])+1)
        
        #  !!! could consider adding an option for <> vs <= >=
        criteria = np.where((spacecraft_coords[:,0] > coord1_range[0]) &
                            (spacecraft_coords[:,0] < coord1_range[1]) &
                            (spacecraft_coords[:,1] > coord2_range[0]) &
                            (spacecraft_coords[:,1] < coord2_range[1]) &
                            (spacecraft_coords[:,2] > coord3_range[0]) &
                            (spacecraft_coords[:,2] < coord3_range[1]) )
                            
                            
        self.data = self.data.iloc[criteria]
        return
    
        def limit_toSW(self):
            #  Return only measurements in the solar wind
            #  (i.e., outside of any magnetospheres)
            
            #   Terrestrial Magnetosphere ()
            
            #   Jovian Magnetopause (Joy+ 2002 Model)

            
            return
            
            
            
            