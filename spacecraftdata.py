#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:35:56 2023

@author: mrutala
"""

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
                self.SPICE_METAKERNEL = basedir_spice+'pioneer/metakernel_pioneer.txt'
                #self.basedir_data = '/Users/mrutala/Data/pioneer/'
            case 'pioneer 11':  
                self.SPICE_ID = -24
                self.SPICE_METAKERNEL = basedir_spice+'pioneer/metakernel_pioneer.txt'
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
                             'p_dyn_proton', 'p_dyn_proton_err', 'p_dyn_alpha', 'p_dyn_alpha_err', 
                             'p_dyn', 'p_dyn_err',
                             'T_proton', 'T_proton_err', 'T_alpha', 'T_alpha_err',
                             'B_r', 'B_r_err', 'B_n', 'B_n_err', 'B_t', 'B_t_err', 
                             'B_mag', 'B_mag_err']
        self.data = pd.DataFrame(columns=data_column_names)
        
    def find_timerange(self, starttime, stoptime, timedelta):
        #import datetime as dt
        import numpy as np
        
        self.datetimes = np.arange(starttime, stoptime, timedelta).astype(dt.datetime)
    
    def find_state(self, reference_frame, observer):
        #import datetime as dt
        import numpy as np
        import spiceypy as spice
        import pandas as pd
        
        # Add reference frame and observer to self
        self.reference_frame = reference_frame
        self.observer = observer
        
        #  Essential solar system and timekeeping kernels
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
        
        #  Custom helio frames
        spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
        
        spice.furnsh(self.SPICE_METAKERNEL)
        
        datetimes_str = [time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in self.datetimes]
        ets = spice.str2et(datetimes_str)
        sc_state, sc_lt = spice.spkezr(str(self.SPICE_ID), ets, reference_frame, 'NONE', observer)
        sc_state_arr = np.array(sc_state)
        sc_state_dataframe = pd.DataFrame(data=sc_state_arr, 
                                          index=self.datetimes,
                                          columns=['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel'])
        sc_state_dataframe['l_time'] = sc_lt
        spice.kclear()
        
        self.data.update(sc_state_dataframe)
        #self.data = pd.concat([self.data, sc_state_dataframe.reindex(columns=self.data.columns)], 
        #                      axis=0)
        
    def read_processeddata(self, starttime=dt.datetime.now(), stoptime=dt.datetime.now(), 
                           everything=False, strict=True):
        import read_SWData
        import pandas as pd
        import spiceypy as spice
        
        import spiceypy_metakernelcontext as spice_mkc
        
        #  everything keyword read in all available data, overwriting starttime and stoptime
        if everything == True:
            spice.furnsh(self.SPICE_METAKERNEL)
            starttime, stoptime = spice_mkc.kernelrange(self.SPICE_ID, kw_verbose=False)
            spice.kclear()
        
        processed_data = read_SWData.read(self.name, starttime, stoptime, basedir=self.basedir + 'Data/')
        
        #  If self.data exists, add to it, else, make it
        if strict:
            processed_data = processed_data.reindex(columns=self.data.columns)
            
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
        self.datetimes = processed_data.index.to_pydatetime()
    
        #self.start_date = None
        #self.stop_date = None
        #self.date_range = None
        #self.date_delta = None
        #self.datetimes = None
        #self.ephemeris = None