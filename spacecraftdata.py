#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:35:56 2023

@author: mrutala
"""

class SpacecraftData:
    #  Physical constants
    #  probably should be replaced with astropy call or something
    au_to_km = 1.496e8
    rj_to_km = 71492.
    
    def __init__(self, name, 
                 basedir = '/Users/mrutala/'):
        import spiceypy as spice
        
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
                self.SPICE_ID = -32
                self.SPICE_METAKERNEL = basedir_spice+'voyager/metakernel_voyager1.txt'
                #self.basedir_data = '/Users/mrutala/Data/voyager/'
            case 'voyager 2':   
                self.SPICE_ID = -32
                self.SPICE_METAKERNEL = basedir_spice+'voyager/metakernel_voyager2.txt'
                #self.basedir_data = '/Users/mrutala/Data/voyager/'
        
    def find_timerange(self, starttime, stoptime, timedelta):
        import datetime as dt
        import numpy as np
        
        self.datetimes = np.arange(starttime, stoptime, timedelta).astype(dt.datetime)
    
    def find_state(self, reference_frame, observer):
        import datetime as dt
        import numpy as np
        import spiceypy as spice
        import pandas as pd
        
        #  Essential solar system and timekeeping kernels
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
        spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
        
        spice.furnsh(self.SPICE_METAKERNEL)
        
        datetimes_str = [time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in self.datetimes]
        ets = spice.str2et(datetimes_str)
        
        sc_state, sc_lt = spice.spkezr(str(self.SPICE_ID), ets, reference_frame, 'NONE', observer)
        sc_state_arr = np.array(sc_state)
        sc_state_dataframe = pd.DataFrame(data=sc_state_arr, 
                                          index=self.datetimes,
                                          columns=['xpos', 'ypos', 'zpos', 'xvel', 'yvel', 'zvel'])
        print(sc_state_dataframe)
        spice.kclear()
        
        #  !!! Check is self.data is a dataframe, if not, make it
        try:
            self.data = pd.concat([sc_state_dataframe, self.data], axis=1)
        except AttributeError:
            self.data = sc_state_dataframe
        
        
    def read_processeddata(self, starttime, stoptime):
        import read_SWData
        import pandas as pd
        
        processed_data = read_SWData.Ulysses(starttime, stoptime, basedir=self.basedir + 'Data/')
        
        #  If self.data exists, add to it, else, make it
        try:
            self.data = pd.concat([self.data, processed_data], axis=1)
        except AttributeError:
            self.data = processed_data
        
        self.data_type = 'processed'
        self.datetimes = processed_data.index.to_pydatetime()
        
        
    
        #self.start_date = None
        #self.stop_date = None
        #self.date_range = None
        #self.date_delta = None
        #self.datetimes = None
        #self.ephemeris = None