#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:25:08 2023

@author: mrutala
"""

"""
Check all currently loaded SPK kernels for the specified object, and return coverage
"""
def kernelrange(obj, kw_verbose=False):
    # =============================================================================
    # 
    # =============================================================================
    import spiceypy as spice
    import datetime as dt
    
    count = spice.ktotal('SPK')
    if kw_verbose: print('Total SPK kernels loaded: ' + str(count))

    kernelfile_list = list()
    
    for i in range(0, count):
        [file, type, source, handle] = spice.kdata(i, 'SPK')
        kernelfile_list.append(file)
        if kw_verbose: print(file)
    
    date_list = list()
    
    for kernelfile in kernelfile_list:
        cover = spice.spkcov(kernelfile, obj)
        
        if len(cover) != 0:
            for j in range(int(float(len(cover))/2)):
                #
                daterange_TDB = spice.wnfetd(cover, j)
                
                #
                daterange = [dt.datetime(2000, 1, 1, 12) + dt.timedelta(seconds = s) for s in daterange_TDB]
                
                #
                date_list.extend(daterange)
                
    if kw_verbose: 
        print("Start : " + min(date_list).strftime('%Y-%m-%d %H:%M:%S.%f'))
        print("Stop  : " + max(date_list).strftime('%Y-%m-%d %H:%M:%S.%f'))
    
    return((min(date_list), max(date_list)))