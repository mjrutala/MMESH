#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:17:07 2023

@author: mrutala
"""

def MSWIM2Dlike(startdate, finaldate):
    #  Things that will be input parameters
    name = 'Ulysses'
    
    import spacecraftdata as SpacecraftData
    
    sc = SpacecraftData.SpacecraftData(name)
    
    # !!! These should go inside SpacecraftData and be determined automatically
    sc.SPICE_ID = -55
    sc.SPICE_METAKERNEL = '/Users/mrutala/SPICE/ulysses/metakernel_ulysses.txt'
    
    sc.startdate = startdate
    sc.finaldate = finaldate