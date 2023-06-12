#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:35:56 2023

@author: mrutala
"""

class SpacecraftData:
    def __init__(self, name):
        
        import spiceypy as spice
        
        self.name = name
        self.SPICE_ID = 0
        self.SPICE_METAKERNEL = ''
        
        self.start_date = None
        self.stop_date = None
        self.date_range = None
        self.date_delta = None
        self.datetimes = None
        
        self.ephemeris = None
