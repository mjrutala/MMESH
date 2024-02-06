#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:12:45 2024

@author: mrutala
"""

from types import SimpleNamespace

def get_plotparam(self, label, param):
     
    return
 
def gpp():
    return get_plotparam()
 
def set_plotparam(self, label, param):
    
    adjustments = {'left': 0.1, 'right':0.95, 'bottom':0.2, 'top':0.95, 
                   'wspace':0.1, 'hspace':0.0}
    
    d = {'data_color': '', 'data_marker': '',
         'model_colors': {'ensemble': 'xkcd:cyan'}, 'model_markers': {'ensemble': 'o'},
         'figsize': (6, 4.5), 'adjustments': adjustments}
    self.plotparams = SimpleNamespace(**d)
    
    return


