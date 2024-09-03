#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:12:45 2024

@author: mrutala
"""

from types import SimpleNamespace
import matplotlib.pyplot as plt

class visualization:
    
    def initialize_plotprops(self):
        adjustments = {'left': 0.15, 'right':0.95, 'bottom':0.1, 'top':0.95, 
                       'wspace':0.1, 'hspace':0.0}
        d = {'color': {}, 
             'marker': {},
             'number_of_props_assigned': 0,
             'figsize': (6, 4.5), 
             'adjustments': adjustments}
        self.plotprops = d
        return
        
    def set_plotprops(self, prop, label, value): 
        self.plotprops[prop][label] = value
        return
    
    def get_plotprops(self, prop, label):
        
        try:
            value = self.plotprops[prop][label]
        except KeyError:
            if prop == 'color':
                #   Find an unused color, set it, and return it
                colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']
                value = colorlist[self.plotprops['number_of_props_assigned'] % len(colorlist)]
                self.plotprops['number_of_props_assigned'] += 1
                self.set_plotprops(prop, label, value)
            if prop == 'marker':
                value = 'X'
                self.set_plotprops(prop, label, value)
            
        return value
     
    def gpp(self, prop, label):
        return self.get_plotprops(prop, label)
     
    
    
    
    
    
    
    
