#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:09:27 2025

@author: mrutala
"""

import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW

import sys
sys.path.append('/Users/mrutala/projects/HUXt/code/')
sys.path.append('/Users/mrutala/projects/OHTransients/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_atObserver as hao
import multihuxt_readers as mr

omni_data = mr.SolarWindData('omni', datetime.datetime(1980, 1, 1), datetime.datetime(2025, 1 , 1))

test_start = datetime.datetime(2024, 10, 1)
test_stop = datetime.datetime(2024, 11, 1)
max_test_chunks = 2*24
nearest_neighbors = 100

train_df = omni_data.data.query("@test_start > Epoch | @test_stop <= Epoch")
test_df = omni_data.data.query("@test_start <= Epoch < @test_stop")

if max_test_chunks < len(test_df):
    test_chunks = max_test_chunks
else:
    test_chunks = len(test_df)

query_var = 'n'
train_var = 'U'
train_arr = sliding_window_view(train_df[train_var].to_numpy(), test_chunks)
# Drop rows with NaNs
train_nonnan = ~np.isnan(train_arr.sum(axis=1))
train_arr = train_arr[train_nonnan]

query_arr = sliding_window_view(train_df[query_var].to_numpy(), test_chunks)[train_nonnan]

# Build a KD tree
tree = KDTree(train_arr)

test_arr = sliding_window_view(test_df[train_var].to_numpy(), test_chunks)
output025_arr = np.zeros_like(test_df[train_var].to_numpy())
output500_arr = np.zeros_like(test_df[train_var].to_numpy())
output975_arr = np.zeros_like(test_df[train_var].to_numpy())
count = np.zeros_like(test_df[train_var].to_numpy())
for i, test_row in enumerate(test_arr):
    
    distances, indices = tree.query(test_row, k=nearest_neighbors)
    weights = (1/(distances/test_chunks))**10
    weights = weights/(weights.sum())
    
    #
    train_members = train_arr[indices]
    train_ensemble = train_members.mean(axis=0)
    
    query_members = query_arr[indices]
    wq = DescrStatsW(data=query_members, weights=weights)
    query_025, query_500, query_975 = wq.quantile([0.05, 0.5, 0.95], return_pandas=False)
    
    window_size = len(test_row)
    output025_arr[i : i + window_size] += query_025
    output500_arr[i : i + window_size] += query_500
    output975_arr[i : i + window_size] += query_975
    count[i:i+window_size] += 1
    
output025 = output025_arr/count
output500 = output500_arr/count
output975 = output975_arr/count
    
fig, axs = plt.subplots(nrows=2)

# Plot the training variable
# axs[0].plot(test_df[train_var])
# axs[0].plot(test_df.index, train_ensemble, color='xkcd:red')
# for member in train_members:
#     axs[0].plot(test_df.index, member, color='xkcd:red', alpha=0.01, lw=0.5)
    
# Plot the query variable
axs[1].plot(test_df[query_var])
axs[1].plot(test_df.index, output500, color='xkcd:crimson')
axs[1].fill_between(test_df.index, output975, output025, color='xkcd:crimson', alpha=0.33)
# for member in query_members:
#     axs[1].plot(test_df.index, member, color='xkcd:red', alpha=0.03, lw=0.5)  