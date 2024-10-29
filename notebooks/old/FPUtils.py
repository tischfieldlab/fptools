import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths
import tdt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
import scipy.stats as stats
import glob
import os

isosbestic = '_415A'
dopa = '_465A'

CHANNEL      = 1
TRANGE       = [-5, 10] 
BASELINE_PER = [-5, 2]

def readFile(blockpath):
    data = tdt.read_block(blockpath)
    time = np.linspace(1,len(data.streams[dopa].data), len(data.streams[dopa].data))/data.streams[dopa].fs
    t = 5
    inds = np.where(time>t)
    ind = inds[0][0]
    time = time[ind:] # go from ind to final index
    data.streams[dopa].data = data.streams[dopa].data[ind:]
    data.streams[isosbestic].data = data.streams[isosbestic].data[ind:]
    N = 10 # Average every 10 samples into 1 value
    F415 = []
    F465 = []
    DSFactor = (data.streams[dopa].fs)/N

    for i in range(0, len(data.streams[dopa].data), N):
        F465.append(np.mean(data.streams[dopa].data[i:i+N-1])) # This is the moving window mean
    data.streams[dopa].data = F465

    for i in range(0, len(data.streams[isosbestic].data), N):
        F415.append(np.mean(data.streams[isosbestic].data[i:i+N-1]))
    data.streams[isosbestic].data = F415

    #decimate time array to match length of demodulated stream
    time = time[::N] # go from beginning to end of array in steps on N
    time = time[:len(data.streams[dopa].data)]

    x = np.array(data.streams[isosbestic].data)
    y = np.array(data.streams[dopa].data)
    bls = np.polyfit(x, y, 1)
    Y_fit_all = np.multiply(bls[0], x) + bls[1]
    Y_dF_all = y - Y_fit_all
    dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
    return data, dFF, DSFactor, time


def CreateBouts(path):
    n = 3 # how many consecutive pokes is a bout
    bout_freq = n-1
    BOUT_TIME_THRESHOLD = 5
    UnNP_diff_indices_offset= []
    data, dFF, streamFactor = readFile(path)
    # for i in data.epocs.UnNP.onset:
    #     if data.epocs.UnNP.onset > (len(dFF)/101.8):

    UnNP_diff = np.diff(data.epocs.UnNP.onset)
    UnNP_diff_indices = np.where(UnNP_diff >= BOUT_TIME_THRESHOLD)[0]
    
    for i in range(len(UnNP_diff_indices)):
        try:
            UnNP_diff_indices_offset.append(UnNP_diff_indices[i+1]-1)
        except IndexError:
            UnNP_diff_indices_offset.append(len(UnNP_diff)-1)
            continue

    bout_indx = np.where((UnNP_diff_indices_offset - UnNP_diff_indices) >= bout_freq)[0]
    UnNP_diff_indices=np.array(UnNP_diff_indices)
    UnNP_diff_indices_offset = np.array(UnNP_diff_indices_offset)
    onset_indx = UnNP_diff_indices[bout_indx]
    offset_indx = UnNP_diff_indices_offset[bout_indx]
    onset_frames = np.array(data.epocs.UnNP.onset[onset_indx]*streamFactor, dtype=int)
    offset_frames = np.array(data.epocs.UnNP.onset[offset_indx]*streamFactor, dtype=int)

    aucsPerTime = []
    aucs = []
    for on, off in zip(onset_frames, offset_frames):
        end = off+102 #first UnNp to last UnNP in bout + 1 second
        if end < len(dFF):
            aucsPerTime.append(auc(range(end-on), dFF[on:end])/((end-on)/streamFactor)) 
            aucs.append(auc(range(end-on), dFF[on:end]))
        elif off > len(dFF):
            fin = int(len(dFF))-1
            aucsPerTime.append(auc(range(fin-on), dFF[on:fin])/((fin-on)/streamFactor)) 
            aucs.append(auc(range(fin-on), dFF[on:fin]))
    diff_arr = []
    for on,off in zip(onset_indx, offset_indx):
        diff_arr.append((off-on)+1)
    # aucPerEpoc = []
    # for i in range(len(aucsPerTime)):
    #     aucPerEpoc.append(aucs[i]/diff_arr[i])

    return aucsPerTime, diff_arr

def StoreBoutstoDF(epocPertime, genotype):
    df = pd.DataFrame()

    for i in range(len(epocPertime)):

        df.loc[i, 'auc per time']= epocPertime[i]
        df.loc[i, 'genotype'] = genotype
        # df.loc[i, 'auc per epoc'] = epoc[i]
        if i <= 4:
            # df. loc[i, 'first 5 bouts: auc per epoc'] = epoc[i]
            df. loc[i, 'first 5 bouts: auc per time'] = epocPertime[i]
        elif i >= (len(epocPertime)-5):
            # df. loc[i, 'last 5 bouts: auc per epoc'] = epoc[i]
            df. loc[i, 'last 5 bouts: auc per time'] = epocPertime[i]
    return df

def BoutFrequency(path, genotype):
    data, dFF, streamFactor = readFile(path)
    rnp_on = data.epocs.RNP_.onset
    rmg_on = data.epocs.RMG_.onset
    unp_on = data.epocs.UnNP.onset
    binned_df = pd.DataFrame()

    vals = []
    for i in range(len(rnp_on)): # each block represents time between rewarded magazine and the subsequent rewarded nose poke
        bin_values = []
        for j in range(len(unp_on)):
            if ( unp_on[j] < rmg_on[len(rmg_on)-1]): # addresses boundary of last rewarded mag entry               
                if (i == 0) and (unp_on[j] < rnp_on[i]): # time before first rnp since no rmg
                    bin_values.append(unp_on[j])
                    binned_df.loc[j,'block'] = i
                    binned_df.loc[j,'Unrewarded_NP'] = unp_on[j]
                    binned_df.loc[j, 'genotype'] = genotype
                elif (unp_on[j] < rnp_on[i]) and (unp_on[j] > rmg_on[i-1]):
                    bin_values.append(unp_on[j]-rmg_on[i-1])
                    binned_df.loc[j,'block'] = i
                    binned_df.loc[j,'Unrewarded_NP'] = (unp_on[j]-rmg_on[i-1])   
                    binned_df.loc[j, 'genotype'] = genotype
        if i==0:
            vals.append(rnp_on[i])
        else:
            vals.append(rnp_on[i]-rmg_on[i-1])
    return binned_df, vals

def binEpocs(path, genotype, epoch_type):
    data, _, _  = readFile(path)
    total_time = (data.info.duration.seconds) + (data.info.duration.microseconds)*1e-6
    nbins = 10
    interval = total_time/nbins
    # binned_epocs = []
    onsets = data.epocs[epoch_type].onset

    another_binned_df= pd.DataFrame()

    x = interval
    n=0
    for i in range(len(onsets)):
        while True:
            if onsets[i] <= x:
                # temp_array.append(onsets[i])
                another_binned_df.loc[i, 'bin'] = n
                another_binned_df.loc[i, f'{epoch_type} onset'] = onsets[i]
                another_binned_df.loc[i, 'genotype'] = genotype
                break
            elif onsets[i]>x:
                x += interval
                # print(x)
                n+=1
                # binned_epocs.append(temp_array)
                # temp_array = []
                continue
            
    return another_binned_df



def create_binned_df(tanks, epoch_type, tbefore, tafter, genotype):
    whole_df = pd.DataFrame()
    for file in glob.glob(tanks):
        df = pd.DataFrame()
        df['Animal'] = ''
        df['bin'] = ''
        df['mean binned stream'] = ''
        df['mean binned stream'] = df['mean binned stream'].astype(object)
        binnedepocs = binEpocs(file, genotype=genotype, epoch_type=epoch_type)
        binned_streams = []
        _,dFF,dsFactor = readFile(file)

        for i in binnedepocs[f'{epoch_type} onset']:
            x = int(i*dsFactor)
            if (i+(tafter*dsFactor)) > len(dFF):
                continue
            binned_streams.append(dFF[x-(int(tbefore*dsFactor)) : x + (int(tafter*dsFactor))])
            binnedepocs['stream'] = ''
            binnedepocs['stream'] = binnedepocs['stream'].astype(object)

        for i in range(len(binned_streams)):
            binnedepocs.at[i, 'stream'] = binned_streams[i] 

        for i in range(10):
            df.loc[i, 'Animal'] = os.path.basename(file)
            df.loc[i, 'bin'] = i
            bin = np.mean(binnedepocs['stream'].loc[binnedepocs['bin']==i], axis=0)
            # print(bin)
            df.at[i, 'mean binned stream'] = bin
        whole_df = pd.concat([whole_df, df], ignore_index=True)
           
    return whole_df
