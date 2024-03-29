# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from ROOT import TFile
from root_numpy import hist2array

import numpy as np
from os import listdir
from os.path import isfile, join
import os, json
from collections import defaultdict
import datetime
import copy

def count_events(folder, key):
    nevents = 0
    dlist = [f for f in listdir(folder) if key in f]
    dlist.sort()
    for dirname in dlist:
        flist = [f for f in listdir(folder + '/' + dirname) if '_y.npy' in f]
        for fname in flist:
            d = np.load(folder + '/' + dirname + '/' + fname)
            nevents += d.shape[0]
    return nevents

def get_patch_size(folder):
    dlist = [f for f in listdir(folder) if '' in f]
    flist = [f for f in listdir(folder + '/' + dlist[0]) if '_x.npy' in f]
    d = np.load(folder + '/' + dlist[0] + '/' + flist[0])
    return d.shape[1], d.shape[2]

def get_event_bounds(A, drift_margin = 0):
    # get center with 99% of signal
    cum = np.cumsum(np.sum(A, axis=0))
    start_ind = np.max([0, np.where(cum > cum[-1]*0.005)[0][0] - drift_margin])
    end_ind = np.min([A.shape[1], np.where(cum > cum[-1]*0.995)[0][0] + drift_margin])
    return start_ind, end_ind

def get_data(folder, fname, drift_margin = 0, crop = True, blur = None, 
             white_noise = 0, coherent_noise = 0):
    print 'Reading', fname
    try:
        if isinstance(folder, TFile):  # read from ROOT file
            A_raw     = hist2array(folder.Get(fname + '_raw'))
            A_deposit = hist2array(folder.Get(fname + '_deposit'))
            A_pdg     = hist2array(folder.Get(fname + '_pdg'))
        else:                # read text files
            A_raw     = np.genfromtxt(folder+'/'+fname + '.raw', delimiter=' ', 
                                      dtype=np.float32)
            A_deposit = np.genfromtxt(folder+'/'+fname + '.deposit', 
                                      delimiter=' ', dtype=np.float32)
            A_pdg     = np.genfromtxt(folder+'/'+fname + '.pdg', delimiter=' ', 
                                      dtype=np.int32)
    except:
        print 'Bad event, return empty arrays'
        return None, None, None, None, None

    if A_raw.shape[0] < 8 or A_raw.shape[1] < 8: 
        return None, None, None, None, None

    test_pdg = np.sum(A_pdg)
    test_dep = np.sum(A_deposit)
    test_raw = np.sum(A_raw)
    if test_raw == 0.0 or test_dep == 0.0 or test_pdg == 0: 
        return None, None, None, None, None

    print test_raw, test_dep, test_pdg
    #assert np.sum(A_deposit) > 0
    # get main event body (99% signal)
    if crop:
        evt_start_ind, evt_stop_ind = get_event_bounds(A_deposit, drift_margin)
        A_raw     = A_raw[:,evt_start_ind:evt_stop_ind]
        A_deposit = A_deposit[:,evt_start_ind:evt_stop_ind]
        A_pdg     = A_pdg[:,evt_start_ind:evt_stop_ind]
    else:
        evt_start_ind = 0
        evt_stop_ind = A_raw.shape[1]
    print evt_start_ind, evt_stop_ind

    A_raw = applyBlur(A_raw, blur)
    A_raw = addWhiteNoise(A_raw, white_noise)
    A_raw = addCoherentNoise(A_raw, coherent_noise)

    deposit_th_ind = A_deposit < 2.0e-5
    A_pdg[deposit_th_ind] = 0
    tracks = A_pdg.copy()
    showers = A_pdg.copy()
    tracks[(A_pdg & 0x0FFF) == 11] = 0
    tracks[tracks > 0]   = 1
    showers[(A_pdg & 0x0FFF) != 11] = 0
    showers[showers > 0] = 1
    return A_raw, A_deposit, A_pdg, tracks, showers


# MUST give the same result as nnet::DataProviderAlg::applyBlur() in 
# PointIdAlg/PointIdAlg.cxx
def applyBlur(a, kernel):
    if kernel is None or kernel.shape[0] < 2: return a

    margin_left = kernel.shape[0] >> 1
    margin_right = kernel.shape[0] - margin_left - 1
    src = np.copy(a)
    for w in range(margin_left, a.shape[0] - margin_right):
        for d in range(a.shape[1]):
            s = 0.
            for i in range(kernel.shape[0]):
                s += kernel[i] * src[w + i - margin_left, d]
            a[w, d] = s
    return a

# MUST give the same result as nnet::DataProviderAlg::addWhiteNoise() in 
# PointIdAlg/PointIdAlg.cxx
# so sigma here should be "effective": divided by ADC scaling (constant, 10) and 
# downsampling factor
def addWhiteNoise(a, sigma):
    if sigma is None or sigma == 0: return a

    a += np.random.normal(0, sigma, a.shape)

    return a

# MUST give the same result as nnet::DataProviderAlg::addCoherentNoise() in 
# PointIdAlg/PointIdAlg.cxx
# so sigma here should be "effective": divided by ADC scaling (constant, 10) and 
# downsampling factor
def addCoherentNoise(a, sigma):
    if sigma is None or sigma == 0: return a

    a += np.random.normal(0, sigma, a.shape)

    amps1 = np.random.normal(1, 0.1, a.shape[0]);
    amps2 = np.random.normal(1, 0.1, 1 + (a.shape[0] >> 5));

    group_amp = 1
    for w in range(a.shape[0]):
        if (w & 31) == 0:
            noise = np.random.normal(0, sigma, a.shape[1])
            group_amp = amps2[w >> 5]
        a[w] += group_amp * amps1[w] * noise

    return a

# MUST give the same result as nnet::PointIdAlg::bufferPatch() in 
# PointIdAlg/PointIdAlg.cxx
def get_patch(a, wire, drift, wsize, dsize):
    halfSizeW = wsize / 2;
    halfSizeD = dsize / 2;

    w0 = wire - halfSizeW;
    w1 = wire + halfSizeW;

    d0 = drift - halfSizeD;
    d1 = drift + halfSizeD;

    patch = np.zeros((wsize, dsize), dtype=np.float32)

    wpatch = 0
    for w in range(w0, w1):
        if w >= 0 and w < a.shape[0]:
            dpatch = 0
            for d in range(d0, d1):
                if d >= 0 and d < a.shape[1]:
                    patch[wpatch,dpatch] = a[w,d];
                dpatch += 1
        wpatch += 1
    
    return patch

def get_vertices(A):
    # rather not more than 25% of plane filled with vertices
    max_count = A.shape[0]*A.shape[1] / 4 
    vtx = np.zeros((max_count, 3), dtype=np.int32)
    nvtx = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if nvtx >= max_count: break
            if (A[i,j] & 0xFF000000) > 0:
                t = A[i,j] >> 24
                v = np.zeros(3)
                v[0] = i
                v[1] = j
                v[2] = t
                vtx[nvtx] = v
                nvtx += 1
    return vtx[:nvtx]

def get_nu_vertices(A):
    max_count = 10 # 10 vertices per view shoud be enough...
    vtx = np.zeros((max_count, 3), dtype=np.int32)
    nvtx = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if nvtx >= max_count: break
            if (A[i,j] & 0xFF0000) > 0:
                t = (A[i,j] >> 16) & 0xFF
                v = np.zeros(3)
                v[0] = i
                v[1] = j
                v[2] = t
                vtx[nvtx] = v
                nvtx += 1
    return vtx[:nvtx]

def shuffle_in_place(a, b):
    assert len(a) == len(b)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def read_config(cfgname):
    config = None
    with open(cfgname, 'r') as fin:
        config = json.loads(fin.read());
    if config is None:
        print 'This script requires configuration file: config.json'
        exit(1)
    return config

def save_model(model, name):
    print (datetime.datetime.now().strftime("%y%m%d-%H:%M"))
    name += '_'
    name += datetime.datetime.now().strftime("%y%m%d-%H:%M")
    print (name)
    with open(name + '_architecture.json', 'w') as f: 
      f.write(model.to_json())
    model.save_weights(name + '_weights.h5', overwrite = True)

def get_unet_data(inputDir, batch, n_channels):

  y_data = np.load(inputDir + '/truth_' + str(batch) + '.npy')
  y_data = y_data.reshape((y_data.shape[0], y_data.shape[1], y_data.shape[2], 1))
  
  n_patches = y_data.shape[0]
  patch_w   = y_data.shape[1]
  patch_h   = y_data.shape[2]
  
  if n_channels == 3:
    x_data = np.zeros((n_patches, patch_w, patch_h, 3))
    x_data[...,0] = np.load(inputDir + '/wire_' + str(batch) + '.npy')
    x_data[...,1] = np.load(inputDir + '/clumichel_' + str(batch) + '.npy')
    x_data[...,2] = np.load(inputDir + '/cluem_' + str(batch) + '.npy')
  else:
    x_data = np.zeros((n_patches, patch_w, patch_h, 1))
    x_data[...,0] = np.load(inputDir + '/clumichel_' + str(batch) + '.npy')
  
  assert len(x_data) == len(y_data)
  
  p = np.random.permutation(len(x_data))
  
  return copy.deepcopy(x_data[p]), copy.deepcopy(y_data[p])

