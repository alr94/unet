# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-t', '--datatype', help = 'Datatype: {MC, data}')
parser.add_argument('-d', '--data',     help = 'Data directory')
parser.add_argument('-p', '--pred',     help = 'Pred directory')
parser.add_argument('-o', '--output',   help = 'Output directory',
                    default = 'test.root')
args = parser.parse_args()

################################################################################
# Other setups
import os
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from array import array
  
import ROOT
import root_numpy

################################################################################
# My stuff
from performance_metrics import *

################################################################################
if not (args.data):
  print ('Please provide data')
  exit()
  
# Name of top level TDirectoryFile
dirname          = 'MichelEnergyImage' if args.datatype == 'MC' else 'MichelEnergyImageData'
# Size of image
patch_w, patch_h = 160, 160
# Crop level from data images to prediction size of CNN
ncrop            = int((164 - patch_w)/ 2)
# Number of iterations per file, set to 0 to do all events
samples          = 0

##############################################################################
# Open output file and make output tree
output_file = ROOT.TFile(args.output, 'RECREATE')
t_ev = ROOT.TTree('performance_tests', 'performance scores')

# Primary params
b_p_id  = array('f', [0.])
b_p_vx  = array('f', [0.])
b_p_vy  = array('f', [0.])
b_p_vz  = array('f', [0.])
b_p_ex  = array('f', [0.])
b_p_ey  = array('f', [0.])
b_p_ez  = array('f', [0.])
b_p_ht0 = array('f', [0.])
b_p_t0  = array('f', [0.])
t_ev.Branch('primary_ID'      , b_p_id  , 'b_p_id/F')
t_ev.Branch('primary_VertexX' , b_p_vx  , 'b_p_vx/F')
t_ev.Branch('primary_VertexY' , b_p_vy  , 'b_p_vy/F')
t_ev.Branch('primary_VertexZ' , b_p_vz  , 'b_p_vz/F')
t_ev.Branch('primary_EndX'    , b_p_ex  , 'b_p_ex/F')
t_ev.Branch('primary_EndY'    , b_p_ey  , 'b_p_ey/F')
t_ev.Branch('primary_EndZ'    , b_p_ez  , 'b_p_ez/F')
t_ev.Branch('primary_HasT0'   , b_p_ht0 , 'b_p_ht0/F')
t_ev.Branch('primary_HasT0'   , b_p_t0  , 'b_p_t0/F')

# Daughter params
b_d_nh  = array('f', [0.])
b_d_nmh = array('f', [0.])
b_d_fmh = array('f', [0.])
b_d_dp  = array('f', [0.])
b_d_dpx = array('f', [0.])
b_d_dpy = array('f', [0.])
b_d_dpz = array('f', [0.])
t_ev.Branch('daughter_NHits'        , b_d_nh  , 'b_d_nh/F')
t_ev.Branch('daughter_NMichHits'    , b_d_nmh , 'b_d_nmh/F')
t_ev.Branch('daughter_FracMichHits' , b_d_fmh , 'b_d_fmh/F')
t_ev.Branch('daughter_DistPrim'     , b_d_dp  , 'b_d_dp/F')
t_ev.Branch('daughter_DistPrimX'    , b_d_dpx , 'b_d_dpx/F')
t_ev.Branch('daughter_DistPrimY'    , b_d_dpy , 'b_d_dpy/F')
t_ev.Branch('daughter_DistPrimZ'    , b_d_dpz , 'b_d_dpz/F')

# Hit params
b_hit_cnn    = ROOT.std.vector('double')()
b_hit_int    = ROOT.std.vector('double')()
b_hit_energy = ROOT.std.vector('double')()

t_ev.Branch('hit_cnn'    , b_hit_cnn)
t_ev.Branch('hit_int'    , b_hit_int)
t_ev.Branch('hit_energy' , b_hit_energy)

##############################################################################

# Loop over all files in path
for filename in os.listdir(args.data):
  
  # Basic file checks and filtering
  if filename.split('.')[-1] != 'root': continue
  if args.datatype not in filename: continue
  
  # Get filenames and heck they exist
  data_filename = args.data + '/' + filename
  pred_filename = args.pred + '/' + filename
  print(data_filename, pred_filename)
  if os.path.isfile(data_filename) and os.path.isfile(pred_filename):
    input_data = ROOT.TFile(data_filename, 'READ')
    input_pred = ROOT.TFile(pred_filename, 'READ')
  else: 
    print ("Filename not valid")
    continue
  print (filename)
  
  # Get top level directory
  data_dir = input_data.Get(dirname)
  pred_dir = input_pred.Get(dirname)
  
  # Find common set of keys based on title
  data_keys = [key.GetTitle() for key in data_dir.GetListOfKeys()]
  pred_keys = [key.GetTitle() for key in pred_dir.GetListOfKeys()]
  key_set  = list(set(data_keys) & set(pred_keys))
  
  # Get keys from common set
  data_keys = [key for key in data_dir.GetListOfKeys() 
               if key.GetTitle() in key_set]
  pred_keys = [key for key in pred_dir.GetListOfKeys() 
               if key.GetTitle() in key_set]
  
  # Basic check for key matching 
  n_keys = len(data_keys)
  if len(data_keys) != len(pred_keys) or n_keys == 0:
    print ('Key matching failed. Exiting.')
    exit()
  
  # Calculate number of iterations to make and loop on keys
  n_steps = min(n_keys, samples) if samples > 0 else n_keys
  for i_key in range(n_steps):
    if i_key % 1000 == 0: print (str(i_key) + ' / ' + str(n_steps))
    
    # Check keys match
    data_key = data_keys[i_key]
    pred_key = pred_keys[i_key]
    if data_key.GetTitle() != pred_key.GetTitle(): 
      print ("Keys don't match: ", key.GetTitle())
      continue
    
    # Read directories
    data_dir = data_key.ReadObj()
    pred_dir = pred_key.ReadObj()
    
    # Get event data
    wire   = root_numpy.hist2array(data_dir.Get('wire'))[ncrop:-ncrop, 
                                                         ncrop:-ncrop]
    energy = root_numpy.hist2array(data_dir.Get('energy'))[ncrop:-ncrop, 
                                                           ncrop:-ncrop]
    pred   = root_numpy.hist2array(pred_dir.Get('prediction'))

    # Get flattened data, filtered by wire value
    wire_flat   = wire.flatten()
    energy_flat = energy.flatten()[wire_flat > 0.1]
    pred_flat   = pred.flatten()[wire_flat > 0.1]
    wire_flat   = wire_flat[wire_flat > 0.1]
    
    # Clear vectors and fill with new event
    b_hit_cnn.clear()
    b_hit_int.clear()
    b_hit_energy.clear()
    for i in range(len(wire_flat)):
      b_hit_cnn.push_back(pred_flat[i])
      b_hit_int.push_back(wire_flat[i])
      b_hit_energy.push_back(energy_flat[i])
  
    # Get primary and daughter parameters from tree
    tree  = data_dir.Get('param tree')
    for entry in tree:
      b_p_id[0]  = entry.primaryID
      b_p_vx[0]  = entry.VertexX
      b_p_vy[0]  = entry.VertexY
      b_p_vz[0]  = entry.VertexZ
      b_p_ex[0]  = entry.EndX
      b_p_ey[0]  = entry.EndY
      b_p_ez[0]  = entry.EndZ
      b_p_ht0[0] = entry.HasT0
      b_p_t0[0]  = entry.T0
      b_d_nh[0]  = entry.NHits
      b_d_nmh[0] = entry.NMichelHits
      b_d_fmh[0] = entry.FractionMichelHits
      b_d_dp[0]  = entry.DistanceToPrimary
      b_d_dpx[0] = entry.DistanceToPrimaryX
      b_d_dpy[0] = entry.DistanceToPrimaryY
      b_d_dpz[0] = entry.DistanceToPrimaryZ
      break
    
    # Fill event into tree
    t_ev.Fill()
    
    # Close dirs to free up memory
    data_dir.Close()
    pred_dir.Close()
      
# Save output
output_file.Write()
output_file.Close()
