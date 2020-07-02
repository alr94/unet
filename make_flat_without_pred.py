# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
from __future__ import print_function

################################################################################
# Parsing of input file and arguments
import argparse 
parser = argparse.ArgumentParser(description ='Run CNN training on patches with'
                                 + ' a few different hyperparameter sets.')
parser.add_argument('-t', '--datatype', help = 'Datatype: {MC, data}')
parser.add_argument('-d', '--data',     help = 'Data directory')
parser.add_argument('-o', '--output',   help = 'Output', default = 'test.root')
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
t_ev.Branch('primary_T0'      , b_p_t0  , 'b_p_t0/F')

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

# event params
b_e_cf  = array('f', [0.])
t_ev.Branch('calibFrac' , b_e_cf  , 'b_e_cf/F')

# Hit params
b_hit_wire        = ROOT.std.vector('double')()
b_hit_wirecalib   = ROOT.std.vector('double')()
b_hit_energy      = ROOT.std.vector('double')()
b_hit_energycalib = ROOT.std.vector('double')()
b_hit_cnnem       = ROOT.std.vector('double')()
b_hit_cnnmichel   = ROOT.std.vector('double')()
b_hit_cluem       = ROOT.std.vector('double')()
b_hit_clumichel   = ROOT.std.vector('double')()
b_hit_x           = ROOT.std.vector('double')()
b_hit_y           = ROOT.std.vector('double')()
b_hit_z           = ROOT.std.vector('double')()
b_hit_closeSP     = ROOT.std.vector('double')()

t_ev.Branch('hit_wire'        , b_hit_wire)
t_ev.Branch('hit_wirecalib'   , b_hit_wirecalib)
t_ev.Branch('hit_energy'      , b_hit_energy)
t_ev.Branch('hit_energycalib' , b_hit_energycalib)
t_ev.Branch('hit_cnnem'       , b_hit_cnnem)
t_ev.Branch('hit_cnnmichel'   , b_hit_cnnmichel)
t_ev.Branch('hit_cluem'       , b_hit_cluem)
t_ev.Branch('hit_clumichel'   , b_hit_clumichel)
t_ev.Branch('hit_x'           , b_hit_x)
t_ev.Branch('hit_y'           , b_hit_y)
t_ev.Branch('hit_z'           , b_hit_z)
t_ev.Branch('hit_closeSP'     , b_hit_closeSP)

# Michel params
if args.datatype == 'MC':
  
  b_m_te  = array('f', [0.])
  b_m_tie = array('f', [0.])
  t_ev.Branch('trueMichelEnergy' , b_m_te  , 'b_m_te/F')
  t_ev.Branch('totalTrueIonE'    , b_m_tie , 'b_m_tie/F')
  
  b_hit_truth      = ROOT.std.vector('double')()
  b_hit_truthcalib = ROOT.std.vector('double')()
  b_hit_trueenergy = ROOT.std.vector('double')()
  t_ev.Branch('hit_truth'      , b_hit_truth)
  t_ev.Branch('hit_truthcalib' , b_hit_truthcalib)
  t_ev.Branch('hit_trueenergy' , b_hit_trueenergy)
##############################################################################

# Loop over all files in path
print ('Starting file loop')
# for filename in os.listdir(args.data):
for _ in range(1): # FIXME
  filename = "MC_Train.root" 
  
  # Basic file checks and filtering
  if filename.split('.')[-1] != 'root': continue
  if args.datatype not in filename: continue
  
  # Get filenames and check they exist
  data_filename = args.data + '/' + filename
  
  if os.path.isfile(data_filename):
    input_data = ROOT.TFile(data_filename, 'READ')
  else: 
    print ("Filename not valid")
    continue
  print (filename)
  
  # Get top level directory
  data_base_dir = input_data.Get(dirname)
  
  # Find common set of keys based on title
  keys = [key for key in data_base_dir.GetListOfKeys()]
  
  # Basic check for key matching 
  n_keys = len(keys)
  if n_keys == 0:
    print ('Key matching failed. Exiting.')
    exit()
  
  # Calculate number of iterations to make and loop on keys
  n_steps = min(n_keys, samples) if samples > 0 else n_keys
  # for i_key in range(n_steps):
  for i_key in range(5000):
    
    if i_key % 1000 == 0: print (str(i_key) + ' / ' + str(n_steps))
    
    # get directory
    key = keys[i_key]
    data_dir = data_base_dir.Get(key.GetName())
    
    # Get event data
    wire        = root_numpy.hist2array(data_dir.Get('wire'))
    ncut = (wire.shape[0] - 160) / 2
    wire        = wire[ncut:-ncut, ncut:-ncut]
    
    wirecalib   = root_numpy.hist2array(data_dir.Get('wireCalib'))[ncut:-ncut, ncut:-ncut]
    energy      = root_numpy.hist2array(data_dir.Get('energy'))[ncut:-ncut, ncut:-ncut]
    energycalib = root_numpy.hist2array(data_dir.Get('energyCalib'))[ncut:-ncut, ncut:-ncut]
    cnnem       = root_numpy.hist2array(data_dir.Get('cnnem'))[ncut:-ncut, ncut:-ncut]
    cnnmichel   = root_numpy.hist2array(data_dir.Get('cnnmichel'))[ncut:-ncut, ncut:-ncut]
    cluem       = root_numpy.hist2array(data_dir.Get('cluem'))[ncut:-ncut, ncut:-ncut]
    clumichel   = root_numpy.hist2array(data_dir.Get('clumichel'))[ncut:-ncut, ncut:-ncut]
    x           = root_numpy.hist2array(data_dir.Get('hit_x'))[ncut:-ncut, ncut:-ncut]
    y           = root_numpy.hist2array(data_dir.Get('hit_y'))[ncut:-ncut, ncut:-ncut]
    z           = root_numpy.hist2array(data_dir.Get('hit_z'))[ncut:-ncut, ncut:-ncut]
    closestSP   = root_numpy.hist2array(data_dir.Get('hit_z'))[ncut:-ncut, ncut:-ncut]
    
    if args.datatype == 'MC':
      truth      = root_numpy.hist2array(data_dir.Get('truth'))[ncut:-ncut, ncut:-ncut]
      truthcalib = root_numpy.hist2array(data_dir.Get('truthCalib'))[ncut:-ncut, ncut:-ncut]
      trueenergy = root_numpy.hist2array(data_dir.Get('trueEnergy'))[ncut:-ncut, ncut:-ncut]
    
    # Get flattened data, filtered by wire value
    wire_flat        = wire.flatten()
    wirecalib_flat   = wirecalib.flatten()[wire_flat > 0.1]
    energy_flat      = energy.flatten()[wire_flat > 0.1]
    energycalib_flat = energycalib.flatten()[wire_flat > 0.1]
    cnnem_flat       = cnnem.flatten()[wire_flat > 0.1]
    cnnmichel_flat   = cnnmichel.flatten()[wire_flat > 0.1]
    cluem_flat       = cluem.flatten()[wire_flat > 0.1]
    clumichel_flat   = clumichel.flatten()[wire_flat > 0.1]
    x_flat           = x.flatten()[wire_flat > 0.1]
    y_flat           = y.flatten()[wire_flat > 0.1]
    z_flat           = z.flatten()[wire_flat > 0.1]
    closestSP_flat   = closestSP.flatten()[wire_flat > 0.1]
    if args.datatype == 'MC': 
      truth_flat = truth.flatten()[wire_flat > 0.1]
      truthcalib_flat = truthcalib.flatten()[wire_flat > 0.1]
      trueenergy_flat = trueenergy.flatten()[wire_flat > 0.1]
    wire_flat   = wire_flat[wire_flat > 0.1]
    
    # Clear vectors and fill with new event
    b_hit_wire.clear()
    b_hit_wirecalib.clear()
    b_hit_energy.clear()
    b_hit_energycalib.clear()
    b_hit_cnnem.clear()
    b_hit_cnnmichel.clear()
    b_hit_cluem.clear()
    b_hit_clumichel.clear()
    b_hit_x.clear()
    b_hit_y.clear()
    b_hit_z.clear()
    b_hit_closeSP.clear()
    
    if args.datatype == 'MC':
      b_hit_truth.clear()
      b_hit_truthcalib.clear()
      b_hit_trueenergy.clear()
    
    for i in range(len(wire_flat)):
      b_hit_wire.push_back(wire_flat[i])
      b_hit_wirecalib.push_back(wirecalib_flat[i])
      b_hit_energy.push_back(energy_flat[i])
      b_hit_energycalib.push_back(energycalib_flat[i])
      b_hit_cnnem.push_back(cnnem_flat[i])
      b_hit_cnnmichel.push_back(cnnmichel_flat[i])
      b_hit_cluem.push_back(cluem_flat[i])
      b_hit_clumichel.push_back(clumichel_flat[i])
      b_hit_x.push_back(x_flat[i])
      b_hit_y.push_back(y_flat[i])
      b_hit_z.push_back(z_flat[i])
      b_hit_closeSP.push_back(closestSP_flat[i])
      if args.datatype == 'MC': 
        b_hit_truth.push_back(truth_flat[i])
        b_hit_truthcalib.push_back(truthcalib_flat[i])
        b_hit_trueenergy.push_back(trueenergy_flat[i])
      
      # b_hit_int.push_back(wire_flat[i])
      # b_hit_energy.push_back(energy_flat[i])
      # if args.datatype == 'MC': b_hit_truth.push_back(float(truth_flat[i]))
  
    # Get primary and daughter parameters from tree
    tree  = data_dir.Get('param tree')
    for entry in tree:

      if args.datatype == 'MC': 
        b_m_te[0]  = entry.trueMichelEnergy
        b_m_tie[0]  = entry.totalTrueIonE
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
      b_e_cf[0] = entry.CalibFrac
      break
    
    # Fill event into tree
    t_ev.Fill()
    
    # Close dirs to free up memory
    data_dir.Close()
      
# Save output
output_file.Write()
output_file.Close()
