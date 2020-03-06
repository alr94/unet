# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import numpy as np
import ROOT

def RecoEnergy(pred, energy, thresh):
  threshed   = (pred > thresh).astype(float)
  mask       = (energy < 10).astype(float)
  e_threshed = energy * mask
  e_selected = np.sum(np.abs(threshed * e_threshed))
  return e_selected

def RecoEnergyHitCut(pred, energy, thresh):
  threshed   = (pred > thresh).astype(float)
  mask       = (energy > 0.75).astype(float)
  e_threshed = energy * mask
  e_selected = np.sum(np.abs(threshed * e_threshed))
  return e_selected
  
def NHits(pred, thresh):
  threshed   = (pred > thresh).astype(float)
  n_selected = np.sum(np.abs(threshed))
  return n_selected

def Locations(pred, thresh):
  locs = np.argwhere(pred > thresh)
  return locs

def HitDistances(pred, thresh):
  distances = ROOT.vector('float')()
  shape     = pred.shape
  
  locs = Locations(pred, thresh)
  locs = locs - [shape[0]/2, shape[1]/2, 0]
  return (np.linalg.norm(locs, axis=1))
