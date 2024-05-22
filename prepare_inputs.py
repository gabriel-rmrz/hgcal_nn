DEBUG=False
import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def flatten_array(arr, empty_value=-1):
  # Initialize the result list
  result = []
  for item in arr:
    # Check if the item is 'None' or empty
    if len(item) ==0:
      # Append the default value for empty entries
      result.append(empty_value)
    else:
      # Append the actual item, assuming it's a single element list
      result.append(item[0])
  return result
def myhist(X, bins=30, title='title', xlabel='time (ns)', ylabel='Counts / bin', color='dodgerblue', alpha=1, fill='stepfilled', range=None, label="data"):
  #plt.figure(dpi=100)
  if range==None:
    plt.hist(np.array(X), bins=bins, color=color, alpha=alpha, histtype=fill, label=label)
  else:
    plt.hist(np.array(X), bins=bins, color=color, alpha=alpha, histtype=fill, range=range, label=label)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid()

def get_delta_phi(bphiTs_i):
  # Computing the difference on phi making sure to keep the convention of -pi - pi
  # Probably this can be done more easily using 2D-vectors.
  bphiTs_mean = np.mean(bphiTs_i)
  if np.any(bphiTs_i < -np.pi/2) and np.any(bphiTs_i > np.pi/2):
    bphiTs_mean_temp = np.mean(np.where(bphiTs_i < 0, bphiTs_i +2*np.pi, bphiTs_i))
    if bphiTs_mean_temp < -np.pi:
      bphiTs_mean = bphiTs_mean_temp + 2*np.pi
    elif bphiTs_mean_temp > np.pi:
      bphiTs_mean = bphiTs_mean_temp - 2*np.pi

  delta_phiTs = np.where(np.sign(bphiTs_i) == np.sign(bphiTs_mean), 
                         bphiTs_i- bphiTs_mean, 
                         bphiTs_i+ np.sign(bphiTs_mean)*2*np.pi - bphiTs_mean)
  delta_phiTs= np.where(delta_phiTs < -np.pi, delta_phiTs + 2*np.pi, delta_phiTs)
  delta_phiTs= np.where(delta_phiTs > np.pi, delta_phiTs - 2*np.pi, delta_phiTs)
  
  return delta_phiTs, bphiTs_mean


def Gauss(x, A, B):
  return A*np.exp(-1*B*x**2)

def get_limits_gauss(X):
  X = np.array(ak.flatten(X, axis=None))
  h1, bins1 = np.histogram(X, bins=100)
  x1 = (bins1[1:] + bins1[:-1])/2
  p1, cov1 = curve_fit(Gauss, x1, h1)
  #fit_y = Gauss(x1, p1[0], p1[1])
  return p1

def save_array_to_file(arr_builder, filename):
  arr = arr_builder.snapshot()
  form, lenght, container = ak.to_buffers(arr)
  ak.to_parquet(arr, f"data/{filename}")
  return arr


def prepare_fromSim(filename, suf, nbins= [6,6,6], isPU=False):
  print(f"Reading {filename}...")
  file = uproot.open(filename)

  simtrackstersSC = file["ticlDumper/simtrackstersSC"]
  simtrackstersCP = file["ticlDumper/simtrackstersCP"]
  #tracksters = file["ticlDumper/tracksters"]
  tracksters = file["ticlDumper/linkedTracksters"]
  trackstersMerged = file["ticlDumper/trackstersMerged"]
  associations = file["ticlDumper/associations"]
  clusters = file["ticlDumper/clusters"]
  TICLCandidate = file["ticlDumper/candidates"]
  simTICLCandidate = file["ticlDumper/simTICLCandidate"]

  simtrackstersCP_raw_energy = simtrackstersCP["raw_energy"].array()

  tsLinkedInCand = TICLCandidate["trackstersLinked_in_candidate"].array()

  tErrTs = tracksters["timeError"].array()
  bxTs = tracksters["barycenter_x"].array()
  byTs = tracksters["barycenter_y"].array()
  bzTs = tracksters["barycenter_z"].array()
  betaTs = tracksters["barycenter_eta"].array()
  bphiTs = tracksters["barycenter_phi"].array()
  reg_enTs = tracksters["regressed_energy"].array()
  raw_enTs = tracksters["raw_energy"].array()

  vtxIdTs = tracksters["vertices_indexes"].array()
  betaCls = clusters["position_eta"].array()
  bphiCls = clusters["position_phi"].array()
  bxCls = clusters["position_x"].array()
  byCls = clusters["position_y"].array()
  bzCls = clusters["position_z"].array()

  bxTsM = trackstersMerged["barycenter_x"].array()
  byTsM = trackstersMerged["barycenter_y"].array()
  bzTsM = trackstersMerged["barycenter_z"].array()
  betaTsM = trackstersMerged["barycenter_eta"].array()
  bphiTsM = trackstersMerged["barycenter_phi"].array()
  #corrected_enCls = clusters["correctedEnergy"].array()
  enCls = clusters["energy"].array()

  recoToSim_en = associations["Mergetracksters_recoToSim_CP_sharedE"].array()
  recoToSim_score = associations["Mergetracksters_recoToSim_CP_score"].array()
  recoToSim_index = associations["Mergetracksters_recoToSim_CP"].array()

  simToReco_en = associations["Mergetracksters_simToReco_CP_sharedE"].array()
  simToReco_score = associations["Mergetracksters_simToReco_CP_score"].array()
  simToReco_index = associations["Mergetracksters_simToReco_CP"].array()

  simTICLCandidate_raw_energy = simTICLCandidate["simTICLCandidate_raw_energy"].array()

  # Creating the voxels for the different trackstersMerged
  #features_perTracksterMerged
  fTsM_en_cls= ak.ArrayBuilder()
  fTsM_tiles = ak.ArrayBuilder()
  fTsM_pos = ak.ArrayBuilder()
  fTsM_scalars = ak.ArrayBuilder()
  fTsM_eta_phi = ak.ArrayBuilder()
  fTsM_s2r_score = ak.ArrayBuilder()
  fTsM_1D_cls = ak.ArrayBuilder()
  deltaTsM_1D_cls = ak.ArrayBuilder()
  deltaTsM_eta_phi_z_1D_cls = ak.ArrayBuilder()
  #target_perTracksterMerged
  fTsM_pos_cls = ak.ArrayBuilder()
  fTsM_eta_phi_cls = ak.ArrayBuilder()
  fTsM_g3D_cls = ak.ArrayBuilder()
  tTsM = ak.ArrayBuilder()
  tTsM_out = ak.ArrayBuilder()
  ev_info =  ak.ArrayBuilder()


  for i, s2r_ind in enumerate(simToReco_index): # looping over all the events
  #for i, s2r_ind in enumerate(simToReco_index[:101]): # looping over all the events
    if not (i%100) :
      print(f"%%%%%%%%%%%%%%% Event {i} %%%%%%%%%%%%%%%")
    
    for j in range(len(s2r_ind)):
      isPassScore0 = simToReco_score[i][j] < 0.5
      if np.sum(isPassScore0, axis=None) < 1:
        continue
      ind = np.arange(len(isPassScore0),dtype=int)
      ind = ind[isPassScore0]
      s2r_ind_j = s2r_ind[j][isPassScore0]  # tracksterMerged that passed the score cut..
      for k, s in zip(ind,s2r_ind_j):


        ## TODO: Use all the s' , then flatten, and compute unique.
        mTs = tsLinkedInCand[i][s]
        mTs_en = simToReco_en[i][j][k]
        mTs = tsLinkedInCand[i][s]
        if len(mTs)==0:
          continue
        ev_info.append(np.array([i,j]))

        ###################################
        ## Defition of the inputs
        #glob_pos = np.array([bxTs[i][mTs], byTs[i][mTs], bzTs[i][mTs]])
        glob_pos = np.array([bxTsM[i][s], byTsM[i][s], bzTsM[i][s]])
        #glob_eta_phi = np.array([betaTs[i][mTs], bphiTs[i][mTs]])
        glob_eta_phi = np.array([betaTsM[i][s], bphiTsM[i][s]])

        vtxIds = ak.flatten(vtxIdTs[i][mTs])
        betaClsInTs = betaCls[i][vtxIds]
        bphiClsInTs = bphiCls[i][vtxIds]
        bxClsInTs = bxCls[i][vtxIds]
        byClsInTs = byCls[i][vtxIds]
        bzClsInTs = bzCls[i][vtxIds]
        glob_pos_cls = np.array([bxClsInTs, byClsInTs, bzClsInTs])
        glob_eta_phi_cls = np.array([betaClsInTs, bphiClsInTs])

        fTsM_pos_cls.append(glob_pos_cls)
        fTsM_eta_phi_cls.append(glob_eta_phi_cls)
        fTsM_pos.append(glob_pos)
        fTsM_scalars.append(np.array([betaClsInTs, bxClsInTs]))
        fTsM_eta_phi.append(glob_eta_phi)
        fTsM_s2r_score.append(np.array(simToReco_score[i][j][k]))
        fTsM_en_cls.append(enCls[i][vtxIds])

        ###################################
        ## Defition of the target
        ##################################
        max_score_s2r = 0.35
        min_energy_s2r = 0.0
        isPassScore = simToReco_score[i][j][k] < max_score_s2r 
        en_frac = mTs_en/simtrackstersCP_raw_energy[i][j]

        isPassEn = False
        if np.any(en_frac > min_energy_s2r):
          isPassEn = True
        tTsM.append(isPassScore)

  ev_info = ev_info.snapshot()
  fTsM_en_cls = save_array_to_file(fTsM_en_cls, f"{suf}_en_cls.parquet")
  fTsM_pos = save_array_to_file(fTsM_pos, f"{suf}_pos.parquet")
  fTsM_eta_phi = save_array_to_file(fTsM_eta_phi, f"{suf}_eta_phi.parquet")
  fTsM_s2r_score = save_array_to_file(fTsM_s2r_score, f"{suf}_s2r_score.parquet")
  fTsM_pos_cls = save_array_to_file(fTsM_pos_cls, f"{suf}_pos_cls.parquet")
  fTsM_eta_phi_cls = save_array_to_file(fTsM_eta_phi_cls, f"{suf}_eta_phi_cls.parquet")
  #tTsM = save_array_to_file(tTsM, f"{suf}_tTsM.parquet")
  tTsM = tTsM.snapshot()

  # TODO: Add pos_tsM to the loop

  z_layers = np.array([322, 323, 325, 326, 328, 329, 331, 332, 334, 335, 337, 338, 340,341, 343, 344, 346, 347, 349, 350, 353, 354, 356, 357, 360, 361,
                  367, 374, 380, 386, 393, 399, 405, 411, 412, 418, 424, 430, 431, 439, 447, 455, 463, 471, 472, 480, 488, 496, 504, 505, 513], dtype=np.int32)
  for i in range(len(fTsM_pos)):
    if not (i%100) :
      print(f"%%%%%%%%%%%%%%% Candidate {i} %%%%%%%%%%%%%%%")
    pos_cls = np.asarray(fTsM_pos_cls[i])
    pos_cls[2,:] = np.absolute(pos_cls[2,:])
    eta_phi_cls = fTsM_eta_phi_cls[i]
    en_cls = fTsM_en_cls[i]


    betaCls_mean = np.mean(eta_phi_cls[0,:])
    delta_etaCls = eta_phi_cls[0,:] - betaCls_mean
    bxCls_mean = np.mean(pos_cls[0,:])
    delta_xCls = pos_cls[0,:] - bxCls_mean
    byCls_mean = np.mean(pos_cls[1,:])
    delta_yCls = pos_cls[1,:] - byCls_mean
    bzCls_mean = np.mean(pos_cls[2,:])
    #delta_zCls = pos_cls[2,:] - z_layers[bzCls_mean_layer] #bzCls_mean
    delta_zCls = pos_cls[2,:] - bzCls_mean

    delta_phiCls, bphiCls_mean = get_delta_phi(eta_phi_cls[1,:])

    rel_pos_cls = np.array([delta_xCls, delta_yCls, delta_zCls]).T
    rel_eta_phi_z_cls = np.array([delta_etaCls, delta_phiCls, delta_zCls]).T

    #range_grid= [[-25,25], [-25,25],[-5,5]]
    nbins_x = nbins[0] # Use a pair number
    nbins_y = nbins[1] # Use a pair number
    nbins_z = nbins[2] # Use a pair number
    xy_bins_width = 1.
    bxCls_mean_tile = xy_bins_width *(bxCls_mean//xy_bins_width)
    bins_x = np.arange(bxCls_mean_tile-xy_bins_width*nbins_x/2, bxCls_mean_tile + xy_bins_width*(nbins_x/2+1), xy_bins_width)
    byCls_mean_tile = xy_bins_width *(byCls_mean//xy_bins_width)
    bins_y = np.arange(byCls_mean_tile -xy_bins_width*nbins_y/2, byCls_mean_tile + xy_bins_width*(nbins_y/2+1), xy_bins_width)
    bzCls_mean_tile_arg = np.absolute(bzCls_mean-z_layers).argmin()
    bins_z = np.zeros(nbins_z+1, dtype=np.double)
    if (bzCls_mean_tile_arg < int(nbins_z/2)):
      for m, j in enumerate(np.arange(bzCls_mean_tile_arg-int(nbins_z/2),0,1)):
        bins_z[m] = z_layers[0] +j -.5
      for k, l in enumerate(np.arange(0,bzCls_mean_tile_arg+int(nbins_z/2)+1,1)):
        bins_z[l- bzCls_mean_tile_arg+int(nbins_z/2)] = z_layers[k] -.5
    elif bzCls_mean_tile_arg+(nbins_z/2) >= (len(z_layers) ):
      for m, j in enumerate(np.arange(bzCls_mean_tile_arg-int(nbins_z/2),len(z_layers),1)):
        bins_z[m] = z_layers[j] -.5
      for k, l in enumerate(np.arange(len(z_layers)-bzCls_mean_tile_arg+int(nbins_z/2),nbins_z+1,1)):
        bins_z[l] = z_layers[-1] +k+1  -.5
    else:
      bins_z = z_layers[np.arange(bzCls_mean_tile_arg-int(nbins_z/2), bzCls_mean_tile_arg+int(nbins_z/2)+1, 1)] -.5
    bins = (bins_x, bins_y, bins_z)

    pos_cls = pos_cls.T
    histo_pos = np.histogramdd(pos_cls, bins=bins)[0]
    histo_en = np.histogramdd(pos_cls, weights=np.asarray(en_cls), bins=bins)[0]
    fTsM_1D_cls.append(np.array([bxCls_mean, byCls_mean, bzCls_mean]))
    deltaTsM_1D_cls.append(rel_pos_cls)
    #deltaTsM_eta_phi_z_1D_cls.append(rel_eta_phi_z_cls)

    tTsM_out.append(tTsM[i])
    fTsM_g3D_cls.append(np.array([histo_pos,histo_en]))

  tTsM_out = save_array_to_file(tTsM_out, f"{suf}_tTsM.parquet")
  fTsM_1D_cls = save_array_to_file(fTsM_1D_cls, f"{suf}_fTsM_1D_cls.parquet")
  deltaTsM_1D_cls = save_array_to_file(deltaTsM_1D_cls, f"{suf}_deltaTsM_1D_cls.parquet")
  fTsM_g3D_cls = save_array_to_file(fTsM_g3D_cls, f"{suf}_grid_3D_cls.parquet")
  return deltaTsM_1D_cls, fTsM_1D_cls, fTsM_g3D_cls, fTsM_s2r_score, tTsM


def main():
  """
  Reads a ROOT file containing RECO and sim information about TICLCandidates, 
  trackstersMerged, linkedTracksters, Clusters, etc...
  Then define voxels for every trackstersMerged and its linkedTracksters.
  Defines the target array based on the association of the simulated and reconstructed objects.
  Parameters:
  - ROOT file name.
  Returns:
  - Parquet file containing voxels of positions and energy (and maybe others), the eta, 
  phi and r array (len(eta) == num of trackstersMerged).
  - Target array.
  """
  # Reading file and input variables
  #filename = 'histo_SinglePi0PU_pT20to200_eta17to27.root'
  #filename = 'histo_SinglePi_withLinks.root'
  #filename = 'histo_4Pions_0PU_pt10to100_eta17to27.root'
  file_sufix = [
   '4Pions_0PU']
   #'4Photons_0PU']
   #'SinglePi']
   #'kaon_PU75']
   #'4Photons_0PU',                    
   #'4Pions_0PU',
   
   #'4Photon_PU200'
   #]
  nbins = [24,24,36]
  for suf in file_sufix:
    deltaTsM_1D_cls, fTsM_1D_cls, fTsM_g3D_cls, fTsM_s2r_score, tTsM = prepare_fromSim(f"data/histo_{suf}.root", suf, nbins=nbins, isPU=False)

if __name__=='__main__':
  main()
