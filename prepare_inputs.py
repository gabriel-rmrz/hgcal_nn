DEBUG=False
DOPLOTS=True
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


def prepare_fromSim(filename, suf, bins= [6,6,6], isPU=False):
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
  fTsM_en = ak.ArrayBuilder()
  fTsM_en_cls= ak.ArrayBuilder()
  fTsM_g3D = ak.ArrayBuilder()
  fTsM_tiles = ak.ArrayBuilder()
  fTsM_pos = ak.ArrayBuilder()
  fTsM_eta_phi = ak.ArrayBuilder()
  fTsM_1D = ak.ArrayBuilder()
  fTsM_1D_cls = ak.ArrayBuilder()
  deltaTsM_1D = ak.ArrayBuilder()
  deltaTsM_1D_cls = ak.ArrayBuilder()
  deltaTsM_eta_phi_z_1D = ak.ArrayBuilder()
  deltaTsM_eta_phi_z_1D_cls = ak.ArrayBuilder()
  #target_perTracksterMerged
  fTsM_pos_cls = ak.ArrayBuilder()
  fTsM_eta_phi_cls = ak.ArrayBuilder()
  fTsM_g3D_cls = ak.ArrayBuilder()
  tTsM = ak.ArrayBuilder()
  tTsM_out = ak.ArrayBuilder()
  ev_info =  ak.ArrayBuilder()


  for i, ev in enumerate(simToReco_index): # looping over all the events
  #for i, ev in enumerate(simToReco_index[:601]): # looping over all the events
    if not (i%100) :
      print(f"%%%%%%%%%%%%%%% Event {i} %%%%%%%%%%%%%%%")
    isPassScore0 =simToReco_score[i] < 0.35
    if np.sum(isPassScore0, axis=None) < 1:
      continue
    en_ratio = simToReco_en[i][isPassScore0]/simTICLCandidate_raw_energy[i] + 0
    en_ratio = flatten_array(en_ratio)
    max_i = np.argmax(en_ratio)
    s = ev[isPassScore0]
    s = s[max_i]
    j = max_i
    mTs = tsLinkedInCand[i][s]
    mTs_en = simToReco_en[i][j]
    mTs = ak.flatten(tsLinkedInCand[i][s])
    #print(f"mTs: {mTs}")
    if len(mTs)==0:
      continue
    ev_info.append(np.array([i,max_i]))

    ###################################
    ## Defition of the inputs
    glob_pos = np.array([bxTs[i][mTs], byTs[i][mTs], bzTs[i][mTs]])
    glob_eta_phi = np.array([betaTs[i][mTs], bphiTs[i][mTs]])

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
    fTsM_eta_phi.append(glob_eta_phi)
    fTsM_en.append(raw_enTs[i][mTs])
    fTsM_en_cls.append(enCls[i][vtxIds])

    ###################################
    ## Defition of the target
    ##################################
    max_score_r2s = 0.35
    min_energy_r2s = 0.5
    #isPass = ((recoToSim_score[i][j] < max_score_r2s) & (recoToSim_en[i][j] > min_energy_r2s))
    isPassScore = recoToSim_score[i][j] < max_score_r2s 
    #en_frac = recoToSim_en[i][j][isPassScore]/simTICLCandidate_regressed_energy[i][recoToSim_index[i][j][isPassScore]]
    #en_frac = simToReco_en[i][j][isPassScore]/simtrackstersCP_raw_energy[i][j]
    en_frac = mTs_en/simtrackstersCP_raw_energy[i][j]

    isPass = False
    if np.any(en_frac > 0.5):
      isPass = True
    tTsM.append(isPass)

  ev_info = ev_info.snapshot()
  fTsM_en = save_array_to_file(fTsM_en, f"{suf}_en.parquet")
  fTsM_en_cls = save_array_to_file(fTsM_en_cls, f"{suf}_en_cls.parquet")
  fTsM_pos = save_array_to_file(fTsM_pos, f"{suf}_pos.parquet")
  fTsM_eta_phi = save_array_to_file(fTsM_eta_phi, f"{suf}_eta_phi.parquet")
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
    pos_ts = fTsM_pos[i,:,:]
    eta_phi_ts = fTsM_eta_phi[i,:,:]
    pos_cls = np.asarray(fTsM_pos_cls[i,:,:])
    pos_cls[2,:] = np.absolute(pos_cls[2,:])
    eta_phi_cls = fTsM_eta_phi_cls[i,:,:]
    en_ts = fTsM_en[i]
    en_cls = fTsM_en_cls[i]
    betaTs_mean = np.mean(eta_phi_ts[0,:])
    delta_etaTs = eta_phi_ts[0,:] - betaTs_mean
    bxTs_mean = np.mean(pos_ts[0,:])
    delta_xTs = pos_ts[0,:] - bxTs_mean
    byTs_mean = np.mean(pos_ts[1,:])
    delta_yTs = pos_ts[1,:] - byTs_mean
    bzTs_mean = np.mean(pos_ts[2,:])
    delta_zTs = pos_ts[2,:] - bzTs_mean
    delta_phiTs, bphiTs_mean = get_delta_phi(eta_phi_ts[1,:])
    #TODO: Check the 1.1 factor for the mins.

    rel_pos = np.array([delta_xTs, delta_yTs, delta_zTs]).T
    rel_eta_phi_z = np.array([delta_etaTs, delta_phiTs, delta_zTs]).T

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
    '''
    nbins_x = 6 # Use a pair number
    nbins_y = 6 # Use a pair number
    nbins_z = 10 # Use a pair number
    xy_bins_width = 5.
    '''
    nbins_x = 24 # Use a pair number
    nbins_y = 24 # Use a pair number
    nbins_z = 16 # Use a pair number
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


        

    #print(f"bxCls_mean_tile: {bxCls_mean_tile}")
    #print(f"bins_x: {bins_x}")
    #range_grid_cls= [[-25,25], [-25,25],[-25.,25.]]
    bins = (bins_x, bins_y, bins_z)
    '''
    # For eta-phi-z constant ranges
    range_grid= [[-0.4,0.4], [-0.4,0.4],[-25.,25.]]
    range_grid_cls= [[-0.4,0.4], [-0.4,0.4],[-25.,25.]]
    range_grid = [[np.min(delta_etaTs), np.max(delta_etaTs)], 
                  [np.min(delta_phiTs), np.max(delta_phiTs)],
                  [np.min(delta_zTs),  np.max(delta_zTs)]]
    # For eta-phi-z dinamic ranges
    range_grid_cls = [[ np.min(delta_etaCls), np.max(delta_etaCls)],
                  [np.min(delta_phiCls), np.max(delta_phiCls)],
                  [np.min(delta_zCls), np.max(delta_zCls)]]
    '''

    #fTsM_g3D.append(np.array([np.histogramdd(pos_cls, bins=bins)[0], np.histogramdd(pos_cls, weights=np.asarray(en_ts), bins=bins)[0]]))
    pos_cls = pos_cls.T
    histo_pos = np.histogramdd(pos_cls, bins=bins)[0]
    histo_en = np.histogramdd(pos_cls, weights=np.asarray(en_cls), bins=bins)[0]
    if np.sum(histo_pos, axis=None) ==0:
      print("******* empty ********")
      print(f"ev_info[i]: {ev_info[i]}")
      print(f"bins_x: {bins_x}")
      print(f"bins_y: {bins_y}")
      print(f"bins_z: {bins_z}")
      print(f"pos_cls: {pos_cls}")
      #print(f"histo_pos: {histo_pos}")
      #print(f"z_layers[bzCls_mean_tile_arg]: {z_layers[bzCls_mean_tile_arg]}")
      #print(f"pos_cls[:,2]: {pos_cls[:,2]}")
      print(f"bzCls_mean: {bzCls_mean}")
      continue
    fTsM_1D.append(np.array([bxTs_mean, byTs_mean, bzTs_mean]))
    fTsM_1D_cls.append(np.array([bxCls_mean, byCls_mean, bzCls_mean]))
    deltaTsM_1D_cls.append(rel_pos_cls)
    #deltaTsM_eta_phi_z_1D_cls.append(rel_eta_phi_z_cls)

    #deltaTsM_1D.append(rel_pos)
    tTsM_out.append(tTsM[i].item())
    deltaTsM_1D.append(rel_pos)
    deltaTsM_eta_phi_z_1D.append(rel_eta_phi_z)
    fTsM_g3D_cls.append(np.array([histo_pos,histo_en]))

  tTsM_out = save_array_to_file(tTsM_out, f"{suf}_tTsM.parquet")
  fTsM_1D = save_array_to_file(fTsM_1D, f"{suf}_fTsM_1D.parquet")
  fTsM_1D_cls = save_array_to_file(fTsM_1D_cls, f"{suf}_fTsM_1D_cls.parquet")
  deltaTsM_1D = save_array_to_file(deltaTsM_1D, f"{suf}_deltaTsM_1D.parquet")
  deltaTsM_1D_cls = save_array_to_file(deltaTsM_1D_cls, f"{suf}_deltaTsM_1D_cls.parquet")
  #fTsM_g3D = save_array_to_file(fTsM_g3D, f"{suf}_grid_3D.parquet")
  fTsM_g3D_cls = save_array_to_file(fTsM_g3D_cls, f"{suf}_grid_3D_cls.parquet")

  print(f"fTsM_g3D_cls.type: {fTsM_g3D_cls.type}")
  print(f"tTsM_out.type: {tTsM_out.type}")
  if DOPLOTS:
    vox_in = (np.array(fTsM_g3D_cls)[tTsM_out,0,:,:,:] > 0).astype(np.int32)
    plot_voxels(vox_in)

  return deltaTsM_1D, deltaTsM_1D_cls, fTsM_1D, fTsM_1D, fTsM_g3D, fTsM_g3D_cls, tTsM


def plot_vars(deltaTsM_1D_cls, fTsM_1D, fTsM_g3D_cls, tTsM, suf):
    myhist(ak.flatten(deltaTsM_1D_cls[:,:,0], axis=None), title="Delta_x_cls", xlabel="x-x_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_x.png")
    plt.clf()
    myhist(ak.flatten(deltaTsM_1D_cls[:,:,1], axis=None), title="Delta_y_cls", xlabel="y-y_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_y.png")
    plt.clf()
    myhist(ak.flatten(deltaTsM_1D_cls[:,:,2], axis=None), title="Delta_z_cls", xlabel="z-z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_z.png")
    plt.clf()

    myhist(fTsM_1D[:,0], title="mean_x", xlabel="x_mean for clusters", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_x.png")
    plt.clf()
    myhist(fTsM_1D[:,1], title="mean_y", xlabel="y_mean for clusters", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_y.png")
    plt.clf()
    myhist(fTsM_1D[:,2], title="mean_z", xlabel="z_mean for clusters", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_z.png")
    plt.clf()
    myhist(np.array(tTsM, dtype=np.int32), title="Truth: en>en_min and score < score_max", xlabel="Passed", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_truth.png")
    plt.clf()

def plot_voxels(vox_in):
    print("####################")
    print("if this does not work is probably because we don't have enough TICLCandidate passing the energy cut")
    print("####################")
    fig = plt.figure(figsize=(30,30), dpi=200)
    for sp in range(1,50,1):
      ax = fig.add_subplot(7,7,sp, projection='3d')
      #ax = fig.gca( projection='3d')
      vox = vox_in[sp]
      ax.voxels(vox, shade=True, alpha=0.45)
    plt.savefig("voxel_test.png")
    plt.clf()

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
  bins = [12,12,12]
  for suf in file_sufix:
    #deltaTsM_1D, fTsM_1D, fTsM_g3D, fTsM_pos, tTsM = prepare(f"data/histo_{suf}.root", suf, bins=bins, isPU=False)
    deltaTsM_1D, deltaTsM_1D_cls, fTsM_1D, fTsM_1D_cls, fTsM_g3D, fTsM_g3D_cls, tTsM = prepare_fromSim(f"data/histo_{suf}.root", suf, bins=bins, isPU=False)

    if DOPLOTS:
      plot_vars(deltaTsM_1D_cls, fTsM_1D, fTsM_g3D_cls, tTsM, suf)

      
      

  '''
  # This solution takes up too much disk space.
  with open('input.pkl', 'wb') as f:
    pickle.dump(fTsM_g3D, f)
  '''



  


if __name__=='__main__':
  main()
