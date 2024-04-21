DEBUG=False
DOPLOTS=False
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


def prepare_fromSim(filename, suf, bins= [6,6,6], isPU=False):
  print(f"Reading {filename}...")
  file = uproot.open(filename)

  simtrackstersSC = file["ticlDumper/simtrackstersSC"]
  simtrackstersCP = file["ticlDumper/simtrackstersCP"]
  tracksters = file["ticlDumper/tracksters"]
  trackstersMerged = file["ticlDumper/trackstersMerged"]
  associations = file["ticlDumper/associations"]
  clusters = file["ticlDumper/clusters"]
  TICLCandidate = file["ticlDumper/candidates"]
  simTICLCandidate = file["ticlDumper/simTICLCandidate"]

  simtrackstersCP_raw_energy = simtrackstersCP["raw_energy"].array()

  tsLinkedInCand = TICLCandidate["trackstersLinked_in_candidate"].array()

  tErrTs = tracksters["timeError"].array()
  bxTs = tracksters["barycenter_x"].array()
  bzTs = tracksters["barycenter_z"].array()
  betaTs = tracksters["barycenter_eta"].array()
  bphiTs = tracksters["barycenter_phi"].array()
  reg_enTs = tracksters["regressed_energy"].array()
  raw_enTs = tracksters["raw_energy"].array()

  vtxIdTs = tracksters["vertices_indexes"].array()
  betaCls = clusters["position_eta"].array()
  bphiCls = clusters["position_phi"].array()
  bzCls = clusters["position_z"].array()
  corrected_enCls = clusters["correctedEnergy"].array()

  betaCls = clusters["position_eta"].array()
  bphiCls = clusters["position_phi"].array()
  bzCls = clusters["position_z"].array()

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
  fTsM_pos = ak.ArrayBuilder()
  fTsM_1D = ak.ArrayBuilder()
  deltaTsM_1D = ak.ArrayBuilder()
  deltaTsM_1D_cls = ak.ArrayBuilder()
  #target_perTracksterMerged
  fTsM_pos_cls = ak.ArrayBuilder()
  fTsM_g3D_cls = ak.ArrayBuilder()
  tTsM = ak.ArrayBuilder()

  for i, ev in enumerate(simToReco_index): # looping over all the events
  #for i, ev in enumerate(simToReco_index[:101]): # looping over all the events
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

    ###################################
    ## Defition of the inputs
    glob_pos = np.array([betaTs[i][mTs], bphiTs[i][mTs], bzTs[i][mTs]])
    '''
    print(f"glob_pos: {glob_pos}")
    print(f"glob_pos.shape: {glob_pos.shape}")
    print(f"glob_pos.T: {glob_pos.T}")
    '''

    vtxIds = ak.flatten(vtxIdTs[i][mTs])
    betaClsInTs = betaCls[i][vtxIds]
    bphiClsInTs = bphiCls[i][vtxIds]
    bzClsInTs = bzCls[i][vtxIds]
    '''
    print(f"mTs: {mTs}")
    print(f"vtxIds: {vtxIds}")
    print(f"betaClsInTs: {betaClsInTs}")
    print(f"bphiClsInTs: {bphiClsInTs}")
    print(f"bzClsInTs: {bzClsInTs}")
    '''
    glob_pos_cls = np.array([betaClsInTs, bphiClsInTs, bzClsInTs])

    '''
    print(f"glob_pos_cls: {glob_pos_cls}")
    print(f"glob_pos_cls.shape: {glob_pos_cls.shape}")
    print(f"glob_pos_cls.T: {glob_pos_cls.T}")
    '''

    fTsM_pos_cls.append(glob_pos_cls)
    fTsM_pos.append(glob_pos)
    fTsM_en.append(raw_enTs[i][mTs])
    fTsM_en_cls.append(corrected_enCls[i][vtxIds])

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

  fTsM_en = fTsM_en.snapshot()
  fTsM_en_cls = fTsM_en_cls.snapshot()
  fTsM_pos = fTsM_pos.snapshot()
  fTsM_pos_cls = fTsM_pos_cls.snapshot()
  tTsM = tTsM.snapshot()
  form_2, lenght_2, container_2 = ak.to_buffers(tTsM)
  ak.to_parquet(tTsM, f"data/{suf}_tTsM.parquet")
  form_4, lenght_4, container_4 = ak.to_buffers(fTsM_pos_cls)
  ak.to_parquet(fTsM_pos_cls, f"data/{suf}_pos_cls.parquet")
  form_5, lenght_5, container_5 = ak.to_buffers(fTsM_pos)
  ak.to_parquet(fTsM_pos, f"data/{suf}_pos.parquet")

  # TODO: Add pos_tsM to the loop

  print(fTsM_pos_cls[0,0,:].type)
  for i in range(len(fTsM_pos)):
    if not (i%100) :
      print(f"%%%%%%%%%%%%%%% Candidate {i} %%%%%%%%%%%%%%%")
    pos_ts = fTsM_pos[i,:,:]
    pos_cls = fTsM_pos_cls[i,:,:]
    en_ts = fTsM_en[i]
    en_cls = fTsM_en_cls[i]
    betaTs_mean = np.mean(pos_ts[0,:])
    delta_etaTs = pos_ts[0,:] - betaTs_mean
    bzTs_mean = np.mean(pos_ts[2,:])
    delta_zTs = pos_ts[2,:] - bzTs_mean
    delta_phiTs, bphiTs_mean = get_delta_phi(pos_ts[1,:])
    #print(f"pos_ts[0,:] : {pos_ts[0,:]}")
    #print(f"pos_cls[0,:] : {pos_cls[0,:]}")
    #TODO: Check the 1.1 factor for the mins.

    rel_pos = np.array([delta_etaTs, delta_phiTs, delta_zTs]).T

    fT = np.array([betaTs_mean, bphiTs_mean, bzTs_mean])

    fTsM_1D.append(np.array([betaTs_mean, bphiTs_mean, bzTs_mean]))
    #deltaTsM_1D.append(rel_pos)
    deltaTsM_1D.append(rel_pos)

    betaCls_mean = np.mean(pos_cls[0,:])
    delta_etaCls = pos_cls[0,:] - betaCls_mean
    bzCls_mean = np.mean(pos_cls[2,:])
    delta_zCls = pos_cls[2,:] - bzCls_mean

    delta_phiCls, bphiCls_mean = get_delta_phi(pos_cls[1,:])

    rel_pos_cls = np.array([delta_etaCls, delta_phiCls, delta_zCls]).T
    deltaTsM_1D_cls.append(rel_pos_cls)

    '''
    min_eta = np.min(delta_etaTs)
    max_eta = np.max(delta_etaTs)
    min_phi = np.min(delta_phiTs)
    max_phi = np.max(delta_phiTs)
    min_z = np.min(delta_zTs)
    max_z = np.max(delta_zTs)
    min_eta_cls = np.min(delta_etaCls)
    max_eta_cls = np.max(delta_etaCls)
    min_phi_cls = np.min(delta_phiCls)
    max_phi_cls = np.max(delta_phiCls)
    min_z_cls = np.min(delta_zCls)
    max_z_cls = np.max(delta_zCls)
    '''

    min_eta = -0.4
    max_eta = 0.4
    min_phi = -0.4
    max_phi = 0.4
    min_z = -25
    max_z = 25
    min_eta_cls = -0.4
    max_eta_cls = 0.4
    min_phi_cls = -0.4
    max_phi_cls = 0.4
    min_z_cls = -25
    max_z_cls = 25



    fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins,range=[[min_eta,max_eta],[min_phi,max_phi], [min_z, max_z]])[0],
                        np.histogramdd(rel_pos, weights=np.asarray(en_ts), bins=bins, range=[[min_eta,max_eta],[min_phi,max_phi], [min_z, max_z]])[0]]))
    fTsM_g3D_cls.append(np.array([np.histogramdd(rel_pos_cls, bins=bins,range=[[min_eta_cls,max_eta_cls],[min_phi_cls,max_phi_cls], [min_z_cls, max_z_cls]])[0],
                        np.histogramdd(rel_pos_cls, weights=np.asarray(en_cls), bins=bins, range=[[min_eta_cls,max_eta_cls],[min_phi_cls,max_phi_cls], [min_z_cls, max_z_cls]])[0]], dtype= np.int16))

  fTsM_1D = fTsM_1D.snapshot()
  deltaTsM_1D = deltaTsM_1D.snapshot()
  deltaTsM_1D_cls = deltaTsM_1D_cls.snapshot()
  fTsM_g3D = fTsM_g3D.snapshot()
  fTsM_g3D_cls = fTsM_g3D_cls.snapshot()
  form, lenght, container = ak.to_buffers(fTsM_g3D)
  ak.to_parquet(fTsM_g3D, f"data/{suf}_grid_3D.parquet")
  form_3, lenght_3, container_3 = ak.to_buffers(fTsM_g3D_cls)
  ak.to_parquet(fTsM_g3D_cls, f"data/{suf}_grid_3D_cls.parquet")
  form_, lenght_, container_ = ak.to_buffers(fTsM_1D)
  ak.to_parquet(fTsM_1D, f"data/{suf}_fTsM_1D.parquet")
  form_1, lenght_1, container_1 = ak.to_buffers(deltaTsM_1D)
  ak.to_parquet(deltaTsM_1D, f"data/{suf}_deltaTsM_1D.parquet")
  form_7, lenght_7, container_7 = ak.to_buffers(deltaTsM_1D_cls)
  ak.to_parquet(deltaTsM_1D_cls, f"data/{suf}_deltaTsM_1D_cls.parquet")

  if DOPLOTS:
    print("####################")
    print("if this does not work is probably because we don't have enough TICLCandidate passing the energy cut")
    print("####################")
    vox_in = (np.array(fTsM_g3D_cls)[tTsM,0,:,:,:] > 0).astype(np.int32)
    fig = plt.figure(figsize=(30,30), dpi=200)
    for sp in range(1,26,1):
      ax = fig.add_subplot(5,5,sp, projection='3d')
      #ax = fig.gca( projection='3d')
      vox = vox_in[sp]
      ax.voxels(vox, shade=True, alpha=0.45)
    plt.savefig("voxel_test.png")
    plt.clf()

  return deltaTsM_1D, deltaTsM_1D, fTsM_1D, fTsM_g3D, fTsM_g3D_cls, tTsM


def plot_vars(deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM, suf):
    myhist(ak.flatten(deltaTsM_1D[0,:], axis=None), title="Delta_eta", xlabel="eta-eta_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_eta.png")
    plt.clf()
    myhist(ak.flatten(deltaTsM_1D[1,:], axis=None), title="Delta_phi", xlabel="phi-phi_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_phi.png")
    plt.clf()
    myhist(ak.flatten(deltaTsM_1D[2,:], axis=None), title="Delta_z", xlabel="z-z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_z.png")
    plt.clf()

    myhist(fTsM_1D[0,:], title="mean_eta", xlabel="eta_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_eta.png")
    plt.clf()
    myhist(fTsM_1D[1,:], title="mean_phi", xlabel="phi_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_phi.png")
    plt.clf()
    myhist(fTsM_1D[2,:], title="mean_z", xlabel="z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_z.png")
    plt.clf()
    myhist(np.array(tTsM, dtype=np.int32), title="Truth: en>en_min and score < score_max", xlabel="Passed", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_truth.png")
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
    deltaTsM_1D, deltaTsM_1D_cls, fTsM_1D, fTsM_g3D, fTsM_g3D_cls, tTsM = prepare_fromSim(f"data/histo_{suf}.root", suf, bins=bins, isPU=False)

    if DOPLOTS:
      plot_vars(deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM, suf)

      
      

  '''
  # This solution takes up too much disk space.
  with open('input.pkl', 'wb') as f:
    pickle.dump(fTsM_g3D, f)
  '''



  


if __name__=='__main__':
  main()
