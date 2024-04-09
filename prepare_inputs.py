DEBUG=False
DOPLOTS=True
import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt

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

def prepare(filename, suf, isPU=False):
  print(f"Reading {filename}...")
  file = uproot.open(filename)
  
  simtrackstersSC = file["ticlDumper/simtrackstersSC"]
  simtrackstersCP = file["ticlDumper/simtrackstersCP"]
  tracksters = file["ticlDumper/tracksters"]
  linkedTracksters = file["ticlDumper/linkedTracksters"]
  trackstersMerged = file["ticlDumper/trackstersMerged"]
  associations = file["ticlDumper/associations"]
  tracks = file["ticlDumper/tracks"]
  simTICLCandidate = file["ticlDumper/simTICLCandidate"]
  TICLCandidate = file["ticlDumper/candidates"]
  clusters = file["ticlDumper/clusters"]
  
  simTICLCandidate_regressed_energy = simTICLCandidate["simTICLCandidate_regressed_energy"].array()
  
  indices_linkedTracksters = linkedTracksters["clue3Dts_indices"].array()

  candidate_raw_energy = TICLCandidate["candidate_raw_energy"].array()
  tsLinkedInCand = TICLCandidate["trackstersLinked_in_candidate"].array()

  tErrTs = tracksters["timeError"].array()
  bxTs = tracksters["barycenter_x"].array()
  bzTs = tracksters["barycenter_z"].array()
  betaTs = tracksters["barycenter_eta"].array()
  bphiTs = tracksters["barycenter_phi"].array()
  reg_enTs = tracksters["regressed_energy"].array()

  recoToSim_en = associations["Mergetracksters_recoToSim_CP_sharedE"].array()
  recoToSim_score = associations["Mergetracksters_recoToSim_CP_score"].array()
  recoToSim_index = associations["Mergetracksters_recoToSim_CP"].array()
  
  simToReco_en = associations["Mergetracksters_simToReco_CP_sharedE"].array()
  simToReco_score = associations["Mergetracksters_simToReco_CP_score"].array()
  simToReco_index = associations["Mergetracksters_simToReco_CP"].array()
  
  '''
  simToRecoPU_en = associations["Mergetracksters_simToReco_PU_sharedE"].array()
  simToRecoPU_score = associations["Mergetracksters_simToReco_PU_score"].array()
  simToRecoPU_index = associations["Mergetracksters_simToReco_PU"].array()
  
  recoToSimPU_en = associations["Mergetracksters_recoToSim_PU_sharedE"].array()
  recoToSimPU_score = associations["Mergetracksters_recoToSim_PU_score"].array()
  recoToSimPU_index = associations["Mergetracksters_recoToSim_PU"].array()
  '''

  # Creating the voxels for the different trackstersMerged
  #features_perTracksterMerged
  fTsM_g3D = ak.ArrayBuilder()
  fTsM_1D = ak.ArrayBuilder()
  deltaTsM_1D = ak.ArrayBuilder()
  #target_perTracksterMerged
  tTsM = ak.ArrayBuilder()

  #for i, ev in enumerate(indices_linkedTracksters): # looping over all the events
  #for i, ev in enumerate(tsLinkedInCand): # looping over all the events
  for i, ev in enumerate(tsLinkedInCand[:10]): # looping over all the events
    if not (i%10) :
      print(f"%%%%%%%%%%%%%%% Event {i} %%%%%%%%%%%%%%%")
    '''
    print(f"simToReco_index[{i}]: {simToReco_index[i]}")
    print(f"len(simToReco_index[{i}]): {len(simToReco_index[i])}")
    print(f"simToReco_index[{i}][0]: {simToReco_index[i][0]}") # Explore the first simulated particle
    print(f"len(simToReco_index[{i}][0]): {len(simToReco_index[i][0])}")
    for j in range(len(simToReco_index[i])):
      print("############")
      print(f"j: {j}")
      print("############")
      isPassScoreSim = simToReco_score[i][j] < 0.35
      print(f"simToReco_score[i][j]: {simToReco_score[i][j]}")
      for simTs0 in simToReco_index[i][j][isPassScoreSim]:
        print(f"candidate_raw_energy[{i}][{simTs0}][isPassScoreSim]: {candidate_raw_energy[i][simTs0]}")
    exit()
    '''
    print(ev.type)
    for j, mTs in enumerate(ev): #looping over the mergedTracksters 
      ###################################
      ## Defition of the inputs
      ##################################
      #betaTs_mean = np.mean(betaTs[i][mTs])
      min_abseta= 1.2
      max_abseta = 3.3
      betaTs_mean = (min_abseta+max_abseta)/2.
      bphiTs_mean = np.mean(bphiTs[i][mTs])
      bzTs_mean = np.mean(bzTs[i][mTs])

      delta_etaTs = np.abs(betaTs[i][mTs]) - min_abseta
      delta_zTs = bzTs[i][mTs] - bzTs_mean

      '''
      print(bzTs_mean)
      print(bzTs[i][mTs])
      print(delta_zTs)
      '''

      delta_phiTs, bphiTs_mean = get_delta_phi(bphiTs[i][mTs])
      min_phi = 1.1*np.min(delta_phiTs)
      max_phi = 1.1*np.max(delta_phiTs)
      min_z = 1.1*np.min(delta_zTs)
      max_z = 1.1*np.max(delta_zTs)

      rel_pos = np.array([delta_etaTs, delta_phiTs, delta_zTs]).T

      bins = [8,8,8]
      if DEBUG:
        print(rel_pos)
        print(reg_enTs[i][mTs])
      fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins,range=[[min_abseta,max_abseta],[min_phi,max_phi], [min_z, max_z]])[0],
                          np.histogramdd(rel_pos, weights=np.asarray(reg_enTs[i][mTs]), bins=bins, range=[[min_abseta,max_abseta],[min_phi,max_phi], [min_z, max_z]])[0]]))
      fTsM_1D.append(np.array([betaTs_mean, bphiTs_mean, bzTs_mean]))
      deltaTsM_1D.append(np.array([delta_etaTs, delta_phiTs, delta_zTs]))
      if DEBUG:
        print(bxTs[i][mTs])
        bxTs[i][mTs]
      '''
      for k, ts in enumerate(bxTs[i][mTs]): #looping over the tracksters in every mergedTracksters
        print(ts)
        #print(tErrTs[mTs][k] > -0.5) # IMPORTANT: not to ignore the trackster with negative time error.
        #print(ts[tErrTs[mTs][k] > -0.5])
      '''
      #Probably I can use ravel and unravel when writing and reading.
      ###################################
      ## Defition of the target
      ##################################
      max_score_r2s = 0.35
      min_energy_r2s = 0.5
      #isPass = ((recoToSim_score[i][j] < max_score_r2s) & (recoToSim_en[i][j] > min_energy_r2s))
      isPassScore = recoToSim_score[i][j] < max_score_r2s 
      #print(f"recoToSim_index[i][j]: {recoToSim_index[i][j]}")
      #print(f"recoToSim_score[{i}][{j}][isPassScore]: {recoToSim_score[i][j][isPassScore]}")
      #print(f"recoToSim_en[{i}][{j}][isPassScore]: {recoToSim_en[i][j][isPassScore]}")
      #print(f"recoToSim_index[{i}][{j}][isPassScore]: {recoToSim_index[i][j][isPassScore]}")
      #print(f"isPassScore: {isPassScore}")
      en_frac = recoToSim_en[i][j][isPassScore]/simTICLCandidate_regressed_energy[i][recoToSim_index[i][j][isPassScore]]


      isPass = False
      if np.any(en_frac > 0.5):
        if DEBUG:
          print(f"i / recoToSim_index[i][j][isPassScore]: {i} / {recoToSim_index[i][j][isPassScore]}")
          print(f"en_frac: {en_frac}")
        isPass = True
      tTsM.append(np.array([isPass]))

  fTsM_g3D = fTsM_g3D.snapshot()
  fTsM_1D = fTsM_1D.snapshot()
  deltaTsM_1D = deltaTsM_1D.snapshot()
  tTsM = tTsM.snapshot()
  form, lenght, container = ak.to_buffers(fTsM_g3D)
  ak.to_parquet(fTsM_g3D, f"data/{suf}_grid_3D.parquet")
  form_, lenght_, container_ = ak.to_buffers(fTsM_1D)
  ak.to_parquet(fTsM_1D, f"data/{suf}_fTsM_1D.parquet")
  form_1, lenght_1, container_1 = ak.to_buffers(deltaTsM_1D)
  ak.to_parquet(deltaTsM_1D, f"data/{suf}_deltaTsM_1D.parquet")
  form_2, lenght_2, container_2 = ak.to_buffers(tTsM)
  ak.to_parquet(tTsM, f"data/{suf}_tTsM.parquet")
  return deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM

def plot_vars(deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM, suf):
    myhist(ak.flatten(deltaTsM_1D[:,0], axis=None), title="Delta_eta", xlabel="eta-eta_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_eta.png")
    plt.clf()
    myhist(ak.flatten(deltaTsM_1D[:,1], axis=None), title="Delta_phi", xlabel="phi-phi_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_phi.png")
    plt.clf()
    myhist(ak.flatten(deltaTsM_1D[:,2], axis=None), title="Delta_z", xlabel="z-z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_delta_z.png")
    plt.clf()

    myhist(fTsM_1D[:,0], title="mean_eta", xlabel="eta_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_eta.png")
    plt.clf()
    myhist(fTsM_1D[:,1], title="mean_phi", xlabel="phi_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
    plt.savefig(f"plots/{suf}_val_mean_phi.png")
    plt.clf()
    myhist(fTsM_1D[:,2], title="mean_z", xlabel="z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
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
  sample_label = "4pions"
  file_sufix = [
   #'4Pion_PU200',
   '4Photon_PU200',
   '4Photons_0PU',                    
   '4Pions_0PU',
   'SinglePi']
  for suf in file_sufix:
    deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM = prepare(f"data/histo_{suf}.root", suf, isPU=False)
    print(deltaTsM_1D.type)
    print(fTsM_1D.type)

    if DOPLOTS:
      plot_vars(deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM, suf)

      
      

  '''
  # This solution takes up too much disk space.
  with open('input.pkl', 'wb') as f:
    pickle.dump(fTsM_g3D, f)
  '''



  


if __name__=='__main__':
  main()
