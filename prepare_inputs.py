DEBUG=False
DOPLOTS=True
import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
  
  simtrackstersCP_raw_energy = simtrackstersCP["raw_energy"].array()

  tsLinkedInCand = TICLCandidate["trackstersLinked_in_candidate"].array()

  tErrTs = tracksters["timeError"].array()
  bxTs = tracksters["barycenter_x"].array()
  bzTs = tracksters["barycenter_z"].array()
  betaTs = tracksters["barycenter_eta"].array()
  bphiTs = tracksters["barycenter_phi"].array()
  reg_enTs = tracksters["regressed_energy"].array()
  raw_enTs = tracksters["raw_energy"].array()

  recoToSim_en = associations["Mergetracksters_recoToSim_CP_sharedE"].array()
  recoToSim_score = associations["Mergetracksters_recoToSim_CP_score"].array()
  recoToSim_index = associations["Mergetracksters_recoToSim_CP"].array()
  
  simToReco_en = associations["Mergetracksters_simToReco_CP_sharedE"].array()
  simToReco_score = associations["Mergetracksters_simToReco_CP_score"].array()
  simToReco_index = associations["Mergetracksters_simToReco_CP"].array()
  
  # Creating the voxels for the different trackstersMerged
  #features_perTracksterMerged
  fTsM_g3D = ak.ArrayBuilder()
  fTsM_1D = ak.ArrayBuilder()
  deltaTsM_1D = ak.ArrayBuilder()
  #target_perTracksterMerged
  tTsM = ak.ArrayBuilder()

  for i, ev in enumerate(simToReco_index): # looping over all the events
    if not (i%10) :
      print(f"%%%%%%%%%%%%%%% Event {i} %%%%%%%%%%%%%%%")
    print(ev.type)
    for j, s in enumerate(ev): #looping over the mergedTracksters 
      isPassScore0 =ak.flatten(simToReco_score[i], axis=None) < 0.35
      simMatched = s[isPassScore0]
      #simMatched = s[simToReco_score[i] < 0.35]
      mTsM = ak.flatten(simToReco_index[i])[isPassScore0]
      print(f"tsLinkedInCand[i][mTsM].type: {tsLinkedInCand[i][mTsM].type}")
      print(f"tsLinkedInCand[i][mTsM]: {tsLinkedInCand[i][mTsM]}")
      mTs = ak.flatten(tsLinkedInCand[i][mTsM])
      mTs_en = ak.flatten(simToReco_en[i])[isPassScore0]
      if len(mTs)==0:
        continue

      ###################################
      ## Defition of the inputs
      '''
      min_abseta= 1.2
      max_abseta = 3.3
      betaTs_mean = (min_abseta+max_abseta)/2.
      delta_etaTs = np.abs(betaTs[i][mTs]) - min_abseta
      '''
      print(f"betaTs[i][mTs]: {betaTs[i][mTs]}")
      betaTs_mean = np.mean(betaTs[i][mTs])
      delta_etaTs = betaTs[i][mTs] - betaTs_mean
      min_eta = np.min(delta_etaTs)
      max_eta = np.max(delta_etaTs)

      bphiTs_mean = np.mean(bphiTs[i][mTs])
      bzTs_mean = np.mean(bzTs[i][mTs])

      delta_zTs = bzTs[i][mTs] - bzTs_mean


      delta_phiTs, bphiTs_mean = get_delta_phi(bphiTs[i][mTs])
      #TODO: Check the 1.1 factor for the mins.
      min_phi = np.min(delta_phiTs)
      max_phi = np.max(delta_phiTs)
      min_z = np.min(delta_zTs)
      max_z = np.max(delta_zTs)

      rel_pos = np.array([delta_etaTs, delta_phiTs, delta_zTs]).T

      if DEBUG:
        print(rel_pos)
        print(raw_enTs[i][mTs])
      fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins,range=[[min_eta,max_eta],[min_phi,max_phi], [min_z, max_z]])[0],
                          np.histogramdd(rel_pos, weights=np.asarray(raw_enTs[i][mTs]), bins=bins, range=[[min_eta,max_eta],[min_phi,max_phi], [min_z, max_z]])[0]]))
      #fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins,range=[[min_abseta,max_abseta],[min_phi,max_phi], [min_z, max_z]])[0],
      #                    np.histogramdd(rel_pos, weights=np.asarray(raw_enTs[i][mTs]), bins=bins, range=[[min_abseta,max_abseta],[min_phi,max_phi], [min_z, max_z]])[0]]))
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
      #en_frac = recoToSim_en[i][j][isPassScore]/simTICLCandidate_regressed_energy[i][recoToSim_index[i][j][isPassScore]]
      #en_frac = simToReco_en[i][j][isPassScore]/simtrackstersCP_raw_energy[i][j]
      en_frac = mTs_en/simtrackstersCP_raw_energy[i][j]
      print(f"mTs_en:{mTs_en}")
      print(en_frac)


      isPass = False
      if np.any(en_frac > 0.5):
        isPass = True
      tTsM.append(isPass)

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

  print(f"tTsM.type: { tTsM.type}")
  print(f"len(tTsM): { len(tTsM)}")
  print(f"np.sum(tTsM): { np.sum(tTsM)}")

  if DOPLOTS:
    print("####################")
    print("if this does not work is probably because we don't have enough TICLCandidate passing the energy cut")
    print("####################")
    vox_in = (np.array(fTsM_g3D)[tTsM,0,:,:,:] > 0).astype(np.int32)
    fig = plt.figure(figsize=(30,30), dpi=200)
    for sp in range(1,26,1):
      ax = fig.add_subplot(5,5,sp, projection='3d')
      #ax = fig.gca( projection='3d')
      vox = vox_in[sp]
      print(sp)
      print(vox)
      ax.voxels(vox, shade=True, alpha=0.45)
    plt.savefig("voxel_test.png")
    plt.clf()

  return deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM

def prepare(filename, suf, bins= [6,6,6], isPU=False):
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
  simTICLCandidate_raw_energy = simTICLCandidate["simTICLCandidate_raw_energy"].array()
  
  indices_linkedTracksters = linkedTracksters["clue3Dts_indices"].array()

  candidate_raw_energy = TICLCandidate["candidate_raw_energy"].array()
  tsLinkedInCand = TICLCandidate["trackstersLinked_in_candidate"].array()

  tErrTs = tracksters["timeError"].array()
  bxTs = tracksters["barycenter_x"].array()
  bzTs = tracksters["barycenter_z"].array()
  betaTs = tracksters["barycenter_eta"].array()
  bphiTs = tracksters["barycenter_phi"].array()
  reg_enTs = tracksters["regressed_energy"].array()
  raw_enTs = tracksters["regressed_energy"].array()

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
  print(len(tsLinkedInCand))

  # Creating the voxels for the different trackstersMerged
  #features_perTracksterMerged
  fTsM_pos = ak.ArrayBuilder()

  fTsM_g3D = ak.ArrayBuilder()
  fTsM_1D = ak.ArrayBuilder()
  deltaTsM_1D = ak.ArrayBuilder()
  #target_perTracksterMerged
  tTsM = ak.ArrayBuilder()

  #for i, ev in enumerate(indices_linkedTracksters): # looping over all the events
  #for i, ev in enumerate(simToReco_index): # looping over all the events
  #for i, ev in enumerate(simToReco_index[:100]): # looping over all the events
  for i, ev in enumerate(tsLinkedInCand): # looping over all the events
  #for i, ev in enumerate(tsLinkedInCand[:40]): # looping over all the events
    if not (i%10) :
      print(f"%%%%%%%%%%%%%%% Event {i} %%%%%%%%%%%%%%%")
    print(ev.type)
    for j, mTs in enumerate(ev): #looping over the mergedTracksters 

      ###################################
      ## Defition of the inputs
      '''
      min_abseta= 1.2
      max_abseta = 3.3
      betaTs_mean = (min_abseta+max_abseta)/2.
      delta_etaTs = np.abs(betaTs[i][mTs]) - min_abseta
      '''
      betaTs_mean = np.mean(betaTs[i][mTs])
      delta_etaTs = betaTs[i][mTs] - betaTs_mean
      min_eta = np.min(delta_etaTs)
      max_eta = np.max(delta_etaTs)

      bphiTs_mean = np.mean(bphiTs[i][mTs])
      bzTs_mean = np.mean(bzTs[i][mTs])

      delta_zTs = bzTs[i][mTs] - bzTs_mean


      delta_phiTs, bphiTs_mean = get_delta_phi(bphiTs[i][mTs])
      #TODO: Check the 1.1 factor for the mins.
      min_phi = np.min(delta_phiTs)
      max_phi = np.max(delta_phiTs)
      min_z = np.min(delta_zTs)
      max_z = np.max(delta_zTs)

      rel_pos = np.array([delta_etaTs, delta_phiTs, delta_zTs]).T
      fTsM_pos.append(rel_pos)

      if DEBUG:
        print(rel_pos)
        print(raw_enTs[i][mTs])
      fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins,range=[[min_eta,max_eta],[min_phi,max_phi], [min_z, max_z]])[0],
                          np.histogramdd(rel_pos, weights=np.asarray(raw_enTs[i][mTs]), bins=bins, range=[[min_eta,max_eta],[min_phi,max_phi], [min_z, max_z]])[0]]))
      #fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins,range=[[min_abseta,max_abseta],[min_phi,max_phi], [min_z, max_z]])[0],
      #                    np.histogramdd(rel_pos, weights=np.asarray(raw_enTs[i][mTs]), bins=bins, range=[[min_abseta,max_abseta],[min_phi,max_phi], [min_z, max_z]])[0]]))
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
      #en_frac = recoToSim_en[i][j][isPassScore]/simTICLCandidate_regressed_energy[i][recoToSim_index[i][j][isPassScore]]
      en_frac = recoToSim_en[i][j][isPassScore]/simTICLCandidate_raw_energy[i][recoToSim_index[i][j][isPassScore]]


      isPass = False
      if np.any(en_frac > 0.5):
        isPass = True
      tTsM.append(isPass)
  fTsM_pos = fTsM_pos.snapshot()

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
  form_3, lenght_3, container_3 = ak.to_buffers(fTsM_g3D)
  ak.to_parquet(fTsM_pos, f"data/{suf}_point_cloud.parquet")

  print(f"tTsM.type: { tTsM.type}")
  print(f"len(tTsM): { len(tTsM)}")
  print(f"np.sum(tTsM): { np.sum(tTsM)}")

  if DOPLOTS:
    print("####################")
    print("if this does not work is probably because we don't have enough TICLCandidate passing the energy cut")
    print("####################")
    vox_in = (np.array(fTsM_g3D)[tTsM,0,:,:,:] > 0).astype(np.int32)
    fig = plt.figure(figsize=(30,30), dpi=200)
    for sp in range(1,26,1):
      ax = fig.add_subplot(5,5,sp, projection='3d')
      #ax = fig.gca( projection='3d')
      vox = vox_in[sp]
      print(sp)
      print(vox)
      ax.voxels(vox, shade=True, alpha=0.45)
    plt.savefig("voxel_test.png")
    plt.clf()

  return deltaTsM_1D, fTsM_1D, fTsM_g3D, fTsM_pos, tTsM

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
  file_sufix = [
   #'kaon_PU75']
   #'4Pion_PU200',
   '4Photons_0PU',                    
   '4Pions_0PU',
   'SinglePi']
   
   #'4Photon_PU200'
   #]
  bins = [8,8,8]
  for suf in file_sufix:
    deltaTsM_1D, fTsM_1D, fTsM_g3D, fTsM_pos, tTsM = prepare(f"data/histo_{suf}.root", suf, bins=bins, isPU=False)
    #deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM = prepare_fromSim(f"data/histo_{suf}.root", suf, bins=bins, isPU=False)
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
