DEBUG=False
import awkward as ak
import numpy as np
import uproot as uproot
#import pickle

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
  filename = 'histo_4Pions_0PU_pt10to100_eta17to27.root'
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


  simTICLCandidate_simTracksterCPIndex = simTICLCandidate["simTICLCandidate_simTracksterCPIndex"].array()
  simTICLCandidate_pdgId = simTICLCandidate["simTICLCandidate_pdgId"].array()
  simTICLCandidate_charge = simTICLCandidate["simTICLCandidate_charge"].array()
  
  track_pt = tracks["track_pt"].array()
  track_id = tracks["track_id"].array()
  track_hgcal_eta = tracks["track_hgcal_eta"].array()
  track_hgcal_pt = tracks["track_hgcal_pt"].array()
  track_missing_outer_hits = tracks["track_missing_outer_hits"].array()
  track_missing_inner_hits = tracks["track_missing_inner_hits"].array()
  track_nhits = tracks["track_nhits"].array()
  track_quality = tracks["track_quality"].array()
  track_time_mtd_err = tracks["track_time_mtd_err"].array()
  track_isMuon = tracks["track_isMuon"].array()
  track_isTrackerMuon = tracks["track_isTrackerMuon"].array()
  
  
  track_boundaryX = simtrackstersSC["track_boundaryX"].array()
  track_boundaryY = simtrackstersSC["track_boundaryY"].array()
  track_boundaryZ = simtrackstersSC["track_boundaryZ"].array()
  
  simTICLCandidate_time = simTICLCandidate["simTICLCandidate_time"].array()
  simTICLCandidate_raw_energy = simTICLCandidate["simTICLCandidate_raw_energy"].array()
  simTICLCandidate_regressed_energy = simTICLCandidate["simTICLCandidate_regressed_energy"].array()
  simTICLCandidate_track_in_candidate = simTICLCandidate["simTICLCandidate_track_in_candidate"].array()
  
  indices_linkedTracksters = linkedTracksters["clue3Dts_indices"].array()

  candidate_pdgId = TICLCandidate["candidate_pdgId"].array()
  candidate_id_prob = TICLCandidate["candidate_id_probabilities"].array()
  tracksters_in_candidate = TICLCandidate["tracksters_in_candidate"].array()
  track_in_candidate = TICLCandidate["track_in_candidate"].array()
  candidate_energy = TICLCandidate["candidate_energy"].array()
  candidate_raw_energy = TICLCandidate["candidate_raw_energy"].array()
  candidate_time = TICLCandidate["candidate_time"].array()
  candidate_timeErr = TICLCandidate["candidate_timeErr"].array()
  NCandidates = TICLCandidate["NCandidates"]
  trackstersMerged_rawEne = trackstersMerged["raw_energy"].array()
  tsLinkedInCand = TICLCandidate["trackstersLinked_in_candidate"].array()
  
  ntrackstersMerged = trackstersMerged["NTrackstersMerged"]
  bxTsM = trackstersMerged["barycenter_x"].array()
  byTsM = trackstersMerged["barycenter_y"].array()
  bzTsM = trackstersMerged["barycenter_z"].array()

  tErrTs = tracksters["timeError"].array()
  bxTs = tracksters["barycenter_x"].array()
  byTs = tracksters["barycenter_y"].array()
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
  
  simToRecoPU_en = associations["Mergetracksters_simToReco_PU_sharedE"].array()
  simToRecoPU_score = associations["Mergetracksters_simToReco_PU_score"].array()
  simToRecoPU_index = associations["Mergetracksters_simToReco_PU"].array()
  
  recoToSimPU_en = associations["Mergetracksters_recoToSim_PU_sharedE"].array()
  recoToSimPU_score = associations["Mergetracksters_recoToSim_PU_score"].array()
  recoToSimPU_index = associations["Mergetracksters_recoToSim_PU"].array()

  
  SC_boundx = simtrackstersSC["boundaryX"].array()
  SC_boundy = simtrackstersSC["boundaryY"].array()
  SC_boundz = simtrackstersSC["boundaryZ"].array()
  SC_bx = simtrackstersSC["barycenter_x"].array()
  SC_by = simtrackstersSC["barycenter_y"].array()
  SC_bz = simtrackstersSC["barycenter_z"].array()
  SC_boundary_time = simtrackstersSC["timeBoundary"].array()
  SC_CALO_time = simtrackstersSC["time"].array()
  SC_CALO_timeErr = simtrackstersSC["timeError"].array()
  SC_trackIdx = simtrackstersSC["trackIdx"].array()

  # Creating the voxels for the different trackstersMerged
  #features_perTracksterMerged
  fTsM_g3D = ak.ArrayBuilder()
  fTsM_1D = ak.ArrayBuilder()
  #target_perTracksterMerged
  tTsM = ak.ArrayBuilder()

  #for i, ev in enumerate(indices_linkedTracksters): # looping over all the events
  for i, ev in enumerate(tsLinkedInCand): # looping over all the events
    if not (i%10) and i>0:
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
    for j, mTs in enumerate(ev): #looping over the mergedTracksters 
      ###################################
      ## Defition of the inputs
      ##################################
      betaTs_mean = np.mean(betaTs[i][mTs])
      bphiTs_mean = np.mean(bphiTs[i][mTs])
      bzTs_mean = np.mean(bzTs[i][mTs])

      delta_etaTs = betaTs[i][mTs] - betaTs_mean
      delta_zTs = bzTs[i][mTs] - bzTs_mean

      # Computing the difference on phi making sure to keep the convention of -pi - pi
      # Probably this can be done more easily using 2D-vectors.
      bphiTs_mean = np.mean(bphiTs[i][mTs])
      if np.any(bphiTs[i][mTs] < -np.pi/2) and np.any(bphiTs[i][mTs] > np.pi/2):
        bphiTs_mean_temp = np.mean(np.where(bphiTs[i][mTs] < 0, bphiTs[i][mTs] +2*np.pi, bphiTs[i][mTs]))
        if bphiTs_mean_temp < -np.pi:
          bphiTs_mean = bphiTs_mean_temp + 2*np.pi
        elif bphiTs_mean_temp > np.pi:
          bphiTs_mean = bphiTs_mean_temp - 2*np.pi

      delta_phiTs = np.where(np.sign(bphiTs[i][mTs]) == np.sign(bphiTs_mean), 
                             bphiTs[i][mTs]- bphiTs_mean, 
                             bphiTs[i][mTs]+ np.sign(bphiTs_mean)*2*np.pi - bphiTs_mean)
      delta_phiTs= np.where(delta_phiTs < -np.pi, delta_phiTs + 2*np.pi, delta_phiTs)
      delta_phiTs= np.where(delta_phiTs > np.pi, delta_phiTs - 2*np.pi, delta_phiTs)

      rel_pos = np.array([delta_etaTs, delta_phiTs, delta_zTs]).T

      bins = [8,8,8]
      if DEBUG:
        print(rel_pos)
        print(reg_enTs[i][mTs])
      fTsM_g3D.append(np.array([np.histogramdd(rel_pos, bins=bins)[0],
                          np.histogramdd(rel_pos, weights=np.asarray(reg_enTs[i][mTs]), bins=bins)[0]]))
      fTsM_1D.append(np.array([betaTs_mean, bphiTs_mean, bzTs_mean]))
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
  tTsM = tTsM.snapshot()
  form, lenght, container = ak.to_buffers(fTsM_g3D)
  ak.to_parquet(fTsM_g3D, "grid_3D.parquet")
  form_, lenght_, container_ = ak.to_buffers(fTsM_1D)
  ak.to_parquet(fTsM_1D, "fTsM_1D.parquet")
  form_2, lenght_2, container_2 = ak.to_buffers(tTsM)
  ak.to_parquet(tTsM, "tTsM.parquet")

  '''
  # This solution takes up too much disck space.
  with open('input.pkl', 'wb') as f:
    pickle.dump(fTsM_g3D, f)
  '''

      
      




  


if __name__=='__main__':
  main()
