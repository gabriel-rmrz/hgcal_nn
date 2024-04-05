### TODO:
 
 #### General
 - Check convention for phi
 - [ ] Settup RECAS machine to run training.
   - [ ]Run simple example as test.

 #### Tree parameters
 - [ ] Convert the point of the point clouds of the trees to voxel before using CLUE

 #### Binary classifier
 - [ ] Preparation of the features for the Binary classifier.
   - [ ] Define aggregation using the voxels information by layers or as a whole with the eta and phi information.
   - Unordered array.
     - PointNet?
 - [ ] Binary classifier.
   - [x] Voxels.
     - [ ] Recomendation by Felice: Use the full range for the eta to avoid a dependency of the algorith on this variable.
       The potential problem with eta covering the whole range is that regions are going to be very big and this will cause either to have too many voxel or to big ones
 #### Open questions:
 - What particles are we able to see and how do they differ among themselves. (physic of the detector)
 - How is the energy, position and time of the CaloParticle treated?
 - What is EV{1,2,3} in the tracksters variables. (Eigenvalues?)
 - What is evector_{1,2,3} (eigenvectors) and sigmaPCA{1,2,3}. Are they connected to a Principal Components Analysis performed on the tracksters?
 - How is the granularity of the detector? Can every hexagon detect only one deposit of energy at the time or do they have some kind of internal structure.
   Every hexagon is divided in different sub-regions (horizontally or vertically) depending on the part of the detector.

#### 2.4.2024
 - The fact that there are some tracksters with negative time error shouldn't be taken into acount (in other words we can keep them). This is explained when the rechit are not energetic enough (or they don't pass other cuts along the way) the time is not computed, but the rest of the information is still OK.
 - Why is there more than one track_hgcal_eta per event, in some cases, for our currente sample? And some times none.
   Because the trackstersMerge are matched to tracks, and some time it can match more than one. Ideally it will not match any for neutral particles.
 - Are we going to have problems with the edges?
 - Binary classifier.
   - Defintion of the target:
     If the RECO TICLCandidate that matches the simSC have a percentage of the simCP we consider it as complete. reco_en[asso.simSC_s2r]/simCP[asso.simCP_r2s]
   - Definition of the inputs:
     - We use all the TICLCandidate voxel grid for the energy and positions.
     - If we define the voxel for the whole region of Eta then this will be a constant parmeter and NOT used as a feature.
     - Phi, time
   - Can we make a cluster energy 3D map ( or energy density map) base on the clusters belonging to the tracksters merged?
     For the moment, I think it should be enought to have the grid/map of the energy of the trackters in a trackterMerged.

#### 21.3.2024
 - [ ] Convert the point of the point clouds of the trees to voxel before using CLUE

#### 19.3.2024
 - Undestanding of the problem.
   - Important definitions:
     - TICLCandidate are objects composed by tracksters, more specifically by a tracksterMerged and (if present) a (some) track.. 
     - A trackster is a cluster formed by the 2D-clusters defined in each layer. 
     - Layer clusters are conglomarates of RecHits defined in the surface of every layer, computed by CLUE.
     - Time information.
       - The time infomation comes from RecHit level. To compute the layer clusters a weighted (by the errors) average of the RecHits times.
     - SIM objects with the same structure a the reco object to used as Truth.
       - At this level two kinds of objects are defined:
         - CaloParticles: Which seem to match the GEN particles.
         - SimParticle: Simulated particules containing the simulated interaction with the detector.
     
   - Statement of the problem.
     - Implement a binnary classifier to evaluate completeness of the TICLCandidates using the information of it's tracksters (trackstersMerged) (and maybe also the information of it's linked tracks)


#### 18.3.2024
 - [ ] Settup RECAS machine to run training.
   - [x] Review Adriano's tutorial.
   - [x] Test access to Titan sever.
   - [ ]Run simple example as test.

 - [ ] Binary classifier.
   - [x] Voxels.
   - [ ] Define aggregation using the voxels information by layers or as a whole with the eta and phi information.
   - Unordered array.
     - PointNet?

