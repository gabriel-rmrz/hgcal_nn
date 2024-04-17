import numpy as np
import awkward as ak
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def cnn_model(input_shape):
  # Definition of the model
  model = models.Sequential()
  
  model.add(layers.Conv3D(64, (3,3,3), activation='relu', input_shape=input_shape))
  model.add(layers.MaxPooling3D((2,2,2)))
 
  # Flatten the output
  model.add(layers.Flatten())

  model.add(layers.Dense(64, activation='relu'))

  model.add(layers.Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']) ### To define
  return model

def cnn_model2(input_shape):
  tf.keras.backend.clear_session()
  voxel_input = Input(shape=input_shape, name="voxel_grid")
  x = layers.Conv3D(8, (3,3,3), padding='valid', activation="relu")(voxel_input)
  x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Conv3D(16, (3,3,3), padding='valid', activation="relu")(x)
  x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Flatten()(x)

  scalar_input = Input(shape=(3,), name="scalar_features")
  y = layers.Dense(16, activation="relu")(scalar_input)
  combined = layers.concatenate([x,y])
  z = layers.Dense(32, activation="relu")(combined)
  output = layers.Dense(1, activation="sigmoid")(z)

  model = models.Model(inputs=[voxel_input, scalar_input], outputs=output)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model



def load_fromParquet(filename):
  return ak.from_parquet(filename)


def main():
  '''
  data/4Photon_PU200_deltaTsM_1D.parquet  data/4Pion_PU200_grid_3D.parquet
  data/4Photon_PU200_fTsM_1D.parquet      data/4Pion_PU200_tTsM.parquet
  data/4Photon_PU200_grid_3D.parquet      data/4Pions_0PU_deltaTsM_1D.parquet
  data/4Photon_PU200_tTsM.parquet         data/4Pions_0PU_fTsM_1D.parquet
  data/4Photons_0PU_deltaTsM_1D.parquet   data/4Pions_0PU_grid_3D.parquet
  data/4Photons_0PU_fTsM_1D.parquet       data/4Pions_0PU_tTsM.parquet
  data/4Photons_0PU_grid_3D.parquet       data/SinglePi_deltaTsM_1D.parquet
  data/4Photons_0PU_tTsM.parquet          data/SinglePi_fTsM_1D.parquet
  data/4Pion_PU200_deltaTsM_1D.parquet    data/SinglePi_grid_3D.parquet
  data/4Pion_PU200_fTsM_1D.parquet        data/SinglePi_tTsM.parquet
  '''
  #target_file='data/4Pion_PU200_tTsM.parquet'
  #features_1D_file ='data/4Pion_PU200_fTsM_1D.parquet'
  #features_3D_file ='data/4Pion_PU200_grid_3D.parquet'
  #target_file='data/4Photon_PU200_tTsM.parquet'
  #features_1D_file ='data/4Photon_PU200_fTsM_1D.parquet'
  #features_3D_file ='data/4Photon_PU200_grid_3D.parquet'
  #target_file='data/4Pions_0PU_tTsM.parquet'
  #features_1D_file ='data/4Pions_0PU_fTsM_1D.parquet'
  #features_3D_file ='data/4Pions_0PU_grid_3D.parquet'
  prefix = "4Pions_0PU"
  #prefix = "4Photons_0PU"
  target_file=f"data/{prefix}_tTsM.parquet"
  features_1D_file =f"data/{prefix}_fTsM_1D.parquet"
  #features_3D_file =f"data/{prefix}_grid_3D.parquet"
  features_3D_file =f"data/{prefix}_grid_3D_cls.parquet"
  tTsM = load_fromParquet(target_file)
  fTsM_1D = load_fromParquet(features_1D_file)
  fTsM_3D = load_fromParquet(features_3D_file)
  tTsM = np.asarray(tTsM)
  fTsM_1D = np.asarray(fTsM_1D)
  fTsM_3D = np.asarray(fTsM_3D)
  fTsM_3D = np.asarray(fTsM_3D)
  fTsM_3D = np.transpose(fTsM_3D, [0,2,3,4,1])
  '''
  for f3D in fTsM_3D[:,:,:,:,0]:
    suma = sum(ak.flatten(f3D, axis=None))
    if suma>1:
      print(suma)
  '''
  
  #fTsM_3D = fTsM_3D[:,:,:,:,0]
  input_3D_shape = (12, 12, 12,2)
  fTsM_3D_std_occ = np.std(fTsM_3D[:,:,:,:,0])
  fTsM_3D_mean_occ = np.mean(fTsM_3D[:,:,:,:,0])
  fTsM_3D[:,:,:,:,0]= (fTsM_3D[:,:,:,:,0] - fTsM_3D_mean_occ)/ fTsM_3D_std_occ

  fTsM_3D_std_en = np.std(fTsM_3D[:,:,:,:,1])
  fTsM_3D_mean_en = np.mean(fTsM_3D[:,:,:,:,1])
  fTsM_3D[:,:,:,:,1]= (fTsM_3D[:,:,:,:,1] - fTsM_3D_mean_en)/ fTsM_3D_std_en

  fTsM_1D_std_eta = np.std(fTsM_1D[:,0])
  fTsM_1D_mean_eta = np.mean(fTsM_1D[:,0])
  fTsM_1D[:,0] = (fTsM_1D[:,0] -fTsM_1D_mean_eta)/ fTsM_1D_std_eta

  fTsM_1D_std_phi = np.std(fTsM_1D[:,1])
  fTsM_1D_mean_phi = np.mean(fTsM_1D[:,1])
  fTsM_1D[:,1] = (fTsM_1D[:,1] -fTsM_1D_mean_phi)/ fTsM_1D_std_phi

  fTsM_1D_std_z = np.std(fTsM_1D[:,2])
  fTsM_1D_mean_z = np.mean(fTsM_1D[:,2])
  fTsM_1D[:,2] = (fTsM_1D[:,2] -fTsM_1D_mean_z)/ fTsM_1D_std_z
  print(fTsM_1D.shape)


  
  # Create the model
  '''
  model = cnn_model(input_3D_shape)
  model.summary()
  history = model.fit(
      fTsM_3D[:1000], tTsM[:1000],
      validation_data=(fTsM_3D[1000:1500], tTsM[1000:1500]),
    epochs=10,
    batch_size=32
  )
  print(history.history['accuracy'])
  '''

  thr = int(.6*len(tTsM))
  thr2=-1
  #thr = 100
  #thr2 = 200

  print(f"1-sum(tTsM[:thr])/len(tTsM[:thr]): {1-sum(tTsM[:thr])/len(tTsM[:thr])}")
  print(f"1-sum(tTsM[thr:thr2])/len(tTsM[thr:thr2]): {1-sum(tTsM[thr:thr2])/len(tTsM[thr:thr2])}")
  model2 = cnn_model2(input_3D_shape)
  model2.summary()
  history2 = model2.fit(
      [fTsM_3D[:thr], fTsM_1D[:thr]], tTsM[:thr],
      validation_data=([fTsM_3D[thr:thr2], fTsM_1D[thr:thr2]], tTsM[thr:thr2]),
      epochs= 20,
      batch_size= 32
      )
  plt.plot(history2.history['accuracy'], label="training")
  plt.plot(history2.history['val_accuracy'], label="validation")
  plt.legend(fontsize=10)
  plt.savefig('plots/accuracy_cnn.png')
  plt.clf()

  plt.plot(history2.history['loss'], label="training")
  plt.plot(history2.history['val_loss'], label="validation")
  plt.legend(fontsize=10)
  plt.plot(history2.history['loss'])
  plt.savefig('plots/loss_cnn.png')
  plt.clf()


  

if __name__ == '__main__':
  main()
