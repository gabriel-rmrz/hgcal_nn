import numpy as np
import awkward as ak
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.metrics import Precision, Recall, F1Score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def weighted_binary_crossentropy(y_true, y_pred):
  weights = (y_true * 1) + (1 - y_true) * 1.5  # Increase weight for negatives
  bce = K.binary_crossentropy(y_true, y_pred)
  weighted_bce = K.mean(bce * weights)
  return weighted_bce
def focal_loss(gamma=2., alpha=4.):
  gamma = float(gamma)
  alpha = float(alpha)
  def focal_loss_fixed(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true = K.cast(y_true, y_pred.dtype)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(1-y_pred)
    fl = - K.log(p_t) * K.pow((K.ones_like(y_true)-p_t), gamma)
    loss = K.sum(alpha_t * fl, axis=1)
    return loss
  return focal_loss_fixed

def cnn_model(input_shape):
  tf.keras.backend.clear_session()
  voxel_input = Input(shape=input_shape, name="voxel_grid")
  x = layers.Conv3D(16, (3,3,3), padding='valid', activation="relu")(voxel_input)
  x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Dropout(0.1)(x)
  #x = layers.Conv3D(64, (3,3,3), padding='valid', activation="relu")(x)
  #x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Flatten()(x)

  x = layers.Dense(16, activation="relu")(x)
  output = layers.Dense(1, activation="sigmoid")(x)
  model = models.Model(inputs=voxel_input, outputs=output)
  model.compile(optimizer='adam', loss=waighted_binary_cossentropy, metrics=['accuracy', Precision(), Recall()])

  return model

def cnn_model2(input_shape):
  tf.keras.backend.clear_session()
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=1e-4,
      decay_steps=10000,
      decay_rate=0.9)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
  l2_lambda = 0.05  # Lambda for L2 regularization
  voxel_input = Input(shape=input_shape, name="voxel_grid")
  x = layers.Conv3D(16, (3,3,3),  activation="relu")(voxel_input)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv3D(32, (2,2,2),  activation="relu")(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv3D(64, (3,3,3),  activation="relu")(voxel_input)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling3D((2,2,2))(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Flatten()(x)

  scalar_input = Input(shape=(3,), name="scalar_features")
  y = layers.Dense(32, activation="relu")(scalar_input)
  combined = layers.concatenate([x,y])
  z = layers.Dropout(0.25)(combined)  # Dropout rate of 50%
  z = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda))(z)
  #z = layers.Dense(32, activation="relu")(z)
  output = layers.Dense(1, activation="sigmoid")(z)

  model = models.Model(inputs=[voxel_input, scalar_input], outputs=output)
  #model.compile(optimizer='adam', loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy', Precision(), Recall()])
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
  #features_1D_file =f"data/{prefix}_fTsM_1D.parquet"
  features_1D_file =f"data/{prefix}_fTsM_1D_cls.parquet"
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
  
  input_3D_shape = (24, 24, 16,2)
  #fTsM_3D = fTsM_3D[:,:,:,:,1]
  fTsM_3D_std_occ = np.std(fTsM_3D[:,:,:,:,0])
  fTsM_3D_mean_occ = np.mean(fTsM_3D[:,:,:,:,0])
  fTsM_3D[:,:,:,:,0]= (fTsM_3D[:,:,:,:,0] - fTsM_3D_mean_occ)/ fTsM_3D_std_occ

  fTsM_3D_std_en = np.std(fTsM_3D[:,:,:,:,1])
  fTsM_3D_mean_en = np.mean(fTsM_3D[:,:,:,:,1])
  fTsM_3D[:,:,:,:,1]= (fTsM_3D[:,:,:,:,1] - fTsM_3D_mean_en)/ fTsM_3D_std_en

  fTsM_1D_std_eta = np.std(fTsM_1D[0,:])
  fTsM_1D_mean_eta = np.mean(fTsM_1D[0,:])
  fTsM_1D[0,:] = (fTsM_1D[0,:] -fTsM_1D_mean_eta)/ fTsM_1D_std_eta

  fTsM_1D_std_phi = np.std(fTsM_1D[1,:])
  fTsM_1D_mean_phi = np.mean(fTsM_1D[1,:])
  fTsM_1D[1,:] = (fTsM_1D[1,:] -fTsM_1D_mean_phi)/ fTsM_1D_std_phi

  fTsM_1D_std_z = np.std(fTsM_1D[2,:])
  fTsM_1D_mean_z = np.mean(fTsM_1D[2,:])
  fTsM_1D[2,:] = (fTsM_1D[2,:] -fTsM_1D_mean_z)/ fTsM_1D_std_z
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

  thr = int(.8*len(tTsM))
  thr2=-1
  tTsM = tTsM.astype(np.float32)

  pos_train, pos_test_val, grid_train, grid_test_val,  truth_train, truth_test_val = train_test_split(
      fTsM_1D, fTsM_3D, tTsM,
      test_size=0.2,
      random_state=4,
      stratify=tTsM)
  print(f"len(pos_train): {len(pos_train)}")
  print(f"len(pos_test_val): {len(pos_test_val)}")

  pos_val, pos_test,  grid_val, grid_test, truth_val, truth_test = train_test_split(
      pos_test_val, grid_test_val, truth_test_val,
      test_size = 0.25,
      random_state=5,
      stratify=truth_test_val)
  #print(f"1-sum(tTsM[:thr])/len(tTsM[:thr]): {1-sum(tTsM[:thr])/len(tTsM[:thr])}")

  #print(f"1-sum(tTsM[thr:thr2])/len(tTsM[thr:thr2]): {1-sum(tTsM[thr:thr2])/len(tTsM[thr:thr2])}")
  print(f"sum(truth_test_val)/len(truth_test_val): {sum(truth_test_val)/len(truth_test_val)}")
  print(f"sum(truth_train)/len(truth_train): {sum(truth_train)/len(truth_train)}")
  print(f"truth_test: {truth_test}")
  model2 = cnn_model2(input_3D_shape)
  model2.summary()
  history2 = model2.fit(
      [grid_train, pos_train], truth_train,
      validation_data=([grid_val, pos_val], truth_val),
      epochs= 50,
      batch_size= 32
      )
  '''
  model = cnn_model(input_3D_shape)
  model.summary()
  history2 = model.fit(
      grid_train, truth_train,
      validation_data=(grid_val, truth_val),
      epochs= 50,
      batch_size= 32
      )
  '''
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
