import numpy as np
import awkward as ak
import tensorflow as tf
from tensorflow.keras import layers, models


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

def load_fromParquet(filename):
  return ak.from_parquet(filename)

def main():
  target_file='tTsM.parquet'
  features_1D_file ='fTsM_1D.parquet'
  features_3D_file ='grid_3D.parquet'
  tTsM = load_fromParquet(target_file)
  fTsM_1D = load_fromParquet(features_1D_file)
  fTsM_3D = load_fromParquet(features_3D_file)
  print(tTsM.type)
  print(fTsM_1D.type)
  print(fTsM_3D.type)
  tTsM = np.asarray(tTsM)
  fTsM_1D = np.asarray(fTsM_1D)
  fTsM_3D = np.asarray(fTsM_3D)
  fTsM_3D = np.asarray(fTsM_3D)
  print(fTsM_3D[0,0,:,:,:])
  A = fTsM_3D[0,0,:,:,:]
  print(A.shape)
  fTsM_3D = np.transpose(fTsM_3D, [0,2,3,4,1])
  print(fTsM_3D[0,:,:,:,0])
  B = fTsM_3D[0,:,:,:,0]
  print(B.shape)
  print(np.all(A==B))
  exit()

  print(tTsM.shape)
  print(fTsM_1D.shape)
  print(fTsM_3D.shape)
  
  input_3D_shape = (8, 8, 8,2)
  #input_3D_shape = (2, 8, 8, 8)
  
  # Create the model
  model = cnn_model(input_3D_shape)
  
  model.summary()
  history = model.fit(
    fTsM_3D, tTsM,
    validation_data=(fTsM_3D, tTsM),
    epochs=10,
    batch_size=32
  )
  

if __name__ == '__main__':
  main()
