import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers

class voxnet_cnn(models.Model):
  def __init__(self):
    super(voxnet_cnn, self).__init__()

    # 3D convolutional
    self.cnv1 = layers.Conv3D(16, (3,3,3), activation="relu")
    self.cnv2 = layers.Conv3D(32, (3,3,3), activation="relu")
    self.cnv3 = layers.Conv3D(64, (3,3,3), activation="relu")

    # Batch normalization
    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.bn3 = layers.BatchNormalization()

    # Max Pooling 3D
    self.mp1 = layers.MaxPooling3D((2,2,2))
    self.mp2 = layers.MaxPooling3D((2,2,2))
    self.mp3 = layers.MaxPooling3D((2,2,2))

    # Dropout
    self.dropout1 = layers.Dropout(0.25)
    self.dropout2 = layers.Dropout(0.25)
    self.dropout3 = layers.Dropout(0.25)
    
    # Flatten
    self.flt = layers.Flatten()

  def call(self, inputs):
    x = self.dropout1(self.mp1(self.bn1(self.conv1(inputs))))
    x = self.dropout2(self.mp2(self.bn2(self.conv2(x))))
    x = self.dropout3(self.mp3(self.bn3(self.conv3(x))))
    x = self.flt(x)

    return x

class voxnet_ffn(models.Model):
  def __init__(self):
    super(voxnet_ffn, self).__init__()

    # TODO: Add more inputs like timing and make this part deeper.
    # Dense
    self.dn = layers.Dense(32, activation="relu")

  def call(self, inputs):
    x = self.dn(inputs)

    return x


class voxnet(models.Model):
  def __init__(self, input_shape):
    super(voxnet, self).__init__()
    # Inputs

    # voxnet parts
    self.voxcnn = voxel_cnn)
    self.voxffn = voxel_ffn()

    # Dense
    self.dn1 = layers.Dense(32, activation="relu")
    self.dn2 = layers.Dense(1, activation="sigmoid")

    # Dropout
    self.dropout = layers.Dropout(0.25)

  def call(sefl, [voxel_input, vector_input]):
    #self.voxel_input = Input(shape = input_shape, name="voxel_grid")
    #self.vector_input = Input(shape = input_shape, name="voxel_vector")
    # Get cnn for grid information.
    features_vox = self.voxccn(voxel_input)
    # Get ffn for vector information for every trackster.
    features_bc = self.voxffn(vector_input)

    # combine voxel and baricenter info
    x = layers.concatenate([features_vox, features_bc])
    x = self.dn1(x)
    x = self.dn2(x)

    return x








