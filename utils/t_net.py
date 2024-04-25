import tensorflow as tf
from tensorflow.keras import layers, models, initializers

class t_net(models.Model):
  def __init__(self, dim, num_points=2500):
    super(t_net, self).__init__()
    # Define the network layers

    self.dim = dim
    self

    self.conv1 = layers.Conv1D(64, 1, activation='relu')
    self.bn1 = layers.BatchNormalization()

    self.conv2 = layers.Conv1D(128, 1, activation='relu')
    self.bn2 = layers.BatchNormalization()

    self.conv3 = layers.Conv1D(1024, 1, activation='relu')
    self.bn3 = layers.BatchNormalization()

    # Fully connected layers
    self.fc1 = layers.Dense(512, activation='relu')
    self.bn4 = layers.BatchNormalization()

    self.fc2 = layers.Dense(256, activation='relu')
    self.bn5 = layers.BatchNormalization()

    self.fc3 = layers.Dense(dim*dim)

    # Max pooling layer
    self.max_pool = layers.GlobalMaxPooling1D()

  def call(self, inputs):
    # Conv1D layers
    x = self.conv1(inputs)
    x = self.bn1(x)

    x = self.conv2(x)
    x = self.bn2(x)

    x = self.conv3(x)
    x = self.bn3(x)

    # Max pooling
    x = self.max_pool(x)
    x = tf.reshape(x, (-1, 1024))

    # Fully connected layers
    x = self.fc1(x)
    x = self.bn4(x)

    x = self.fc2(x)
    x = self.bn5(x)

    x = self.fc3(x)

    # Reshape output into transformation matrix
    x = tf.reshape(x, (-1, self.dim, self.dim))
    
    # Initialize identity matrix and add
    batch_size = tf.shape(inputs)[0]
    iden = tf.eye(self.dim, batch_shape=[batch_size])

    return x

