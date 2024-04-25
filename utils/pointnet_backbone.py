import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from t_net import t_net
class pointnet_backbone(models.Model):
  def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
    '''
    Initializers
      num_points: number of points in point cloud
      num_global_feats: number of Global Features for the main Max Pooling Layer
      local_feat: if True, call() returns the concatenation of the locan and global features
    '''
    super(pointnet_backbone, self).__init__()
    self.num_points = num_points
    self.local_feat = local_feat

    # t_nets
    self.tnet1 = t_net(dim=3, num_points=num_points)
    self.tnet2 = t_net(dim=64, num_points=num_points)

    # Shared MLPs
    self.conv1 = layers.Conv1D(64, 1, activation='relu')
    self.conv2 = layers.Conv1D(64, 1, activation='relu')
    self.conv3 = layers.Conv1D(64, 1, activation='relu')
    self.conv4 = layers.Conv1D(128, 1, activation='relu')
    self.conv5 = layers.Conv1D(num_global_feats, 1, activation='relu')
    
    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.bn3 = layers.BatchNormalization()
    self.bn4 = layers.BatchNormalization()
    self.bn5 = layers.BatchNormalization()

  def call(self, inputs):
    bs = tf.shape(inputs)[0]

    # Pass through the firs t_net
    A_input = self.tnet1(inputs)
    
    # Perform first tranformation across each point in the batch
    x = tf.linalg.matmul(inputs, A_input)

    x = self.bn1(self.conv1(x))
    x = self.bn2(self.conv2(x))

    A_feat = self.tnet2(x)
    x = tf.linalg.matmul(x, A_feat)

    local_features = tf.identity(x)
    x = self.bn3(self.conv2(x))
    x = self.bn4(self.conv4(x))
    x = self.bn5(self.conv5(x))

    global_features = tf.reduce_max(x, axis=2) # Global max pooling

    if self.local_feat:
      global_features_expanded = tf.repeat(tf.expand_dims(global_features, -1), self.num_points, axis=2)
      combined_features = tf.concat([local_features, global_features_expanded], axis=-1)
      return combined_features
    else:
      return global_features



