import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from pointnet_backbone import pointnet_backbone

class pointnet_segmentation(models.Model):
  def __init__(self, num_points=2500, num_global_feats=1024, m=2):
    super(pointnet_segmentation, self).__init__()

    self.num_point = num_points
    self.m = m

    # Get the backbone
    self.backbone = pointnet_backbone(num_points, num_global_feats, local_feat=True)

    # Shared MLP
    num_features = num_global_feats + 64 # local and global features
    self.conv1 = layers.Conv1D(512, 1, activation='relu')
    self.conv2 = layers.Conv1D(256, 1, activation='relu')
    self.conv3 = layers.Conv1D(128, 1, activation='relu')
    self.conv4 = layers.Conv1D(m, 1)


    # Batch normalizaiton
    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.bn3 = layers.BatchNormalization()

  def call(self, inputs):
    # Get combine features from the backbone
    combined_features = self.backbone(inputs)

    x = self.bn1(self.conv1(combined_features))
    x = self.bn2(self.conv2(x))
    x = self.bn3(self.conv3(x))
    x = self.conv4(x)

    # Rechape output for segmentation task (batch_size, points, classes)
    x = tf.transpose(x, [0, 2, 1])
    return x
