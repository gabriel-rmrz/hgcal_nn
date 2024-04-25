import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from pointnet_backbone import pointnet_backbone


class pointnet_classification(models.Model):
  def __init__(self, num_points=2500, num_global_feats=1024, k=2):
    super(pointnet_classification, self).__init__()

    # Get the backbone
    self.backbone = pointnet_backbone(num_points, num_global_feats, local_feat=False)

    # MLP for classification
    self.fc1 = layers.Dense(512, activation='relu')
    self.fc2 = layers.Dense(256, activation='relu')
    self.fc3 = layers.Dense(k)

    # Batch normalization
    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()


    # Dropout
    self.dropout = layers.Dropout(0.3)

  def call(self, inputs):
    # Get global features from the backbone
    global_features = self.backbone(inputs)

    x = self.bn1(self.fc1(global_features))
    x = self.bn2(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)

    return x



