import tensorflow as tf
from t_net import t_net
from pointnet_classification import pointnet_classification
from pointnet_segmentation import pointnet_segmentation
from pointnet_backbone import pointnet_backbone

def main():
  num_points = 2500
  test_data = tf.random.normal([32,num_points,3])

  # Testing T-net
  tnet = t_net(dim=3)
  transform = tnet(test_data)
  print(f"T-net output shape: {transform.shape}")

  # Testing pointnet_backbone
  backbone = pointnet_backbone(local_feat=False)
  global_features = backbone(test_data)
  print(f"Global Features shape: {global_features.shape}")

  backbone = pointnet_backbone(local_feat=True)
  combined_features = backbone(test_data)
  print(f"Combined Features shape: {combined_features.shape}")


  # Testing Classification Head
  classifier = pointnet_classification(k=1)
  class_output = classifier(test_data)
  print(f"Class output shape: {class_output.shape}")

  # Testing Segmentation Head
  segmenter = pointnet_segmentation(m=3)
  seg_output = segmenter(test_data)
  print(f"Segmentationtion output shape: {seg_output.shape}")


if __name__ == '__main__':
  main()
