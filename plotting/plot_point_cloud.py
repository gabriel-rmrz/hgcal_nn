#import pandas as pd
#import ellipsis as el
import matplotlib.pyplot as plt
import numpy as np


def plot_point_cloud(X, Y, Z, ax1, ax2, ax3, ax4, s = 1, color='blue', alpha=1, prefix=""):
  '''
  Creates 3D plots of a point cloud from different perspectives
  '''
  #geom = o3d.geometry.PointCloud()
  #geom.points = o3d.utility.Vector3dVector(P)

  #viewer = o3d.visualization.Visualizer()
  #viewer.create_window()
  #viewer.add_geometry(geom)
  #img = viewer.capture_screen_float_buffer(True)
  #plt.imshow(np.asarray(img))
  # Perspective view
  ax1.scatter(X, Y, Z, color=color, marker='.', s=s, alpha=alpha)
  ax1.set_title('Perspective View')
  ax1.axis('equal')

  #Top view
  ax2.scatter(X, Y, color=color, marker='.', s=s, alpha=alpha)
  ax2.set_title('Top View')
  ax2.set_xlabel('X-axis')
  ax2.set_ylabel('Y-axis')
  ax2.axis('equal')

  #Side view
  ax3.scatter(X, Z, color=color, marker='.', s=s, alpha=alpha)
  ax3.set_title('Side View')
  ax3.set_xlabel('X-axis')
  ax3.set_ylabel('Z-axis')
  ax3.axis('equal')

  #From view
  ax4.scatter(Y, Z, color=color, marker='.', s=s, alpha=alpha)
  ax4.set_title('Front View')
  ax4.set_xlabel('Y-axis')
  ax4.set_ylabel('Z-axis')
  ax4.axis('equal')


  #viewer.destroy_window()
