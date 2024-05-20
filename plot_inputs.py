
import numpy as np
import awkward as ak
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
'''
def Gauss(x, A, B):
  return A*np.exp(-1*B*x**2)
'''
def plot_voxels(vox_in, prefix):
    print("####################")
    print("if this does not work is probably because we don't have enough TICLCandidate passing the energy cut")
    print("####################")
    fig = plt.figure(figsize=(10,10), dpi=200)
    for sp in range(1,5,1):
      ax = fig.add_subplot(2,2,sp, projection='3d')
      #ax = fig.gca( projection='3d')
      vox = vox_in[sp]
      ax.voxels(vox, shade=True, alpha=0.45)
    plt.savefig(f"plots/{prefix}_voxels.png")
    plt.clf()
    projection_x = np.sum(vox_in, axis=(0,2,3))
    plt.plot(projection_x, 'bo-')
    plt.xlabel('x')
    plt.ylabel('Occupancy')
    plt.grid(True)
    plt.savefig(f"plots/{prefix}_voxels_x_projection.png")
    plt.clf()
    projection_y = np.sum(vox_in, axis=(0,1,3))
    plt.plot(projection_y, 'bo-')
    plt.xlabel('y')
    plt.ylabel('Occupancy')
    plt.grid(True)
    plt.savefig(f"plots/{prefix}_voxels_y_projection.png")
    plt.clf()
    projection_z = np.sum(vox_in, axis=(0,1,2))
    plt.plot(projection_z, 'bo-')
    plt.xlabel('z')
    plt.ylabel('Occupancy')
    plt.grid(True)
    plt.savefig(f"plots/{prefix}_voxels_z_projection.png")
    plt.clf()

def Gauss(x, a, x0, sigma): 
      return a*np.exp(-(x-x0)**2/(2*sigma**2)) 

def myhist(X, bins=30, title='title', xlabel='time (ns)', ylabel='Counts / bin', color='dodgerblue', alpha=1, fill='stepfilled', range=None, label="data"):
  #plt.figure(dpi=100)
  if range==None:
    plt.hist(np.array(X), bins=bins, color=color, alpha=alpha, histtype=fill, label=label)
  else:
    plt.hist(np.array(X), bins=bins, color=color, alpha=alpha, histtype=fill, range=range, label=label)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid()

def myhistWithGauss(X, bins=30, title='title', xlabel='time (ns)', ylabel='Counts / bin', color='dodgerblue', alpha=1, fill='stepfilled', range=None, label="data"):
  d1 = np.array(X)
  print(d1)
  h1, bins1 = np.histogram(d1, bins=bins, range=range)
  norm = sum(h1)
  #h1 = h1/norm
  x1 = (bins1[1:] + bins1[:-1])/2
  p1, cov1 = curve_fit(Gauss, x1, h1)
  fit_y = Gauss(x1, p1[0], p1[1], p1[2])
  myhist(X, title=title, xlabel=xlabel, ylabel=ylabel, bins=bins, range=range, label=label)
  plt.plot(x1, fit_y, '-', label='fit', color='red')
  plt.legend()
  print(f"p1 fot {title}: {p1}")

def plot_vars(deltaTsM_1D, fTsM_1D, tTsM, fTsM_score, prefix):
  print(f"fTsM_1D[0,:]: {fTsM_1D[0,:]}")
  myhist(fTsM_1D[:, 0], title="mean_x", xlabel="x_mean for TsM", ylabel="Counts/bin", bins=100, label=prefix)
  plt.savefig(f"plots/{prefix}_val_mean_eta.png")
  plt.clf()
  myhist(fTsM_1D[:, 1], title="mean_y", xlabel="y_mean for TsM", ylabel="Counts/bins", bins=100,  label=prefix)
  plt.savefig(f"plots/{prefix}_val_mean_phi.png")
  plt.clf()
  myhist(fTsM_1D[:, 2], title="mean_z", xlabel="z_mean for TsM", ylabel="Counts/bin", bins=45,  label=prefix)
  plt.savefig(f"plots/{prefix}_val_mean_z.png")
  plt.clf()
  myhist(np.array(tTsM, dtype=np.int32), title="Truth: en>en_min and score < score_max", xlabel="Passed", ylabel="Counts/bin", bins=30, label=prefix)
  plt.savefig(f"plots/{prefix}_val_truth.png")
  plt.clf()
  print(f"fTsM_score: {fTsM_score}")
  myhist(np.array(fTsM_score), title="simToRecos_score", xlabel="score", ylabel="Counts/bin", bins=30, label=prefix)
  plt.savefig(f"plots/{prefix}_val_score.png")
  plt.clf()

  myhistWithGauss(ak.flatten(deltaTsM_1D[:,:,0], axis=None), title="Delta_x", xlabel="x-x_mean for TsM", ylabel="Counts/bin", bins=100, range=(-10,10), label=prefix)
  plt.savefig(f"plots/{prefix}_val_delta_x_withGauss.png")
  plt.clf()
  myhistWithGauss(ak.flatten(deltaTsM_1D[:,:,1], axis=None), title="Delta_y", xlabel="y-y_mean for TsM", ylabel="Counts/bin", bins=100, range=(-10,10), label=prefix)
  plt.savefig(f"plots/{prefix}_val_delta_y_withGauss.png")
  plt.clf()
  myhistWithGauss(ak.flatten(deltaTsM_1D[:,:,2], axis=None), title="Delta_z", xlabel="z-z_mean for TsM", ylabel="Counts/bin", bins=300, range=(-40,40),label=prefix)
  plt.savefig(f"plots/{prefix}_val_delta_z_withGauss.png")
  plt.clf()


def load_fromParquet(filename):
  return ak.from_parquet(filename)

def main():
  #prefix = "SinglePi"
  #prefix = "4Photons_0PU"
  prefix = "4Pions_0PU"
  target_file=f"data/{prefix}_tTsM.parquet"
  score_file=f"data/{prefix}_s2r_score.parquet"
  delta_file = f"data/{prefix}_deltaTsM_1D_cls.parquet"
  features_1D_file =f"data/{prefix}_fTsM_1D_cls.parquet"
  features_3D_file =f"data/{prefix}_grid_3D_cls.parquet"

  tTsM = load_fromParquet(target_file)
  fTsM_score = load_fromParquet(score_file)
  deltaTsM_1D = load_fromParquet(delta_file)
  fTsM_1D = load_fromParquet(features_1D_file)
  fTsM_3D = load_fromParquet(features_3D_file)
  tTsM = np.asarray(tTsM)
  print(deltaTsM_1D.type)
  #deltaTsM_1D = np.asarray(deltaTsM_1D)
  ##print(deltaTsM_1D.type)

  fTsM_1D = np.asarray(fTsM_1D)
  tTsM = np.asarray(tTsM)
  #fTsM_3D = np.asarray(fTsM_3D)
  #fTsM_3D = np.transpose(fTsM_3D, [0,2,3,4,1])
  
  vox_in = (np.array(fTsM_3D)[tTsM,0,:,:,:] > 0).astype(np.int32)
  plot_voxels(vox_in, prefix)
  #fTsM_3D = np.transpose(fTsM_3D, [0,2,3,4,1])
  plot_vars(deltaTsM_1D, fTsM_1D, tTsM, fTsM_score, prefix)
  


  

if __name__ == '__main__':
  main()
