import numpy as np
import awkward as ak
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def Gauss(x, A, B):
  return A*np.exp(-1*B*x**2)

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
  h1, bins1 = np.histogram(d1, bins=100)
  x1 = (bins1[1:] + bins1[:-1])/2
  p1, cov1 = curve_fit(Gauss, x1, h1)
  fit_y = Gauss(x1, p1[0], p1[1])
  myhist(X, title=title, xlabel=xlabel, ylabel=ylabel, bins=bins, range=range, label=label)
  plt.plot(x1, fit_y, '-', label='fit', color='red')
  plt.legend()

def plot_vars(deltaTsM_1D, fTsM_1D, fTsM_g3D, tTsM, suf):
  myhistWithGauss(ak.flatten(deltaTsM_1D[:,0], axis=None), title="Delta_eta", xlabel="eta-eta_mean for TsM", ylabel="Counts/bin", bins=45, range=(-.2,.2), label=suf)
  plt.savefig(f"plots/{suf}_val_delta_eta_withGauss.png")
  plt.clf()
  myhistWithGauss(ak.flatten(deltaTsM_1D[:,1], axis=None), title="Delta_phi", xlabel="phi-phi_mean for TsM", ylabel="Counts/bin", bins=45, range=(-.2,.2), label=suf)
  plt.savefig(f"plots/{suf}_val_delta_phi_withGauss.png")
  plt.clf()
  myhistWithGauss(ak.flatten(deltaTsM_1D[:,2], axis=None), title="Delta_z", xlabel="z-z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
  plt.savefig(f"plots/{suf}_val_delta_z_withGauss.png")
  plt.clf()

  myhist(fTsM_1D[:,0], title="mean_eta", xlabel="eta_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
  plt.savefig(f"plots/{suf}_val_mean_eta.png")
  plt.clf()
  myhist(fTsM_1D[:,1], title="mean_phi", xlabel="phi_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
  plt.savefig(f"plots/{suf}_val_mean_phi.png")
  plt.clf()
  myhist(fTsM_1D[:,2], title="mean_z", xlabel="z_mean for TsM", ylabel="Counts/bin", bins=45, label=suf)
  plt.savefig(f"plots/{suf}_val_mean_z.png")
  plt.clf()
  myhist(np.array(tTsM, dtype=np.int32), title="Truth: en>en_min and score < score_max", xlabel="Passed", ylabel="Counts/bin", bins=45, label=suf)
  plt.savefig(f"plots/{suf}_val_delta_truth.png")
  plt.clf()

def load_fromParquet(filename):
  return ak.from_parquet(filename)


def main():
  prefix = "SinglePi"
  target_file=f"data/{prefix}_tTsM.parquet"
  delta_file = f"data/{prefix}_deltaTsM_1D.parquet"
  features_1D_file =f"data/{prefix}_fTsM_1D.parquet"
  features_3D_file =f"data/{prefix}_grid_3D.parquet"
  tTsM = load_fromParquet(target_file)
  deltaTsM_1D = load_fromParquet(delta_file)
  fTsM_1D = load_fromParquet(features_1D_file)
  fTsM_3D = load_fromParquet(features_3D_file)
  tTsM = np.asarray(tTsM)
  print(deltaTsM_1D.type)
  #deltaTsM_1D = np.asarray(deltaTsM_1D)
  ##print(deltaTsM_1D.type)

  fTsM_1D = np.asarray(fTsM_1D)
  fTsM_3D = np.asarray(fTsM_3D)
  fTsM_3D = np.asarray(fTsM_3D)
  fTsM_3D = np.transpose(fTsM_3D, [0,2,3,4,1])
  
  plot_vars(deltaTsM_1D, fTsM_1D, fTsM_3D, tTsM, prefix)
  


  

if __name__ == '__main__':
  main()