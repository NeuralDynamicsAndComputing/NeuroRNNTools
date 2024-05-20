import numpy as np
import torch

# Right now, all of these functions use the old NumPy versions
# and then convert to torch tensors. This is not ideal.
# TODO: Write pure PyTorch versions.


# Function to generate blockwise ER connection matrix
# NsPre = tuple of ints containing number of pre neurons in each block
# Jm = matrix connection weights in each block
# P = matrix of connection probs in each block
# NsPost = number of post neurons in each block
# If NsPost == None, connectivity is assumed recurrent (so NsPre=NsPost)
def GetBlockErdosRenyi(NsPre,Jm,P,NsPost=None):

  # Convert tensors to numpy arrays.
  # Get rid of this after changing to PyTorch version
  if torch.is_tensor(NsPre):
      NsPre=NsPre.numpy()
  if torch.is_tensor(Jm):
      Jm=Jm.numpy()
  if torch.is_tensor(P):
      P=P.numpy()
  if torch.is_tensor(NsPost):
      NsPost=NsPost.numpy()

  if NsPost==None:
    NsPost=NsPre

  # # If Jm is a 1D array, reshape it to column vector
  # if len(Jm.shape)==1:
  #   Jm = np.array([Jm]).T
  # if len(P.shape)==1:
  #   P = np.array([P]).T

  Npre = int(np.sum(NsPre))
  Npost = int(np.sum(NsPost))
  cNsPre = np.cumsum(np.insert(NsPre,0,0)).astype(int)
  cNsPost = np.cumsum(np.insert(NsPost,0,0)).astype(int)
  J = np.zeros((Npost,Npre), dtype = np.float32)

  for j1,N1 in enumerate(NsPost):
    for j2,N2 in enumerate(NsPre):
      J[cNsPost[j1]:cNsPost[j1+1],cNsPre[j2]:cNsPre[j2+1]]=Jm[j1,j2]*(np.random.binomial(1, P[j1,j2], size=(N1, N2)))
  J = torch.tensor(J)
  return J



# Create a smooth Gaussian process by convolving
# white noise with a Gaussian kernel.
# Noise will have variance=1
def MakeSmoothGaussianProcess(taux, Nt, dt, N=1, device='cpu'):

  import torch.nn.functional as F
  pi0 = 3.1415927410125732

  # Make kernel
  taus = torch.range(-4 * taux, 4 * taux, dt).to(device)
  K = (1 / (taux * np.sqrt(2 * pi0))) * torch.exp(-taus ** 2 / (2 * taux ** 2))
  K = K / (dt * K.sum())

  if N==1:
    white_noise = (1/np.sqrt(dt))*torch.randn(Nt).to(device)
    X = F.conv1d(white_noise, K, padding='same')*dt
  else:
    K = K[None,None,:]
    # Interpret white_noise=temp as N batches and 1 channel.
    # This lets us apply the same kernel to all "channels"
    # because channels are interpreted as batches
    white_noise=(1/np.sqrt(dt))*torch.randn(N, 1, Nt).to(device)

    X = torch.squeeze(F.conv1d(white_noise, K, padding='same')*dt)

  return X


# Generate Poisson process with batch dimension too
def PoissonProcess(r,dt,batch_size=1,N=1,T=None,rep='sparse'):

  if torch.is_tensor(r):
    r=r.numpy()



  # If r is a 2D array, then there are multiple rates
  # and they are inhomogeneous in time.
  # r is interpreted as (time)x(neuron)
  # ie, r[j,k] is the rate of neuron k at time index j.
  # If r is a 1D array, then each neuron has its own rate,
  # but rates are constant in time.
  # If r is a scalar, then all neurons have the same constant rate.
  # If rep=='full' then s has the same shape as r.
  # If rep=='sparse' then s is 2xNumSpikes

  # First generate dense s
  if np.ndim(r)==0:
    Nt = int(T / dt)
    s = np.random.binomial(1,r*dt,(batch_size,Nt,N))/dt
  else:
    raise Exception('Not implemented yet.')

  if rep == 'sparse':
    [I, J] = np.nonzero(s)
    temp = np.zeros((2, len(I)))
    temp[0, :] = J * dt
    temp[1, :] = I
    s = temp
    temp = np.argsort(s[0, :])
    s = s[:, temp]

  return torch.tensor(s, dtype=torch.float32)

# Convolve signals with an exponential kernel
def ConvWithExp(x,tau,dt):
  if len(x.shape)!=3:
    raise Exception('x should be 3dim with shape (batch_size,Nt,N)')
  Nt = x.shape[1]
  batch_size = x.shape[0]
  N = x.shape[2]
  KernelTime = torch.arange(int(tau * 6), -int(tau * 6) - dt, -dt).to(x.device)
  ExpKernel = torch.zeros(1, 1, len(KernelTime)).to(x.device)
  ExpKernel[0, 0, :] = (1 / tau) * torch.exp(-KernelTime / tau) * (KernelTime >= 0)
  Y = torch.nn.functional.conv1d(x.permute(0, 2, 1).reshape(batch_size * N, 1, Nt), ExpKernel, padding='same') * dt
  Y = Y.reshape(batch_size, N, Nt).permute(0, 2, 1)
  return Y


# Returns 2D array of spike counts from sparse spike train, s.
# Counts spikes over window size winsize.
# h is represented as (neuron)x(time)
# so h[j,k] is the spike count of neuron j at time window k
def GetSpikeCounts(s,winsize,N,T):

  if torch.is_tensor(s):
      s=s.numpy()

  xedges=np.arange(0,N+1,1)
  yedges=np.arange(0,T+winsize,winsize)
  h,_,_=np.histogram2d(s[1,:],s[0,:],bins=[xedges,yedges])
  return h

# Convert a tensory to numpy array for plotting, etc
def ToNP(x):
  return x.detach().cpu().numpy()

# # Returns a resampled version of x
# # with a different dt.
# def DumbDownsample(x,dt_old,dt_new):
#   n = int(dt_new/dt_old)
#   if n<=1:
#     print('New dt should be larger than old dt. Returning x.')
#     return x
#   return x.reshape()
#
#   def unit_vector(vector):
#     return vector / np.linalg.norm(vector)


def GetOneAngle(v1, v2):
  if torch.is_tensor(v1):
      v1=v1.numpy()
  if torch.is_tensor(v2):
      v2=v2.numpy()

  return (180.0/np.pi)*np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)),-1.0,1.0))


