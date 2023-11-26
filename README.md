# NeuroRNNTools
PyTorch models for RNNs commonly used in computational neuroscience.

PyTorch nn modules and associated utilities for recurrent rate network models and spiking network models commonly used in computational neuroscience. 

## Rate network models.

There are two types of rate network models. The first is R-Type, which obey
 
$$
r_{n+1}=r_n+\eta(-r_n+f(Jr_n+X_n))
$$
 
and the second is Z-Type, which obey

$$
z_{n+1}=z_n+\eta(-z_n+Jf(z_n)+X_n)
$$

Here, $J$ is an $N\times N$ connectivity matrix and $N$ is the number of units (neurons) in the recurrent network. 

Note that taking $\eta=dt/\tau$ gives a forward Euler solver for the ODEs

$$
\tau\frac{dr}{dt}=-r+f(Jr+X)
$$

or

$$
\tau\frac{dz}{dt}=-z+Jf(z)+X
$$

Taking $\eta=1$ for Type-R gives the standard RNN model more commonly used in machine learnin applications,

$$
r_{n+1}=f(Jr_n+X_n)
$$

A network module object is created using:

```
model = RateModel(recurrent, readin=None, readout=None, f='tanh', eta=1, rho_recurrent=1, rho_input=1, 
                  rho_output=1, bias_recurrent=False, bias_output=False, Network_Type='R')
```

