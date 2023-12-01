# NeuroRNNTools
PyTorch models for RNNs commonly used in computational neuroscience.

PyTorch nn modules and associated utilities for recurrent rate network models and spiking network models commonly used in computational neuroscience. 


# Rate network models.

## Quick start


A standard RNN can be created using
```
model = RateModel(N, Nx)
```
where `N` is the number of hidden units and `Nx` is the input size. This will create an RNN that is similar to a standard PyTorch 
RNN (`model = torch.nn.RNN(N,Nx)`) except `bias=False` by default in RateModel. A forward pass can then be computed by 
```
y = model(x)
```
where the input, `x`, should have shape `(batch_size, Nt, Nx)` and `Nt` is the number of time steps. Then `y` will have shape `(batch_size, Nt, N)` and it will be defined by the dynamics

$$
y_{n+1} = f(Jy_n + J_x x_n)
$$

where $x_n$ is the input on time step $n$, $J$ is the recurrent connectivity matrix, and $J_x$ is the input or ``read-in'' matrix. 

## Details

A model object is created by:

```
model = RateModel(recurrent, readin=None, readout=None, f='tanh', eta=1, rho_recurrent=1, rho_input=1, 
                  rho_output=1, bias_recurrent=False, bias_output=False, Network_Type='R')
```


There are two types of rate network models. The first is R-Type, which obey
 
$$
r_{n+1}=r_n+\eta(-r_n+f(Jr_n+J_x x_n+b))
$$

and the second is Z-Type, which obey

$$
z_{n+1}=z_n+\eta(-z_n+Jf(z_n)+J_x x_n+b)
$$

Here, $J$ is an $N\times N$ connectivity matrix and $N$ is the number of units (neurons) in the recurrent network. 

Note that taking $\eta=dt/\tau$ gives a forward Euler solver for the continuous ODEs

$$
\tau\frac{dr}{dt}=-r+f(Jr+J_x x+b)
$$

or

$$
\tau\frac{dz}{dt}=-z+Jf(z)+J_x x+b
$$

Taking $\eta=1$ for Type-R gives the standard RNN model more commonly used in machine learnin applications,

$$
r_{n+1}=f(Jr_n+X_n)
$$

We next describe the different arguments to `NeuroRNN`

### `recurrent`

`recurrent` determines the recurrent weight matrix. There are two options:

If `recurrent` is an integer then this integer is interpreted as the number of hidden dimensions. A weight matrix, $J$, is generated with normally distributed entries having standard deviation `rho_recurrent/sqrt(N)`

If `recurrent` is a square matrix (2-dim PyTorch tensor) then this is interpreted as the recurrent weight matrix and the number of hidden dimensions is inferred from its size.

### `readin`

`readin` determines the input/readin weight matrix. There are three options:

If `readin` is `None` (default) then there is no input layer. Effectively, $J_x=I$ is the identity matrix. In this case, the number of input dimensions must be equal to the number of hidden units. 

If `readin` is an integer then this integer is interpreted as the number of input dimensions, $N_x$. A weight matrix, $J$, is generated with normally distributed entries having standard deviation `rho_input/sqrt(Nx)`

If `readin` is a matrix (2-dim PyTorch tensor) then this is interpreted as the input weight matrix, $J_x$, and the number of input dimensions is inferred from its size. The shape should be `(N,Nx)` 

### readout

`readout` determines the output/readout matrix. Specifically, the output from a forward pass is defined by

$$
y_n = J_{out}r_n
$$

for R-Type networks and 

$$
y_n = J_{out}f(z_n)
$$

for Z-Type networks. 

There are three options:

If `readout` is `None` (default) then there is no readout applied. Effectively, $J_{out}=I$ so $y_n=r_n$ or $y_n=f(z_n)$. 

If `readout` is an integer then this integer is interpreted as the number of output dimensions, $N_{out}$. A weight matrix, $J_{out}$, is generated with normally distributed entries having standard deviation `rho_output/sqrt(N)`

If `readout` is a matrix (2-dim PyTorch tensor) then this is interpreted as the readout weight matrix, $J_{out}$,  and the number of output dimensions is inferred from its size. The shape should be `(Nout,N)` 


