# NeuroRNNTools
PyTorch models for RNNs commonly used in computational neuroscience.

PyTorch nn modules and associated utilities for recurrent rate network models and spiking network models commonly used in computational neuroscience. 


# Rate network model

## Quick reference

```
model = RateModel(recurrent, readin=None, readout=None, f='tanh', eta=1, rho_recurrent=1, rho_input=1, 
                  rho_output=1, bias_recurrent=False, bias_output=False, Network_Type='R')


y = model(x, Nt = None, return_time_series = True, initial_state = 'zero')
```



## Quick start: a basic model


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

For another example (simulating continuous time dynamics), see `RateModelExamples.ipynb`

## Model details

There are two types of rate network models. The first is R-Type, which obey
 
$$
r_{n+1}=r_n+\eta(-r_n+f(Jr_n+J_x x_n+b))
$$

with output from the network defined by

$$
y_n = J_{out}r_n+b_{out}
$$

and the second is Z-Type, which obey

$$
z_{n+1}=z_n+\eta(-z_n+Jf(z_n)+J_x x_n+b)
$$

with output from the network defined by

$$
y_n = J_{out}f(z_n)+b_{out}
$$

Here, $J$ is an $N\times N$ connectivity matrix and $N$ is the number of units (neurons) in the recurrent network. 

Under default settings, $J_{out}$ and $J_x$ are identity matrices and $b=b_{out}=0$ so $y_n=r_n$ and $J_x x_n+b=x_n$. 

Note that taking $\eta=dt/\tau$ gives a forward Euler solver for the continuous ODEs

$$
\tau\frac{dr}{dt}=-r+f(Jr+J_x x+b)
$$

or

$$
\tau\frac{dz}{dt}=-z+Jf(z)+J_x x+b
$$

Taking $\eta=1$ for Type-R gives the standard PyTorch type of RNN model commonly used in machine learning applications,

$$
r_{n+1}=f(Jr_n+J_x x_n+b_n)
$$

## Creating a new model object

A model object is created by:

```
model = RateModel(recurrent, readin=None, readout=None, f='tanh', eta=1, rho_recurrent=1, rho_input=1, 
                  rho_output=1, bias_recurrent=False, bias_output=False, Network_Type='R')
```




We next describe the different arguments to `NeuroRNN`

### `recurrent`

`recurrent` determines the recurrent weight matrix. There are two options:

If `recurrent` is an integer then this integer is interpreted as the number of hidden dimensions. A weight matrix, $J$, is generated with normally distributed entries having standard deviation `rho_recurrent/sqrt(N)`

If `recurrent` is a square matrix (2-dim PyTorch tensor) then this is interpreted as the recurrent weight matrix and the number of hidden dimensions is inferred from its size.

### `readin`

`readin` determines the input/readin weight matrix. There are three options:

If `readin` is `None` (default) then there is no input layer. Effectively, $J_x=I$ is the identity matrix. In this case, the number of input dimensions must be equal to the number of hidden units ($N_x=N$). 

If `readin` is an integer then this integer is interpreted as the number of input dimensions, $N_x$. A weight matrix, $J$, is generated with normally distributed entries having standard deviation `rho_input/sqrt(Nx)`

If `readin` is a matrix (2-dim PyTorch tensor) then this is interpreted as the input weight matrix, $J_x$, and the number of input dimensions is inferred from its size. The shape should be `(N,Nx)` 

### `readout`

`readout` determines the output/readout matrix. Specifically, the output from a forward pass is defined by

$$
y_n = J_{out}r_n+b_{out}
$$

for R-Type networks and 

$$
y_n = J_{out}f(z_n)+b_{out}
$$

for Z-Type networks. Here $b_{out}$ is a readout bias (see below)

There are three options:

If `readout` is `None` (default) then there is no readout applied. Effectively, $J_{out}=I$ so $y_n=r_n$ or $y_n=f(z_n)$. Then the number of output dimensions is equal to the number of hidden units, $N_{out}=N$.

If `readout` is an integer then this integer is interpreted as the number of output dimensions, $N_{out}$. A weight matrix, $J_{out}$, is generated with normally distributed entries having standard deviation `rho_output/sqrt(N)`

If `readout` is a matrix (2-dim PyTorch tensor) then this is interpreted as the readout weight matrix, $J_{out}$,  and the number of output dimensions is inferred from its size. The shape should be `(Nout,N)` 

### `f`

`f` is the activation function or f-I curve used for the network. There are two options:

If `f` is one of the following strings: 'relu', 'tanh', or 'id' then the corresponding activation function is used (where 'id' is the identity)

If `f` is a function then this function is used. 

### `rho_recurrent`, `rho_input`, and `rho_output`

These parameters determine the standard deviation of the weight matrices when weight matrices are generated during initialization (see description of `recurrent`, `readin`, and `readout` above). 

If `readin` is a matrix or is None, then `rho_input`. Similarly for recurrent and output. 

### `bias_recurrent` and `bias_output`

These determine whether a bias is used in the recurrent layer and the output layer. Note that bias in the recurrent layer is equivalent to bias in the input layer, so there is no `bias_input`.

If `bias_recurrent=False` (default) then no bias is used ($b=0$). Otherwise, a bias is used and it is initialized using the default initialization of PyTorch linear layers. Similarly for `bias_output`. 

### `Network_Type`

If `Network_Type=='R'` then an R-Type network is used. If `Network_Type=='Z'` then a Z-type network is used. 


## Forward pass

A forward pass is completed by calling:
```
y = model(x, Nt = None, return_time_series = True, initial_state = 'zero')
```

### `x`

`x` is the input to the network. It should be a 3-dimensional or 2-dimensional tensor.

If `x` is 3-dimensional then it is interpreted as the input $x_n$ to the network with shape `(batch_size, Nt, Nx)` where `Nt` is the number of time steps.

If `x` is 2-dimensional then it is interpreted as a time-constant input $x_n=x$ and therefore has shape `(batch_size, Nx)` and the number of timesteps must be passed in as `Nt`

### `Nt`

An integer representing the  number of time steps if `x` is 2-dimensional. If `x` is 3-dimensional, then `Nt` should be `None`.

### `return_time_series`

A flag determining whether a time series is returned from the forward pass or just the last state of the network is returned. 

If `return_time_series==True` then `y` has shape `(batch_size, Nt, Nout)`

If `return_time_series==False` then `y` has shape `(batch_size, Nout)` because only the final state is returned.

### `initial_state`

Determines the initial hidden state. There are three options:

If `initial_state=='zero'` then the hidden state is initialized to zeros.

If `initial_state=='keep'` then the hidden state from the last time step of the previous forward pass is kept. If this is the first forward pass with this model, then a warning will appear and the hidden state will be initialized to zeros.

If `initial_state` is a 2-dimensional array then it this array is used as the initial state. Therefore, it must have shape `(batch_size, N)`

### `y`

The output from the network. 

If `return_time_series==True`, this will be a 3-dimensional array with shape `(batch_size, Nt, Nout)` representing the time series of the output.

If `return_time_series==True`, this will be a 2-dimensional array with shape `(batch_size, Nout)` representing the final state of the output.

# Spiking network model

