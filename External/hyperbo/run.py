# @title Imports and some utilities for plotting
import random
import time

from External.hyperbo.basics import definitions as defs
from External.hyperbo.basics import params_utils
from External.hyperbo.gp_utils import gp
from External.hyperbo.gp_utils import kernel
from External.hyperbo.gp_utils import mean
from External.hyperbo.gp_utils import utils
import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

font = {
    'family': 'serif',
    'weight': 'normal',
    'size': 7,
}
axes = {'titlesize': 7, 'labelsize': 7}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
SubDataset = defs.SubDataset


def plot_function_samples(
    mean_func,
    cov_func,
    params,
    warp_func=None,
    num_samples=1,
    random_seed=0,
    x_min=0,
    x_max=1,
):
  """Plot function samples from a 1-D Gaussian process.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in
      gp_utils/mean.py for more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in gp_utils/kernel.py for more details).
    params: GPParams, parameters for covariance, mean, and noise variance.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    num_samples: number of draws from the 1-D Gaussian process.
    random_seed: random seed for sampling.
    x_min: the min of the range of x.
    x_max: the max of the range of x.
  """
  key = jax.random.PRNGKey(random_seed)
  key, y_key = jax.random.split(key, 2)
  x = jnp.linspace(x_min, x_max, 100)[:, None]
  y = gp.sample_from_gp(
      y_key,
      mean_func,
      cov_func,
      params,
      x,
      warp_func=warp_func,
      num_samples=num_samples,
      method='svd',
  )
  fig = plt.figure(dpi=200, figsize=(2, 1))
  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('f(x)')


def plot_training_data(dataset, loss_function, num_samples=3, random_seed=0):
  """Plot datapoints from each (sampled) training function on a 1-D input space.

  Args:
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    loss_function: 'nll' or 'ekl'. If loss_function is 'nll', only plot generic
      training data. If 'ekl', only plot matching-input training data.
    num_samples: the number of sub-datasets to be plotted.
    random_seed: random seed used to sample sub-datasets for plotting.
  """
  print(f'Using the {loss_function} loss function.')
  def plot(dataset):
    random.shuffle(dataset)
    cnt = 0
    fig = plt.figure(dpi=200, figsize=(2, 1))
    for subdataset in dataset:
      if cnt == num_samples:
        break
      plt.scatter(subdataset.x, subdataset.y, s=1)
      cnt += 1
    plt.xlabel('x')
    plt.ylabel('f(x)')

  if loss_function == 'nll':
    dataset = [d for d in dataset.values() if d.aligned is None]
  elif loss_function == 'ekl':
    aligned_dataset = [d for d in dataset.values() if d.aligned is not None]
    dataset = []
    for subdataset in aligned_dataset:
      for i in range(subdataset.y.shape[1]):
        d = SubDataset(x=subdataset.x, y=subdataset.y[:, i:i+1])
        dataset.append(d)
  else:
    raise ValueError(f'{loss_function} is not a valid loss function.')
  print(f'dataset has {len(dataset)} training functions in total.')
  info = ''
  if num_samples >= len(dataset):
    num_samples = len(dataset)
  else:
    info = 'randomly sampled '
  print(f'Visualizing data from {num_samples} {info}training functions.')
  plot(dataset)

# @title Define a ground truth GP and generate training data
params = GPParams(
    model={
        'lengthscale': 0.1,
        'signal_variance': 10.0,
        'noise_variance': 1e-6,
        'constant': 5.0,
    }
)  # parameters of the GP


def ground_truth_mean_func(params, x, warp_func=None):
  return -jax.nn.relu(x - 0.5) * 20


mean_func = ground_truth_mean_func  # mean function of the GP
cov_func = kernel.matern52  # kernel (covariance) function of the GP

random_seed = 10  #@param{type: "number", isTemplate: true}
key = jax.random.PRNGKey(random_seed)
# number of training functions
num_train_functions = 10  #@param{type: "number", isTemplate: true}
# number of datapoints per training function
num_datapoints_per_train_function = 10  #@param{type: "number", isTemplate: true}

dataset = {}  # Training dataset
# Generate generic training data (only used by NLL)
for sub_dataset_id in range(num_train_functions):
  key, x_key, y_key = jax.random.split(key, 3)
  x = jax.random.uniform(x_key, (num_datapoints_per_train_function, 1))
  y = gp.sample_from_gp(y_key, mean_func, cov_func, params, x, method='svd')
  dataset[str(sub_dataset_id)] = SubDataset(x, y)
# Generate matching-input training data (only used by EKL)
key, x_key, y_key = jax.random.split(key, 3)
x = jax.random.uniform(x_key, (num_datapoints_per_train_function, 1))
y = gp.sample_from_gp(
    y_key,
    mean_func,
    cov_func,
    params,
    x,
    num_samples=num_train_functions,
    method='svd',
)
dataset['matching-input'] = SubDataset(x, y, aligned=1)

#@title Visualize function samples from the ground truth GP
random_seed = 0  #@param{type: "number", isTemplate: true}
num_samples = 10  #@param{type: "number", isTemplate: true}
plot_function_samples(mean_func,
                      cov_func,
                      params,
                      num_samples=num_samples,
                      random_seed=random_seed)

plt.show()
# @title Initialize a GP model to be pre-trained
optimization_method = 'lbfgs'  # @param ['lbfgs', 'adam']
loss_function = 'ekl'  # @param ['nll', 'ekl']
max_training_step = 1000  #@param{type: "number", isTemplate: true}

key = jax.random.PRNGKey(1)
params = GPParams(
    model={
        'lengthscale': jnp.array([.0]),
        'signal_variance': 0.0,
        'noise_variance': -6.,
    },
    config={
        'mlp_features': (8, 8),
        'Method': optimization_method,
        'max_training_step': max_training_step,
        'batch_size': 100,
        'objective': loss_function if loss_function == 'nll' else 'kl',
        'learning_rate': 1e-3,
    },
)
mean_func = mean.linear_mlp
cov_func = kernel.squared_exponential_mlp
warp_func = DEFAULT_WARP_FUNC

model = gp.GP(
    dataset=dataset,
    params=params,
    mean_func=mean_func,
    cov_func=cov_func,
    warp_func=warp_func,
)

key, subkey = jax.random.split(key, 2)
model.initialize_params(subkey)


#@title Visualize function samples from the initialized GP
random_seed = 3  #@param{type: "number", isTemplate: true}
num_samples = 5  #@param{type: "number", isTemplate: true}
plt.clf()
plot_function_samples(model.mean_func,
                      model.cov_func,
                      model.params,
                      num_samples=num_samples,
                      warp_func=warp_func,
                      random_seed=random_seed)

plt.show()

#@title Visualize training data
plt.clf()
random_seed = 3  #@param{type: "number", isTemplate: true}
num_samples = 5  #@param{type: "number", isTemplate: true}
plot_training_data(model.dataset,
                   loss_function,
                   num_samples=num_samples,
                   random_seed=random_seed)

plt.show()
#@title Pre-train the GP (you can run more than once to train for more steps)
print('Before pre-training.')
_ = model.stats()
start = time.time()
print('Pre-training..')
trained_params = model.train()
print(f'Pre-training time (s): {time.time() - start}')
print('After pre-training.')
_ = model.stats()



#@title Visualize function samples from the pre-trained GP
random_seed = 0  #@param{type: "number", isTemplate: true}
num_samples = 10  #@param{type: "number", isTemplate: true}
plot_function_samples(model.mean_func,
                      model.cov_func,
                      model.params,
                      num_samples=num_samples,
                      warp_func=warp_func,
                      random_seed=random_seed)



#@title Print pre-trained model parameters (only the interpretable ones)
lengthscale, signal_variance, noise_variance = params_utils.retrieve_params(model.params, ['lengthscale', 'signal_variance', 'noise_variance'], warp_func)
print(f'lengthscale={lengthscale} signal_variance={signal_variance} noise_variance={noise_variance}')