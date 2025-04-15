import numpy as np
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import fminbound
from Bench.Problems import Problems
from scipy.optimize import OptimizeResult
from tqdm import trange
import sobol_seq
import os



# EVOLUTIONARY OPERATORS
def sbx_crossover(p1, p2, sbxdi):
  D = p1.shape[0]
  cf = np.empty([D])
  u = np.random.rand(D)        

  cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
  cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

  c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
  c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

  c1 = np.clip(c1, 0, 1)
  c2 = np.clip(c2, 0, 1)

  return c1, c2

def mutate(p, pmdi):
  mp = float(1. / p.shape[0])
  u = np.random.uniform(size=[p.shape[0]])
  r = np.random.uniform(size=[p.shape[0]])
  tmp = np.copy(p)
  for i in range(p.shape[0]):
    if r[i] < mp:
      if u[i] < 0.5:
        delta = (2*u[i]) ** (1/(1+pmdi)) - 1
        tmp[i] = p[i] + delta * p[i]
      else:
        delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
        tmp[i] = p[i] + delta * (1 - p[i])
  tmp = np.clip(tmp, 0, 1)
  return tmp

def variable_swap(p1, p2, probswap):
  D = p1.shape[0]
  swap_indicator = np.random.rand(D) <= probswap
  c1, c2 = p1.copy(), p2.copy()
  c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]
  c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
  return c1, c2

# MULTIFACTORIAL EVOLUTIONARY HELPER FUNCTIONS
def find_relative(population, skill_factor, sf, N):
  return population[np.random.choice(np.where(skill_factor[:N] == sf)[0])]

def calculate_scalar_fitness(factorial_cost):
  return 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)

# MULTIFACTORIAL EVOLUTIONARY WITH TRANSFER PARAMETER ESTIMATION HELPER FUNCTIONS
def get_subpops(population, skill_factor, N):
  K = len(set(skill_factor))
  subpops = []
  for k in range(K):
    idx = np.where(skill_factor == k)[0][:N//K]
    subpops.append(population[idx, :])
  return subpops

class Model:
  def __init__(self, mean, std, num_sample):
    self.mean        = mean
    self.std         = std
    self.num_sample  = num_sample

  def density(self, subpop):
    N, D = subpop.shape
    prob = np.ones([N])
    for d in range(D):
      prob *= norm.pdf(subpop[:, d], loc=self.mean[d], scale=self.std[d])
    return prob

def log_likelihood(rmp, prob_matrix, K):
  posterior_matrix = deepcopy(prob_matrix)
  value = 0
  for k in range(2):
    for j in range(2):
      if k == j:
        posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * (1 - 0.5 * (K - 1) * rmp / float(K))
      else:
        posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * 0.5 * (K - 1) * rmp / float(K)
    value = value + np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
  return value

def learn_models(subpops):
  K = len(subpops)
  D = subpops[0].shape[1]
  models = []
  for k in range(K):
    subpop            = subpops[k]
    num_sample        = len(subpop)
    num_random_sample = int(np.floor(0.1 * num_sample))
    rand_pop          = np.random.rand(num_random_sample, D)
    mean              = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
    std               = np.std(np.concatenate([subpop, rand_pop]), axis=0)
    models.append(Model(mean, std, num_sample))
  return models

def learn_rmp(subpops, D):
  K          = len(subpops)
  rmp_matrix = np.eye(K)
  models = learn_models(subpops)

  for k in range(K - 1):
    for j in range(k + 1, K):
      probmatrix = [np.ones([models[k].num_sample, 2]), 
                    np.ones([models[j].num_sample, 2])]
      probmatrix[0][:, 0] = models[k].density(subpops[k])
      probmatrix[0][:, 1] = models[j].density(subpops[k])
      probmatrix[1][:, 0] = models[k].density(subpops[j])
      probmatrix[1][:, 1] = models[j].density(subpops[j])

      rmp = fminbound(lambda rmp: log_likelihood(rmp, probmatrix, K), 0, 1)
      rmp += np.random.randn() * 0.01
      rmp = np.clip(rmp, 0, 1)
      rmp_matrix[k, j] = rmp
      rmp_matrix[j, k] = rmp

  return rmp_matrix

# OPTIMIZATION RESULT HELPERS
def get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, sf):
  # select individuals from task sf
  idx                = np.where(skill_factor == sf)[0]
  subpop             = population[idx]
  sub_factorial_cost = factorial_cost[idx]
  sub_scalar_fitness = scalar_fitness[idx]

  # select best individual
  idx = np.argmax(sub_scalar_fitness)
  x = subpop[idx]
  fun = sub_factorial_cost[idx, sf]
  return x, fun




def get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message):
    K = len(set(skill_factor))
    N = len(population) // 2
    results = []
    for k in range(K):
        result         = OptimizeResult()
        x, fun         = get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, k)
        result.x       = x
        result.fun     = fun
        result.message = message
        result.nit     = t
        result.nfev    = (t + 1) * N
        results.append(result)
    return results


def mfea2(Problems, config, evaluation_num, Seed, Init_method='random'):
    results_list = []
    match_num = [0,0,0]
    shut_flag = 0
    # unpacking hyper-parameters
    functions = Problems.functions
    K = len(functions)
    N = config['pop_size'] * K
    D = config['dimension']
    T = config['num_iter']
    sbxdi = config['sbxdi']
    pmdi  = config['pmdi']
    pswap = config['pswap']
    rmp_matrix = np.zeros([K, K])

    # initialize
    if Init_method == 'random':
        population = 2 * np.random.random(size=(2 * N, D)) - 1

    elif Init_method == 'uniform':
        # train_x = 2 * lhs(Xdim, Init) - 1
        population = 2 * sobol_seq.i4_sobol_generate(D, 2 * N) - 1

    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # evaluate
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = functions[sf](population[i][np.newaxis,:])
    scalar_fitness = calculate_scalar_fitness(factorial_cost)
    query_list, total_num = Problems.query_num()

  # sort
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]

    message = {'algorithm': 'mfeaii', 'rmp': round(rmp_matrix[0, 1], 1)}
    results = get_optimization_results(0, population, factorial_cost, scalar_fitness, skill_factor, message)
    results_list.append(results)

  # evolve
    iterator = range(1,T)
    for t in iterator:
    # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf

        # learn rmp
        subpops    = get_subpops(population, skill_factor, N)
        rmp_matrix = learn_rmp(subpops, D)

        # select pair to crossover
        for i in range(0, N, 2):
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]

            # crossover
            if sf1 == sf2:
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                c1 = mutate(c1, pmdi)
                c2 = mutate(c2, pmdi)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
                match_num[0]+=1
            elif sf1 != sf2 and np.random.rand() < rmp_matrix[sf1, sf2]:
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                c1 = mutate(c1, pmdi)
                c2 = mutate(c2, pmdi)
                # c1, c2 = variable_swap(c1, c2, pswap)
                if np.random.rand() < 0.5: skill_factor[N + i] = sf1
                else: skill_factor[N + i] = sf2
                if np.random.rand() < 0.5: skill_factor[N + i + 1] = sf1
                else: skill_factor[N + i + 1] = sf2
                match_num[1] += 1
            else:
                p2  = find_relative(population, skill_factor, sf1, N)
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                c1 = mutate(c1, pmdi)
                c2 = mutate(c2, pmdi)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
                match_num[2] += 1

            population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]

        # evaluate
        for i in range(N, 2 * N):
          sf = skill_factor[i]
          factorial_cost[i, sf] = functions[sf](population[i][np.newaxis,:])
          query_list, total_num = Problems.query_num()

          if (total_num >= evaluation_num):
            shut_flag = 1
            population = population[:N+i]
            break
        scalar_fitness = calculate_scalar_fitness(factorial_cost)

        # sort
        sort_index = np.argsort(scalar_fitness)[::-1]
        population = population[sort_index]
        skill_factor = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]

        best_fitness = np.min(factorial_cost, axis=0)
        scalar_fitness = scalar_fitness[sort_index]
        # print(best_fitness)
        # optimization info
        message = {'algorithm': 'mfeaii', 'rmp':round(rmp_matrix[0, 1], 1)}
        results = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
        results_list.append(results)
        print(best_fitness)


        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join('{:0.6f}'.format(res.fun) for res in results), message)
        print(desc)
        # iterator.set_description(desc)
        if shut_flag == 1:
            break


    return results_list, query_list,match_num


def run_MFEA2(problems, Seed, Evol,  Xdim, Exper_floder, Init_method='random'):
    Method ='MFEA2'
    config = {}

    config['pop_size'] = 2*Xdim
    config['dimension'] = Xdim
    config['num_iter'] = 100

    config['sbxdi'] = 10
    config['pmdi'] = 10
    config['pswap'] = 0.5
    results_list, query_list, match_num = mfea2(problems, config, Evol*len(problems.functions), Seed, Init_method = Init_method)
    temp = []
    for res_id, res in enumerate(results_list):
        temp.append([])
        for i in res:
            temp[res_id].append(i.fun)

    result_list = np.array(temp)
    query_list = np.array(query_list)

    if not os.path.exists('{}/data/{}/{}d/{}/'.format(Exper_floder, Method, Xdim, Seed)):
        os.makedirs('{}/data/{}/{}d/{}/'.format(Exper_floder, Method, Xdim, Seed))


    np.savetxt('{}/data/{}/{}d/{}/output.txt'.format(Exper_floder, Method, Xdim, Seed),
               result_list)
    np.savetxt('{}/data/{}/{}d/{}/query_num.txt'.format(Exper_floder, Method, Xdim, Seed),
               query_list)
    np.savetxt('{}/data/{}/{}d/{}/match_num.txt'.format(Exper_floder, Method, Xdim, Seed),
               match_num)

if __name__ == '__main__':
    different_task = [
        'Griewank_stretch_0.15',
        'Rastrigin_stretch_0.93_inv',
        'Rastrigin_stretch_0.93',
        'Rastrigin_stretch_0.93_shift_-3',
        'Rastrigin_stretch_0.93_shift_2',
        'Levy',
        'Schwefel',
        'Schwefel_shift_-300',
        'Ackley_stretch_0.4_shift_20',
    ]

    different_task2 = [
        'Griewank_stretch_0.15',
        'Rastrigin_stretch_0.93_inv',
        'Ackley_stretch_0.4_shift_-20',
        'Ackley_stretch_0.7_shift_-20',
        'Ackley_stretch_0.8_shift_-10',
        'Ackley_stretch_0.8_shift_10',
    ]
    Dim = 5
    func_name_list = different_task2
    problems = Problems(Dim, func_name_list, 0)
    for i in [0,1,2,3,4,5,6,7,8,9,10]:
        run_MFEA2(problems, i,  16*Dim,  Dim, '../../LFL_experiments/test1')