"""
Use NES under the entropic regularizer to optimize viewpoint parameters
Modify on the basis of estool: http://blog.otoro.net/2017/11/12/evolving-stable-strategies/
"""

import numpy as np
from evaluate import comput_fitness
from rendering_image import render_image
# from classifier.predict import test_baseline
from tqdm import tqdm
from datasets.opts import get_opts
import joblib
import torch

import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
import classifier.predict 
from classifier.predict import test_baseline 


def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_normalize(x):
  mean = np.mean(x)
  var = np.var(x)
  y = x-mean/var
  return y

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
  def __init__(self, pi, epsilon=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon
    self.t = 0

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  def _compute_step(self, globalg):
    raise NotImplementedError


class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step


class PEPG:
  '''Extension of PEPG with bells and whistles.'''

  def __init__(self, num_params,  # number of model parameters
               sigma_init=0.10,  # initial standard deviation
               sigma_alpha=0.20,  # learning rate for standard deviation
               sigma_decay=0.999,  # anneal standard deviation
               sigma_limit=0.01,  # stop annealing if less than this
               sigma_max_change=0.2,  # clips adaptive sigma to 20%
               sigma_min=0.05,  # The minimum sigma allowed
               sigma_update=True,
               learning_rate=0.01,  # learning rate for standard deviation
               learning_rate_decay=0.9999,  # annealing the learning rate
               learning_rate_limit=0.01,  # stop annealing learning rate
               elite_ratio=0,  # if > 0, then ignore learning_rate
               popsize=256,  # population size
               average_baseline=True,  # set baseline to average of batch
               weight_decay=0.01,  # weight decay coefficient
               rank_fitness=True,  # use rank rather than fitness numbers
               forget_best=True,
               mu_lambda=0.0001,
               sigma_lambda=0.0001):  # don't keep the historical best solution

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_alpha = sigma_alpha
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.sigma_max_change = sigma_max_change
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.average_baseline = average_baseline
    self.sigma_update = sigma_update
    self.sigma_min = sigma_min
    self.mu_lamba = mu_lambda
    self.sigma_lamba = sigma_lambda

    if self.average_baseline:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.batch_size = int(self.popsize / 2)
    else:
      assert (self.popsize & 1), "Population size must be odd"
      self.batch_size = int((self.popsize - 1) / 2)

    # option to use greedy es method to select next mu, rather than using drift param
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.use_elite = False
    if self.elite_popsize > 0:
      self.use_elite = True

    self.forget_best = forget_best
    self.batch_reward = np.zeros(self.batch_size * 2)
    self.mu = np.zeros(self.num_params)
    self.sigma = np.ones(self.num_params) * self.sigma_init
    self.curr_best_mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True  # always forget the best one if we rank
    # choose optimizer
    self.optimizer = SGD(self, learning_rate)

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma * sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
    if self.average_baseline:
      epsilon = self.epsilon_full
    else:
      # first population is mu, then positive epsilon, then negative epsilon
      epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
      self.r = epsilon/self.sigma.reshape(1, self.num_params)
    solutions = self.mu.reshape(1, self.num_params) + epsilon
    self.solutions = solutions
    return solutions

  def comput_entropy(self):
    # Calculate the entropy of the Gaussian distribution for each batch
    r = torch.Tensor(self.r)  # Sample point in N(0,1).
    sigma = torch.Tensor(self.sigma)
    sigma.requires_grad = True
    mu = torch.Tensor(self.mu)
    mu.requires_grad = True


    inside = 1-torch.pow(torch.tanh(mu+sigma*r), 2)+1e-8
    neg_logp = torch.log(sigma+1e-8) + 1/2*torch.pow(r, 2) + torch.log(inside)
    entropy = torch.sum(neg_logp, 0)/self.popsize
    print('entropy:\n', entropy)
    Entropy = torch.sum(entropy)

  
    Entropy.backward()
    mu_entropy_grad = mu.grad.clone()
    sigma_entropy_grad = sigma.grad.clone()

    mu.grad.data.zero_()
    sigma.grad.data.zero_()
    print("Entropy:\n", Entropy)
    self.entropy = Entropy

    return mu_entropy_grad.cpu().detach().numpy(), sigma_entropy_grad.cpu().detach().numpy()





  def tell(self, reward_table_result, mu_entropy_grad, sigma_entropy_grad):
    # input must be a numpy float array
    assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    reward_table = np.array(reward_table_result)

    if self.rank_fitness:
      reward_table = compute_centered_ranks(reward_table)
      # reward_table = compute_normalize(reward_table)

    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay

    reward_offset = 1
    if self.average_baseline:
      b = np.mean(reward_table)
      reward_offset = 0
    else:
      b = reward_table[0]  # baseline

    reward = reward_table[reward_offset:]
    if self.use_elite:
      idx = np.argsort(reward)[::-1][0:self.elite_popsize]
    else:
      idx = np.argsort(reward)[::-1]

    best_reward = reward[idx[0]]
    if (best_reward > b or self.average_baseline):
      best_mu = self.mu + self.epsilon_full[idx[0]]
      best_reward = reward[idx[0]]
    else:
      best_mu = self.mu
      best_reward = b

    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.sigma = np.ones(self.num_params) * self.sigma_init
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # short hand
    epsilon = self.epsilon
    sigma = self.sigma

    # update the mean

    # move mean to the average of the best idx means
    if self.use_elite:
      self.mu += self.epsilon_full[idx].mean(axis=0)
    else:
      rT = (reward[:self.batch_size] - reward[self.batch_size:])
      change_mu = np.dot(rT, epsilon) + self.mu_lamba*mu_entropy_grad
      #print('rt:\n', rT)
      #print('epsilon', epsilon)
      print('mu-loss1:', np.dot(rT, epsilon))
      print('mu-loss2:', self.mu_lamba*mu_entropy_grad)

      self.optimizer.stepsize = self.learning_rate
      update_ratio = self.optimizer.update(-change_mu)  # adam, rmsprop, momentum, etc.
      # self.mu += (change_mu * self.learning_rate) # normal SGD method

    # adaptive sigma
    # normalization
    
    #if (self.sigma[a] > self.sigma_min for a in range(self.num_params)):
    if (self.sigma_alpha > 0 and self.sigma_update):
      stdev_reward = 1.0
      if not self.rank_fitness:
        stdev_reward = reward.std()
      S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
      reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
      rS = reward_avg - b

      delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

      # adjust sigma according to the adaptive sigma calculation
      # for stability, don't let sigma move more than 10% of orig value
      change_sigma = self.sigma_alpha * (delta_sigma + self.sigma_lamba*sigma_entropy_grad)

      print('sigma-loss1:', delta_sigma)
      print('sigma-loss2:', self.sigma_lamba*sigma_entropy_grad)

      change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
      change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
      self.sigma += change_sigma
      self.sigma = np.clip(self.sigma, 0.0, 0.15)

      if (self.sigma_decay < 1):
        self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

    if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  def current_param(self):
    return self.curr_best_mu

  def set_mu(self, mu):
    self.mu = np.array(mu)

  def best_param(self):
    return self.best_mu

  def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)





def NES_search():
  args = get_opts()
  search_num = args.search_num

  # Search from 6D space: both Angle and position (ψ, θ, ϕ, ∆x, ∆y, ∆z)
  if search_num == 6:

    MAX_ITERATION = args.iteration
    POPSIZE = args.popsize
    NUM_PARAMS = 6
    N_JOBS = 3
    # Search for 6D spaces，th phi gamma r x y
    solver = PEPG(num_params=NUM_PARAMS,  # number of model parameters
                  sigma_init=0.1,  # initial standard deviation
                  sigma_update=True,  # Sigma is not significantly updated
                  learning_rate=0.1,  # learning rate for standard deviation
                  learning_rate_decay=0.99,
                  learning_rate_limit=0,  # don't anneal the learning rate
                  popsize=POPSIZE,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=args.mu_lamba,
                  sigma_lambda=args.sigma_lamba
                  )

    logging = {'mu': [], 'sigma': [], 'fitness': [], 'entropy':[]}
    history = []
    fitness_origin = []
    history_best_solution = []
    for j in range(MAX_ITERATION):
      solutions = solver.ask()
      mu_entropy_grad, sigma_entropy_grad = solver.comput_entropy()

      # gamma (-30,30)
      solutions[:, 0] = 30 * np.tanh(solutions[:, 0])
      # th (-180,180)
      solutions[:, 1] = 180 * np.tanh(solutions[:, 1])
      # phi (-70, 70)
      solutions[:, 2] = 70 * np.tanh(solutions[:, 2])
      # r (3, 5)
      solutions[:, 3] = np.tanh(solutions[:, 3]) + 4
      # x (-0.5, 0.5)
      solutions[:, 4] = 0.5 * np.tanh(solutions[:, 4])
      # x (-0.5, 0.5)
      solutions[:, 5] = 0.5 * np.tanh(solutions[:, 5])

      fitness_list = np.zeros(solver.popsize)


      #  Multi-process
      with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        #for i in tqdm(range(solver.popsize)):
          #fitness_list[i] = comput_fitness(solutions[i])

        fitness_list = parallel(joblib.delayed(comput_fitness)(solutions[i], solver.sigma) for i in tqdm(range(solver.popsize)))

      solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad)
      result = solver.result()  # first element is the best solution, second element is the best fitness

      history.append(result[1])
      fitness_origin.append(np.max(fitness_list))
      average_fitness = np.mean(fitness_list)


      max_idx = np.argmax(fitness_list)
      history_best_solution.append(solutions[max_idx])
      if (j + 1) % 1 == 0:
        print("fitness at iteration\n", (j + 1), max(fitness_origin))
        print("average fitness at iteration\n", (j + 1), average_fitness)
        print("sigma at iteration\n", (j + 1), result[3])
        print("mu at iteration\n", (j + 1), result[0])

        logging['fitness'].append(result[1])
        logging['sigma'].append(result[3])
        logging['mu'].append(result[0])
        logging['entropy'].append(solver.entropy)
      # print('fitness_list', fitness_list)

      #if average_fitness > -0.25:
        #break
        

    max_idx_ = 0

    for i in range(len(history) - 1):
      if history[i + 1] > history[i]:
        max_idx_ = i + 1
      else:
        continue

    best_solutions = history_best_solution[max_idx_]

    # Outputs the sampled values of sigma and mu after tanh
    random = np.zeros([args.num_sample+1, 6])
    gamma = np.random.normal(loc=result[0][0], scale=result[3][0], size=args.num_sample)
    th = np.random.normal(loc=result[0][1], scale=result[3][1], size=args.num_sample)
    phi = np.random.normal(loc=result[0][2], scale=result[3][2], size=args.num_sample)
    r = np.random.normal(loc=result[0][3], scale=result[3][3], size=args.num_sample)
    a = np.random.normal(loc=result[0][4], scale=result[3][4], size=args.num_sample)
    b = np.random.normal(loc=result[0][5], scale=result[3][5], size=args.num_sample)
  
    gamma = np.append(gamma, result[0][0])
    th = np.append(th, result[0][1])
    phi = np.append(phi, result[0][2])
    r = np.append(r, result[0][3])
    a = np.append(a, result[0][4])
    b = np.append(b, result[0][5])
    

    random[:, 0] = 30 * np.tanh(gamma)
    random[:, 1] = 180 * np.tanh(th)
    random[:, 2] = 70 * np.tanh(phi)
    random[:, 3] = np.tanh(r) + 4.0
    random[:, 4] = 0.5 * np.tanh(a)
    random[:, 5] = 0.5 * np.tanh(b)
    mu = random.mean(axis=0)
    var = (random - mu).T @ (random - mu) / random.shape[0]
    var = np.sqrt(np.diagonal(var))  # this is slightly suboptimal, but instructive
    print('final sigma after tanh(var)', var)

    mu = np.zeros([6])
    mu[0] = 30 * np.tanh(result[0][0])
    mu[1] = 180 * np.tanh(result[0][1])
    mu[2] = 70 * np.tanh(result[0][2])
    mu[3] = np.tanh(result[0][3]) + 4.0
    mu[4] = 0.5 * np.tanh(result[0][4])
    mu[5] = 0.5 * np.tanh(result[0][5])

    print('final mu after tanh(mean)', mu)

    np.save('logging_test4.npy', logging)
    
    "Render 100 images of this distribution"
    print('begin render 100 images in current adv-distribution')
    print('--------------------------------------------------')
    render_image(random, is_over=True)

    "Verify Accuracy"
    print('begin test the accuracy')
    print('--------------------------------------------------')
    path = '/HOME/scz1972/run/rsw_/NeRFAttack/results/blender_for_attack/' + args.scene_name + '/'
    test_baseline(path=path, label=args.label_name, model='resnet')

    print('entropy')
    print('--------------------------------------------------')
    print(solver.entropy)


    print('no.100 the mean img')
    print('--------------------------------------------------')
    path = '/HOME/scz1972/run/rsw_/NeRFAttack/results/blender_for_attack/' + args.scene_name + '/'
    test_baseline(path=path, label=args.label_name, model='resnet', is_mean=True)


    #x = render_image(best_solutions)
    #test_baseline(path="C:/Users/Silvester/PycharmProjects/NeRFAttack/NeRF/results/blender_for_attack/'hotdog'/",label='hotdog, hot dog, red hot')




  # only Angle search (ψ, θ, ϕ)
  if search_num == 123:

    MAX_ITERATION = args.iteration
    POPSIZE = args.popsize
    NUM_PARAMS = 3
    N_JOBS = 3
    max_stop_fitness = 6.0

    solver = PEPG(num_params=NUM_PARAMS,  # number of model parameters
                  sigma_init=0.1,  # initial standard deviation
                  sigma_update=True,  # Sigma is not significantly updated
                  learning_rate=0.1,  # learning rate for standard deviation
                  learning_rate_decay=0.99,
                  learning_rate_limit=0,  # don't anneal the learning rate
                  popsize=POPSIZE,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=args.mu_lamba,
                  sigma_lambda=args.sigma_lamba
                  )

    logging = {'mu': [], 'sigma': [], 'fitness': [], 'entropy':[]}
    history = []
    fitness_origin = []
    history_best_solution = []
    for j in range(MAX_ITERATION):
      solutions = solver.ask()
      mu_entropy_grad, sigma_entropy_grad = solver.comput_entropy()
      solutions_ = np.zeros([POPSIZE, 6])

      # gamma (-60,60)
      solutions_[:, 0] = 30 * np.tanh(solutions[:, 0])
      # th (-180,180)
      solutions_[:, 1] = 180 * np.tanh(solutions[:, 1])
      # phi (-60, 60)
      solutions_[:, 2] = 70 * np.tanh(solutions[:, 2])

      # The fixed position parameter is （4.0， 0， 0）

      # r (4, 6)
      solutions_[:, 3] = 4.0
      # x (-1, 1)
      solutions_[:, 4] = 0.0
      # x (-1, 1)
      solutions_[:, 5] = 0.0

      fitness_list = np.zeros(solver.popsize)


      #  Multi-process
      with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        #for i in tqdm(range(solver.popsize)):
          #fitness_list[i] = comput_fitness(solutions[i])

        fitness_list = parallel(joblib.delayed(comput_fitness)(solutions_[i], solver.sigma) for i in tqdm(range(solver.popsize)))

      solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad)
      result = solver.result()  # first element is the best solution, second element is the best fitness

      fitness_origin.append(np.max(fitness_list))
      history.append(result[1])
      average_fitness = np.mean(fitness_list)
      max_idx = np.argmax(fitness_list)
      history_best_solution.append(solutions[max_idx])
      if (j + 1) % 1 == 0:
        print("fitness at iteration\n", (j + 1), max(fitness_origin))
        print("average fitness at iteration\n", (j + 1), average_fitness)
        print("sigma at iteration\n", (j + 1), result[3])
        print("mu at iteration\n", (j + 1), result[0])

        logging['fitness'].append(result[1])
        logging['sigma'].append(result[3])
        logging['mu'].append(result[0])
        logging['entropy'].append(solver.entropy)
      # print('fitness_list', fitness_list)

      #if average_fitness > max_stop_fitness:
        #break

    max_idx_ = 0

    for i in range(len(history) - 1):
      if history[i + 1] > history[i]:
        max_idx_ = i + 1
      else:
        continue

    best_solutions = history_best_solution[max_idx_]

    random = np.zeros([args.num_sample+1, 6])
    gamma = np.random.normal(loc=result[0][0], scale=result[3][0], size=args.num_sample)
    th = np.random.normal(loc=result[0][1], scale=result[3][1], size=args.num_sample)
    phi = np.random.normal(loc=result[0][2], scale=result[3][2], size=args.num_sample)

    gamma = np.append(gamma, result[0][0])
    th = np.append(th, result[0][1])
    phi = np.append(phi, result[0][2])


    random[:, 0] = 30 * np.tanh(gamma)
    random[:, 1] = 180 * np.tanh(th)
    random[:, 2] = 70 * np.tanh(phi)
    random[:, 3] = 4.0
    random[:, 4] = 0.0
    random[:, 5] = 0.0
    mu = random.mean(axis=0)
    var = (random - mu).T @ (random - mu) / random.shape[0]
    var = np.sqrt(np.diagonal(var))  # this is slightly suboptimal, but instructive
    print('final sigma after tanh(var)', var)

    mu = np.zeros([6])
    mu[0] = 30 * np.tanh(result[0][0])
    mu[1] = 180 * np.tanh(result[0][1])
    mu[2] = 70 * np.tanh(result[0][2])
    mu[3] = 4.0
    mu[4] = 0.0
    mu[5] = 0.0

    print('final mu after tanh(mean)', mu)

    "Render 100 images of this distribution"
    print('begin render 100 images in current adv-distribution')
    print('--------------------------------------------------')
    render_image(random, is_over=True)

    "Verify Accuracy"
    print('begin test the accuracy')
    print('--------------------------------------------------')
    path = '/HOME/scz1972/run/rsw_/NeRFAttack/run_NES_A/results/blender_for_attack/' + args.scene_name + '/'
    test_baseline(path=path, label=args.label_name, model='vit')

    print('no.100 the mean img')
    print('--------------------------------------------------')
    path = '/HOME/scz1972/run/rsw_/NeRFAttack/results/blender_for_attack/' + args.scene_name + '/'
    test_baseline(path=path, label=args.label_name, model='vit', is_mean=True)

    #x = render_image(best_solutions)
    #test_baseline(path="C:/Users/Silvester/PycharmProjects/NeRFAttack/NeRF/results/blender_for_attack/'hotdog'/",label='hotdog, hot dog, red hot')




# only position search(∆x, ∆y, ∆z)
  if search_num == 456:

    MAX_ITERATION = args.iteration
    POPSIZE = args.popsize
    NUM_PARAMS = 3
    N_JOBS = 3
    max_stop_fitness = 6.0

    solver = PEPG(num_params=NUM_PARAMS,  # number of model parameters
                  sigma_init=0.1,  # initial standard deviation
                  sigma_update=True,  # Sigma is not significantly updated
                  learning_rate=0.1,  # learning rate for standard deviation
                  learning_rate_decay=0.99,
                  learning_rate_limit=0,  # don't anneal the learning rate
                  popsize=POPSIZE,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=args.mu_lamba,
                  sigma_lambda=args.sigma_lamba
                  )

    logging = {'mu': [], 'sigma': [], 'fitness': [], 'entropy':[]}
    history = []
    fitness_origin = []
    history_best_solution = []
    for j in range(MAX_ITERATION):
      solutions = solver.ask()
      mu_entropy_grad, sigma_entropy_grad = solver.comput_entropy()
      solutions_ = np.zeros([POPSIZE, 6])

      # gamma (-60,60)
      solutions_[:, 0] = 0.0
      # th (-180,180)
      solutions_[:, 1] = 0.0
      # phi (-60, 60)
      solutions_[:, 2] = 45.0

      # The fixed position parameter is（4.0， 0， 0）

      # r (4, 6)
      solutions_[:, 3] = np.tanh(solutions[:, 0]) + 4
      # x (-1, 1)
      solutions_[:, 4] = 0.5 * np.tanh(solutions[:, 1])
      # x (-1, 1)
      solutions_[:, 5] = 0.5 * np.tanh(solutions[:, 2])

      fitness_list = np.zeros(solver.popsize)


      #  Multi-process
      with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        #for i in tqdm(range(solver.popsize)):
          #fitness_list[i] = comput_fitness(solutions[i])

        fitness_list = parallel(joblib.delayed(comput_fitness)(solutions_[i], solver.sigma) for i in tqdm(range(solver.popsize)))

      solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad)
      result = solver.result()  # first element is the best solution, second element is the best fitness

      fitness_origin.append(np.max(fitness_list))
      history.append(result[1])
      average_fitness = np.mean(fitness_list)
      max_idx = np.argmax(fitness_list)
      history_best_solution.append(solutions[max_idx])
      if (j + 1) % 1 == 0:
        print("fitness at iteration\n", (j + 1), max(fitness_origin))
        print("average fitness at iteration\n", (j + 1), average_fitness)
        print("sigma at iteration\n", (j + 1), result[3])
        print("mu at iteration\n", (j + 1), result[0])

        logging['fitness'].append(result[1])
        logging['sigma'].append(result[3])
        logging['mu'].append(result[0])
        logging['entropy'].append(solver.entropy)
      # print('fitness_list', fitness_list)
  
      #if average_fitness > max_stop_fitness:
        #break

    max_idx_ = 0

    for i in range(len(history) - 1):
      if history[i + 1] > history[i]:
        max_idx_ = i + 1
      else:
        continue

    best_solutions = history_best_solution[max_idx_]

    random = np.zeros([args.num_sample+1, 6])
    r = np.random.normal(loc=result[0][0], scale=result[3][0], size=args.num_sample)
    a = np.random.normal(loc=result[0][1], scale=result[3][1], size=args.num_sample)
    b = np.random.normal(loc=result[0][2], scale=result[3][2], size=args.num_sample)

    r = np.append(r, result[0][0])
    a = np.append(a, result[0][1])
    b = np.append(b, result[0][2])


    random[:, 0] = 0.0
    random[:, 1] = 0.0
    random[:, 2] = 45.0
    random[:, 3] = np.tanh(r) + 4.0
    random[:, 4] = 0.5 * np.tanh(a)
    random[:, 5] = 0.5 * np.tanh(b)

    
    mu = random.mean(axis=0)
    var = (random - mu).T @ (random - mu) / random.shape[0]
    var = np.sqrt(np.diagonal(var))  # this is slightly suboptimal, but instructive
    print('final sigma after tanh(var)', var)

    mu = np.zeros([6])
    mu[0] = 0.0
    mu[1] = 0.0
    mu[2] = 45.0
    mu[3] = np.tanh(result[0][0]) + 4.0
    mu[4] = 0.5 * np.tanh(result[0][1])
    mu[5] = 0.5 * np.tanh(result[0][2])

    print('final mu after tanh(mean)', mu)

    "Render 100 images of this distribution"
    print('begin render 100 images in current adv-distribution')
    print('--------------------------------------------------')
    render_image(random, is_over=True)

    "Verify Accuracy"
    print('begin test the accuracy')
    print('--------------------------------------------------')
    path = '/HOME/scz1972/run/rsw_/NeRFAttack/run_NES_P/results/blender_for_attack/' + args.scene_name + '/'
    test_baseline(path=path, label=args.label_name, model='vit')

    print('no.100 the mean img')
    print('--------------------------------------------------')
    path = '/HOME/scz1972/run/rsw_/NeRFAttack/results/blender_for_attack/' + args.scene_name + '/'
    test_baseline(path=path, label=args.label_name, model='vit', is_mean=True)

    #x = render_image(best_solutions)
    #test_baseline(path="C:/Users/Silvester/PycharmProjects/NeRFAttack/NeRF/results/blender_for_attack/'hotdog'/",label='hotdog, hot dog, red hot')