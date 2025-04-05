
# %%
import argparse
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import time
from nbro import *
import multiprocessing
pd.options.display.float_format = '{:.2f}'.format

# %%


def EGO_plug_in(seed):

    n_xi_0 = 100

    # real world data generation
    # xi_all = real_world_data(n_xi * n_iteration + n_xi_0, seed, true_dist)
    xi_all = real_world_data(n_xi * n_iteration//3 + n_xi_0, seed, true_dist)

    # initial experiment design
    D1 = inital_design(n_sample, seed, lower_bound, upper_bound)

    P = xi_all[:, :n_xi_0]

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i, :dimension_x]
        Y_list.append(f.evaluate(xx, P, n_rep, method))

    Y1 = np.array(Y_list).reshape(-1, 1)
    print(D1.shape, Y1.shape)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp0 = GP_model(D1, Y1, method)
    # test = gp0.predict(D1,full_cov = True)[1]
    # print(np.mean(test),np.max(test),np.min(test))

    # get prediction and obtain minimizer
    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    mu_g, sigma_g = gp0.predict(X_, full_cov=False, include_likelihood=False)

    minimizer = list([X_[np.argmin(mu_g)]])
    minimum_value = list([min(mu_g)])

    # Prepare for iteration
    TIME_RE = list()
    D_update = D1
    X_update = np.vstack({tuple(row) for row in D_update})
    Y_update = Y1

    # # Global Optimization algorithm
    for i in range(n_iteration):

        start = time.time()

        # 1. Algorithm for x
        D_new = EGO(X_, X_update, mu_g, sigma_g, gp0)

        # 2. Add selected samples
        D_update, Y_update = update_data(
            D_new, D_update, Y_update, P, n_rep, method, f)
        X_update = np.vstack({tuple(row) for row in D_update})

        # 3. Update GP model and make prediction
        gp0 = GP_model(D_update, Y_update, method)
        mu_g, sigma_g = gp0.predict(
            X_, full_cov=False, include_likelihood=False)

        # 4. Update x^*
        minimizer.append(X_[np.argmin(mu_g)])
        minimum_value.append(min(mu_g))

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

        P = xi_all[:, :(n_xi_0 + n_xi * (i//3+1))]
        # P = xi_all[:, :(n_xi_0 + n_xi * (i+1))]

    output(file_pre, Y_update, D_update, minimizer, minimum_value, TIME_RE,
           f, x_star, f_star, seed, n_sample, n_iteration, method, n_xi)
# %%


def NBRO(seed):

    n_xi_0 = 100

    # real world data generation
    # xi_all = real_world_data(n_xi * n_iteration + n_xi_0, seed, true_dist)
    xi_all = real_world_data(n_xi * n_iteration//3 + n_xi_0, seed, true_dist)
    n_xi_all = n_xi * n_iteration//3 + n_xi_0

    # obtain posterior and get n_MC samples
    P0_list = [None]

    xi = xi_all[:, :n_xi_0]
    # n_k = n_xi * 8
    n_k = n_xi_all * 8
    posterior = posterior_generation(xi, n_MC, n_k, dimension_p, P0_list)

    # the ture minimizer and minimizer value
    x_star = f.x_star
    f_star = f.f_star

    # initial experiment design
    X1, D1 = candidate_space(xi, lower_bound, upper_bound, n_sample,
                             n_k, dimension_p, P0_list, seed=seed, alpha=alpha)
    Y_list = list()
    for i in range(D1.shape[0]):
        D_selected = D1[i]
        xx = D_selected[:dimension_x]
        P = []
        for p in range(dimension_p):
            P.append(
                (D_selected[(dimension_x+2*p*n_k):(dimension_x+(2*p+1)*n_k)],
                 D_selected[(dimension_x+(2*p+1)*n_k):(dimension_x+(2*p+2)*n_k)])
            )
        Y_list.append(f.evaluate(xx, P, n_rep, method))
        # Y_list.append(cost_simulator(s,S,P,rep,method)[0])
    Y1 = np.array(Y_list).reshape(-1, 1)
    print(D1.shape, Y1.shape, Y1.mean())

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp0 = GP_model(D1, Y1, method)
    # test = gp0.predict(D1,full_cov = True)[1]
    # print(np.mean(test),np.max(test),np.min(test))

    # get prediction and obtain minimizer
    X_, D_ = candidate_space(xi, lower_bound, upper_bound, n_candidate,
                             n_k, dimension_p, P0_list, seed=None, alpha=alpha)
    # mu_g,sigma_g= Mu_Std_G(X_,gp0,posterior)
    mu_g = Mu_G(X_, gp0, posterior)
    minimizer = list([X_[np.argmin(mu_g)]])
    minimum_value = list([min(mu_g)])
    print(mu_g)

    # Prepare for iteration
    TIME_RE = list()
    D_update = D1
    X_update = D_update[:, 0:dimension_x]
    Y_update = Y1

    # # Global Optimization algorithm
    for i in range(n_iteration):

        start = time.time()
        # 0. regenerate candidate space for each iteration
        X_, D_ = candidate_space(xi, lower_bound, upper_bound, n_candidate,
                                 n_k, dimension_p, P0_list, seed=None, alpha=alpha)

        # 1. Algorithm for x and \lambda
        D_new = EGO_PB(gp0, posterior, X_update, D_, dimension_x)
        print(D_new.shape)

        # 2. Add selected samples
        D_update, Y_update = update_data_PB(
            D_new, D_update, Y_update, n_rep, method, n_k, f, dimension_x, dimension_p)

        # 3. Update GP model and make prediction
        gp0 = GP_model(D_update, Y_update, method)
        # mu_g, _ = Mu_Std_G(X_,gp0,posterior)
        mu_g = Mu_G(X_, gp0, posterior)
        print(mu_g)

        # 4. Update x^*
        minimizer.append(X_[np.argmin(mu_g)])
        minimum_value.append(min(mu_g))

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

        xi = xi_all[:, :(n_xi_0 + n_xi * (i//3+1))]
        n_k = n_xi_all * 8
        posterior = posterior_generation(xi, n_MC, n_k, dimension_p, P0_list)

        # xi = xi_all[:, :(n_xi_0 + n_xi * (i+1))]
        # n_k = n_xi * 8

    output(file_pre, Y_update, D_update, minimizer, minimum_value, TIME_RE, f, x_star, f_star,
           seed, n_sample, n_iteration, method, n_xi, dimension_p, posterior, n_k, get_g_hat_x)

# %%


def Experiment(seed):

    if method in ["exp", 'hist', 'lognorm']:
        EGO_plug_in(seed)
    elif method == 'NBRO':
        NBRO(seed)


# %%
parser = argparse.ArgumentParser(description='NBRO-algo')
# parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

parser.add_argument('-t', '--time', type=str,
                    help='Date of the experiments, e.g. 20201109', default='20240907_online')
parser.add_argument('-method', '--method', type=str,
                    help='NBRO/hist/exp/lognorm', default="NBRO")
parser.add_argument('-problem', '--problem', type=str,
                    help='Function', default='inventory')

parser.add_argument('-smacro', '--start_num', type=int,
                    help='Start number of macroreplication', default=0)
parser.add_argument('-macro', '--repet_num', type=int,
                    help='Number of macroreplication', default=12)
parser.add_argument('-core', '--n_cores', type=int,
                    help='Number of Cores', default=12)

parser.add_argument('-n_xi', '--n_xi', type=int,
                    help='Number of observations of the random variable', default=10)
parser.add_argument('-n_sample', '--n_sample', type=int,
                    help='number of initial samples', default=20)
parser.add_argument('-n_i', '--n_iteration', type=int,
                    help='Number of iteration', default=90)

parser.add_argument('-n_rep', '--n_replication', type=int,
                    help='Number of replications at each point', default=2)
parser.add_argument('-n_candidate', '--n_candidate', type=int,
                    help='Number of candidate points for each iteration', default=100)
parser.add_argument('-n_MC', '--n_MC', type=int,
                    help='Number of Monte Carlo samples for the posterior', default=100)
parser.add_argument('-n_k', '--n_k', type=int,
                    help='Number of DP truncation', default=1000)  # 50 #100
parser.add_argument('-get_g_hat_x', '--get_g_hat_x',
                    help='get g_hat_x', action='store_true')

parser.add_argument('-dist', '--true_distribution', type=str,
                    help='exp/m-lognorm', default='m-lognorm')
parser.add_argument('-alpha', '--alpha', type=int, help='alpha', default=10)


# cmd = ['-t','20230909',
#        '-method', 'NBRO',
#        '-problem', 'inventory',
#        '-smacro', '0',
#        '-macro','1',
#        '-n_xi', '10',
#        '-n_sample', '20',
#        '-n_i', '5',
#        '-n_rep', '10',
#        '-n_candidate','100',
#        '-n_MC', '100',
#        '-n_k', '100',
#        '-get_g_hat_x',
#        '-core', '4',
#        '-dist', 'exp']

# cmd = ['-get_g_hat_x']
# args = parser.parse_args(cmd)
args = parser.parse_args()
print(args)

time_run = args.time
method = args.method
problem = args.problem

n_xi = args.n_xi

alpha = args.alpha
n_sample = args.n_sample  # number of initial samples
n_iteration = args.n_iteration  # Iteration

n_rep = args.n_replication  # number of replication on each point
# Each iteration, number of the candidate points are regenerated+
n_candidate = args.n_candidate
n_MC = args.n_MC  # number of Monte Carlo samples
get_g_hat_x = args.get_g_hat_x

# true distribution of the demand distribution
if args.true_distribution == 'm-lognorm':
    if problem == "inventory":
        true_dist = [{'dist': "m-lognorm",
                      'mu': [5000, 10000],
                      'sigma': [5000, 5000],
                      'p_i': [0.5, 0.5]}]
    else:
        true_dist = [{'dist': "m-lognorm",
                      'mu': [10, 20],
                      'sigma': [10, 5], 'p_i': [0.5, 0.5]}]
else:
    true_dist = [{'dist': 'exp', 'rate': 0.0002}]

# %%
if problem == "inventory":
    f = inventory_problem(true_dist)

# Inventory problem specification
dimension_x = f.dimension_x
dimension_p = f.dimension_p

# lamda lower_bound and upper bound
lower_bound = f.lb  # [1, 2.26]
upper_bound = f.ub  # [2.25, 3.5]

f_star = f.f_star
x_star = f.x_star

print(f_star, x_star)

# f.estimate_minimizer_minimum()
# (323.60402621886226, array([2.197462  , 2.62609054]))

file_pre = '_'.join(["outputs/bo", time_run, 'alpha',
                    str(alpha), problem, method, true_dist[0].get('dist')])

if __name__ == '__main__':

    start_num = args.start_num
    repet_num = args.repet_num  # Macro-replication

    import multiprocessing
    starttime = time.time()
    pool = multiprocessing.Pool()
    pool.map(Experiment, range(start_num, repet_num))
    pool.close()
    print('That took {} secondas'.format(time.time() - starttime))
