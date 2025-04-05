import os
from numpy.random import beta
from typing import List
from scipy.special import rel_entr
from numpy.random import choice
from paramz.transformations import Logexp
from GPy.core.parameterization import Param
from GPy.kern import Kern
import GPy
import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy
from pyDOE import *


class NumericalFunctions:
    def __int__(self, true_dist, dy):

        self.dimension_x = None
        self.dimension_p = None
        self.lb = None
        self.ub = None
        self.true_dist = true_dist[0]
        self.dy = dy
        self.x_star = None
        self.f_star = None

    def __call__(self, xx):

        val = self.evaluate(xx)

        return val

    def evaluate(self, xx, P, n_rep, method):

        val = self.evaluate_simulate(xx, P, method)
        noise = np.random.normal(0, self.dy, n_rep)
        val = val + np.mean(noise)
        val = val.item()

        return val

    def evaluate_true(self, xx, xi_number=10000):
        values = 0
        xi = real_world_data(xi_number, 0, [self.true_dist])
        xi = xi.reshape(-1, 1)
        for i in range(xi_number):
            values += self.function_evaluation(np.concatenate((xx, xi[i, :])))
        values = values/xi_number
        return values

    def evaluate_simulate(self, xx, P, method, xi_number=100):

        if method == 'NBRO':  # "nonparametric"
            p, theta = P[0]
            D = np.random.choice(theta, size=xi_number, p=p)
        elif method == 'hist':  # 'histogram'
            print(P, P.shape)
            D = np.random.choice(P.reshape(-1), size=xi_number)
        elif method == 'exp':  # "parametric"
            D = fit_and_generate_exponential(xi=P, size=xi_number)
        elif method == 'lognorm':
            D = fit_and_generate_log_normal(xi=P, size=xi_number)
        else:
            print('Wriong input!')

        values = 0
        D = D.reshape(-1, 1)
        for i in range(D.shape[0]):
            values += self.function_evaluation(np.concatenate((xx, D[i, :])))
        values = values/D.shape[0]

        return values

    def function_evaluation(self, x):
        pass

    def estimate_minimizer_minimum(self, n_sample=1000):

        np.random.seed(None)
        lhd = lhs(self.dimension_x, samples=n_sample, criterion="maximin")
        X_ = np.zeros((n_sample, self.dimension_x))
        for i in range(self.dimension_x):
            X_[:, i] = lhd[:, i]*(self.ub[i]-self.lb[i]) + self.lb[i]
        Y_ = np.zeros(X_.shape[0])
        for i in range(X_.shape[0]):
            Y_[i] = self.evaluate_true(X_[i])
        self.f_star = min(Y_)
        self.x_star = X_[np.argmin(Y_)]

        return self.f_star, self.x_star


class StybTang(NumericalFunctions):

    def __init__(self, true_dist, dy=0.1):

        self.dimension_x = 2
        self.dimension_p = 1
        self.lb = np.ones(self.dimension_x)
        self.ub = np.ones(self.dimension_x)
        self.lb[:self.dimension_x] = np.array(
            [-5] * self.dimension_x)  # change
        self.ub[:self.dimension_x] = np.array([5] * self.dimension_x)  # change
        self.true_dist = true_dist[0]
        self.dy = dy
        self.x_star = None
        self.f_star = None
        self.mean = 536.8  # 398184
        self.std = 27443.7  # 172876.76
        # self.mean = -8.72
        # self.std = 45.17

    def function_evaluation(self, x):  # change

        R = x**4 - 16*x**2 + 5*x
        R = 1/2 * np.sum(R)

        val = R.item()

        val = (val - self.mean)/self.std

        return val


class Griewank(NumericalFunctions):

    def __init__(self, true_dist, dy=0.1):

        self.dimension_x = 2
        self.dimension_p = 1
        self.lb = np.ones(self.dimension_x)
        self.ub = np.ones(self.dimension_x)
        self.lb[:self.dimension_x] = np.array(
            [-50] * self.dimension_x)  # change
        self.ub[:self.dimension_x] = np.array(
            [50] * self.dimension_x)  # change
        self.true_dist = true_dist[0]
        self.dy = dy
        self.x_star = None
        self.f_star = None
        self.mean = 1.49
        self.std = 0.48
        # self.mean = 1.42
        # self.std = 0.57

    def function_evaluation(self, x):  # change

        d = self.dimension_x + self.dimension_p
        R = np.sum(x**2/4000) - \
            np.prod(np.cos(x/np.sqrt(np.arange(1, d+1)))) + 1
        val = R.item()

        val = (val - self.mean)/self.std

        return val


class inventory_problem:

    def __init__(self, true_dist):

        self.dimension_x = 2
        self.dimension_p = 1
        self.lb = [1, 2.26]
        self.ub = [2.25, 3.5]
        self.true_dist = true_dist[0]
        # list([np.array([0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]).reshape(1,-1)])
        self.x_star = None
        self.f_star = None  # -3.32237

        # self.mean = -0.26
        # self.std = 0.38
        # self.f_star = (self.f_star - self.mean)/self.std
        # x_star = np.array([2.20849609,2.30601563])
        # f_star = inventory(2.20849609, 2.30601563,true_dist.get("lambda"))
        # print(x_star, f_star)

    def __call__(self, xx):

        value = self.evaluate_true(xx)

        return value

    def evaluate(self, xx, P, n_rep, method):
        s = xx[0]
        S = xx[1]
        P = P[0]
        value = cost_simulator(s, S, P, n_rep, method)[0]
        return value

    def evaluate_true(self, xx):

        if self.true_dist.get("dist") == "exp":
            s = xx[0]
            S = xx[1]
            value = inventory(s, S, self.true_dist.get("rate"))
        else:
            s = xx[0]
            S = xx[1]
            value = cost_simulator(s, S, self.true_dist,
                                   100, self.true_dist.get('dist'))[0]

        return value

    def estimate_minimizer_minimum(self, n_sample=1000):
        np.random.seed(None)
        lhd = lhs(self.dimension_x, samples=n_sample, criterion="maximin")
        X_ = np.zeros((n_sample, self.dimension_x))
        for i in range(self.dimension_x):
            X_[:, i] = lhd[:, i]*(self.ub[i]-self.lb[i]) + self.lb[i]
        Y_ = np.zeros(X_.shape[0])
        for i in range(X_.shape[0]):
            Y_[i] = self.evaluate_true(X_[i])
        self.f_star = min(Y_)
        self.x_star = X_[np.argmin(Y_)]

        return self.f_star, self.x_star


def cost_simulator(s, S, P, n_rep=10, method='NBRO', NOP=1000, b=100, h=1, c=1, K=100):
    # h holding cost per period per unit of inventory
    # b shortage cost per period per unit of inventory
    # c per-unit ordering cost
    # K setup cost for placing an order

    s = s*10000
    S = S*10000
    output = np.zeros(n_rep)

    for i in np.arange(n_rep):
        output[i] = costsim(NOP, P, b, h, c, K, s, S, method)

    # take the average over all replications (x-bar[n])
    cost_y = np.mean(output)
    cost_v = np.var(output)/n_rep  # noise withe rep replications

    return cost_y, cost_v


def costsim(NOP, P, b, h, c, K, s, S, method):
    # types
    burnin = 100

    if isinstance(P, dict):
        # if P is a dictionary contains the parameter information
        if method == 'm-lognorm':
            D = set_generate_mixture_of_log_normal(xi_n=NOP, mu=P.get(
                "mu"), sigma=P.get("sigma"), p_i=P.get("p_i"))
        elif method == 'exp':
            D = set_generate_exponential(NOP, P.get("lambda"))
    else:
        if method == 'NBRO':  # "nonparametric"
            p, theta = P
            D = np.random.choice(theta, size=NOP, p=p)
        elif method == 'hist':  # 'histogram'
            D = np.random.choice(P, size=NOP)
        elif method == 'exp':  # "parametric"
            D = fit_and_generate_exponential(xi=P, size=NOP)
        elif method == 'lognorm':
            D = fit_and_generate_log_normal(xi=P, size=NOP)
        else:
            print('Wriong input!')

    if np.min(D) < 0:
        print('negative demand')

    cost = np.zeros(NOP)
    O = np.zeros(NOP)  # Order
    I = np.zeros(NOP)  # inventory on-hand at the end of period

    for t in np.arange(NOP):

        if t == 0:
            I[t] = S-D[t]  # inventory on-hand at the end of period
        else:
            I[t] = I[t-1]+O[t-1] - D[t]

        if I[t] > s:
            O[t] = 0
        else:
            O[t] = S-I[t]

        if O[t] > 0:
            cost[t] = c * O[t] + K + h * \
                np.max((I[t], 0)) + b * np.max((-I[t], 0))

        else:
            cost[t] = c * O[t] + h * np.max((I[t], 0)) + b * np.max((-I[t], 0))

        # print(c * O[t], h * np.max(I[t],0), b * np.max((-I[t],0)))

    cost2 = cost[burnin:]

    Ecost = np.sum(cost2)/len(cost2)

    return Ecost/30000.0  # 100


def inventory(s, S, lamda, b=100, h=1, c=1, K=100):
    s = s*10000
    S = S*10000
    output = c/lamda + (K+h*(s-1/lamda+0.5*lamda*(S**2 - s**2)) +
                        (h+b)/lamda*np.exp(-lamda*s))/(1+lamda*(S-s))
    return output/30000.0  # 100


class critical_care_facility_problem:

    def __init__(self, true_dist):

        self.dimension_x = 3
        self.dimension_p = 6
        self.lb = [10, 3, 10]
        self.ub = [15, 8, 25]
        self.true_dist = true_dist
        self.x_star = None
        self.f_star = None
        self.sample_space = None

    def set_sample_space(self):

        B1_list = np.arange(10, 15)
        B2_list = np.arange(3, 8)
        B3_list = np.arange(10, 25)
        candidate = np.array([[i, j, k]
                             for i in B1_list for j in B2_list for k in B3_list])
        self.sample_space = candidate[((candidate[:, 0] + candidate[:, 1] + candidate[:, 2]/2) <= 28) & (
            (candidate[:, 0] + candidate[:, 1] + candidate[:, 2]/2) >= 26), :]

        return self.sample_space

    def __call__(self, xx):

        value = self.evaluate_true(xx)

        return value

    def evaluate(self, xx, P, n_rep, method):
        b1 = xx[0]
        b2 = xx[1]
        b3 = xx[2]
        P0, P1, P2, P3, P4, P5 = P
        value = ccf_simulator(b1, b2, b3, P0, P1, P2, P3,
                              P4, P5, n_rep, method, N=500)[0]

        return value

    def evaluate_true(self, xx):

        b1 = xx[0]
        b2 = xx[1]
        b3 = xx[2]
        P0, P1, P2, P3, P4, P5 = set_input_distributions(self.true_dist)
        value = ccf_simulator(b1, b2, b3, P0, P1, P2, P3,
                              P4, P5, 100, "parametric", N=5000)[0]

        return value

    def estimate_minimizer_minimum(self, n_sample=1000):

        np.random.seed(None)

        if self.sample_space is None:
            lhd = lhs(self.dimension_x, samples=n_sample, criterion="maximin")
            X_ = np.zeros((n_sample, self.dimension_x))
            for i in range(self.dimension_x):
                X_[:, i] = lhd[:, i]*(self.ub[i]-self.lb[i]) + self.lb[i]
        else:
            X_ = self.sample_space

        Y_ = np.zeros(X_.shape[0])
        for i in range(X_.shape[0]):
            Y_[i] = self.evaluate_true(X_[i])
        self.f_star = min(Y_)
        self.x_star = X_[np.argmin(Y_)]

        return self.f_star, self.x_star


def ccf_simulator(B1, B2, B3, P0, P1, P2, P3, P4, P5, n_rep=1, method='NBRO',  N=500):

    # P0, P1, P2, P3, P4, P5 = P
    output = np.zeros(n_rep)
    for i in np.arange(n_rep):
        Inform, output[i], _, _ = CriticalCareFacility(
            B1, B2, B3, P0, P1, P2, P3, P4, P5, method,  N)
    output_mean = np.mean(output)
    return output_mean, Inform


def CriticalCareFacility(B1, B2, B3, P0, P1, P2, P3, P4, P5, method, N, BurnInTime=300):

    if method == 'NBRO':  # "non-parametric"

        p = P0[0]
        theta = P0[1]
        PatientCat = np.random.choice(theta, size=N, p=p)

        p = P1[0]
        theta = P1[1]
        ArrivalTime = np.random.choice(theta, size=N, p=p)

    elif method == 'hist':
        PatientCat = np.random.choice(P0, size=N)
        ArrivalTime = np.random.choice(P1, size=N)

    elif method == "parametric":
        p, theta = P0
        PatientCat = np.random.choice(theta, size=N, p=p)
        ArrivalTime = P1.rvs(size=N)

    else:
        print('Wriong input!')

    ArrivalTime = np.cumsum(ArrivalTime)
    Inform = pd.DataFrame(
        {"ArrivalTime": ArrivalTime, "PatientCat": PatientCat})

    if ArrivalTime[-1] > 300:
        BurnIn = Inform[Inform['ArrivalTime'] > 300].index[0]
    else:
        BurnIn = int(N/2)

    if ArrivalTime[-1] > 600:
        N = Inform[Inform['ArrivalTime'] > 600].index[0]
        Inform = Inform.iloc[:N, :]

    # print(BurnIn,Inform.shape[0],ArrivalTime[BurnIn-1],ArrivalTime[N-1]- ArrivalTime[BurnIn-1])

    Inform['StayTimeInICU'] = 0
    Inform['StayTimeInCCU'] = 0
    Inform['StayTimeInIICU'] = 0
    Inform['StayTimeInICCU'] = 0
    Inform['Status'] = 0
    Inform['DischargeTime'] = 0

    if method == "NBRO":  # "non-parametric"

        p = P3[0]
        theta = P3[1]
        Inform.loc[Inform['PatientCat'] == 1, 'StayTimeInICU'] = np.random.choice(
            theta, size=sum(Inform['PatientCat'] == 1), p=p)
        Inform.loc[Inform['PatientCat'] == 2, 'StayTimeInICU'] = np.random.choice(
            theta, size=sum(Inform['PatientCat'] == 2), p=p)

        p = P2[0]
        theta = P2[1]
        Inform.loc[Inform['PatientCat'] == 2, 'StayTimeInIICU'] = np.random.choice(
            theta, size=sum(Inform['PatientCat'] == 2), p=p)

        p = P4[0]
        theta = P4[1]
        Inform.loc[Inform['PatientCat'] == 3, 'StayTimeInCCU'] = np.random.choice(
            theta, size=sum(Inform['PatientCat'] == 3), p=p)
        Inform.loc[Inform['PatientCat'] == 4, 'StayTimeInCCU'] = np.random.choice(
            theta, size=sum(Inform['PatientCat'] == 4), p=p)

        p = P5[0]
        theta = P5[1]
        Inform.loc[Inform['PatientCat'] == 4, 'StayTimeInICCU'] = np.random.choice(
            theta, size=sum(Inform['PatientCat'] == 4), p=p)

    elif method == 'hist':
        Inform.loc[Inform['PatientCat'] == 1, 'StayTimeInICU'] = np.random.choice(
            P3, size=sum(Inform['PatientCat'] == 1))
        Inform.loc[Inform['PatientCat'] == 2, 'StayTimeInICU'] = np.random.choice(
            P3, size=sum(Inform['PatientCat'] == 2))

        Inform.loc[Inform['PatientCat'] == 2, 'StayTimeInIICU'] = np.random.choice(
            P2, size=sum(Inform['PatientCat'] == 2))

        Inform.loc[Inform['PatientCat'] == 3, 'StayTimeInCCU'] = np.random.choice(
            P4, size=sum(Inform['PatientCat'] == 3))
        Inform.loc[Inform['PatientCat'] == 4, 'StayTimeInCCU'] = np.random.choice(
            P4, size=sum(Inform['PatientCat'] == 4))

        Inform.loc[Inform['PatientCat'] == 4, 'StayTimeInICCU'] = np.random.choice(
            P5, size=sum(Inform['PatientCat'] == 4))

    elif method == 'parametric':

        Inform.loc[Inform['PatientCat'] == 1, 'StayTimeInICU'] = generate_sample_mixture_of_log_normal(
            sum(Inform['PatientCat'] == 1), P3)
        Inform.loc[Inform['PatientCat'] == 2, 'StayTimeInICU'] = generate_sample_mixture_of_log_normal(
            sum(Inform['PatientCat'] == 2), P3)
        Inform.loc[Inform['PatientCat'] == 2, 'StayTimeInIICU'] = generate_sample_mixture_of_log_normal(
            sum(Inform['PatientCat'] == 2), P2)
        Inform.loc[Inform['PatientCat'] == 3, 'StayTimeInCCU'] = generate_sample_mixture_of_log_normal(
            sum(Inform['PatientCat'] == 3), P4)
        Inform.loc[Inform['PatientCat'] == 4, 'StayTimeInCCU'] = generate_sample_mixture_of_log_normal(
            sum(Inform['PatientCat'] == 4), P4)
        Inform.loc[Inform['PatientCat'] == 4, 'StayTimeInICCU'] = generate_sample_mixture_of_log_normal(
            sum(Inform['PatientCat'] == 4), P5)

    else:
        print('Wriong input!')

    Inform['HostipalTime'] = Inform['StayTimeInICU'] + Inform['StayTimeInCCU'] + \
        Inform['StayTimeInIICU'] + Inform['StayTimeInICCU']
    Inform['HostipalTimeInICUCCU'] = Inform['StayTimeInICU'] + \
        Inform['StayTimeInCCU']

    if (Inform.loc[0, 'PatientCat'] == 1) or (Inform.loc[0, 'PatientCat'] == 2):
        Inform.loc[0, 'Status'] = "ICU"
    else:
        Inform.loc[0, 'Status'] = "CCU"

    # N1 = sum(Inform['Status'] == 'ICU')
    # N2 = sum(Inform['Status'] == 'CCU')
    # N3 = sum(Inform['Status'] == 'IC')
    # print(N1,N2,N3)

    for i in range(1, N):

        # Calculate remaining hospital time
        # print(i)
        T = Inform['ArrivalTime'][i] - Inform['ArrivalTime'][i-1]
        # print("InterTime", T)

        Inform.loc[(Inform.index < i) & (Inform['Status']
                                         != "Discharge"), 'HostipalTime'] -= T
        Inform.loc[(Inform.index < i) & (Inform['Status'] !=
                                         "Discharge"), 'HostipalTimeInICUCCU'] -= T
        # Inform.head(20)

        # Discharge patients
        # print("# of discharege", sum((Inform.index < i) & (Inform['HostipalTime'] <= 0)&(Inform['Status']!= "Discharge")))
        Inform.loc[(Inform.index < i) & (Inform['HostipalTime'] <= 0) & (
            Inform['Status'] != "Discharge"), "DischageTime"] = i
        Inform.loc[(Inform.index < i) & (Inform['HostipalTime'] <= 0) & (
            Inform['Status'] != "Discharge"), "Status"] = "Discharge"
        # Inform.head(20)

        # Move suitable patients to IC
        B3a = int(B3) - sum(Inform['Status'] == 'IC')
        B3n = sum((Inform['HostipalTimeInICUCCU'] < 0) & (
            Inform['HostipalTime'] > 0) & (Inform['Status'] != "IC"))
        # print("IC bed Available", B3a)
        # print("IC beds needed", B3n)
        if B3a >= B3n:
            Inform.loc[(Inform['HostipalTimeInICUCCU'] < 0) & (
                Inform['HostipalTime'] > 0) & (Inform['Status'] != "IC"), "Status"] = "IC"
        else:
            index = Inform.loc[(Inform['HostipalTimeInICUCCU'] < 0) & (Inform['HostipalTime'] > 0) & (
                Inform['Status'] != "IC"),].sort_values("HostipalTimeInICUCCU").index.values[:B3a]
            Inform.loc[index, "Status"] = "IC"
        # Inform.head(20)

        # New Patient admission
        if (Inform.loc[i, 'PatientCat'] == 1) or (Inform.loc[i, 'PatientCat'] == 2):
            if sum(Inform['Status'] == 'ICU') < B1:
                Inform.loc[i, 'Status'] = "ICU"
            else:
                Inform.loc[i, 'Status'] = "TurnAway"
                Inform.loc[i, 'HostipalTimeInICUCCU'] = np.nan
                Inform.loc[i, 'HostipalTime'] = np.nan
        else:
            if sum(Inform['Status'] == 'CCU') < B2:
                Inform.loc[i, 'Status'] = "CCU"
            else:
                Inform.loc[i, 'Status'] = "TurnAway"
                Inform.loc[i, 'HostipalTimeInICUCCU'] = np.nan
                Inform.loc[i, 'HostipalTime'] = np.nan

        N1 = sum(Inform['Status'] == 'ICU')
        N2 = sum(Inform['Status'] == 'CCU')
        N3 = sum(Inform['Status'] == 'IC')
        # print("# of patients in ICU,CCU and IC",N1,N2,N3)

        TurnAwayPerDay = sum(Inform.loc[BurnIn:, 'Status'] == "TurnAway")/(
            Inform.loc[N-1, 'ArrivalTime'] - Inform.loc[BurnIn-1, 'ArrivalTime'])
        TurnAwayRate = sum(
            Inform.loc[BurnIn:, 'Status'] == "TurnAway")/(N-BurnIn)
        TurnAwayRate2 = sum(Inform['Status'] == "TurnAway")/N

    return Inform, TurnAwayPerDay, TurnAwayRate, TurnAwayRate2


# #Validation of the simulator
# s = 2.20849609
# S = 2.30601563
# P = 0.0002
# n_rep = 10
# is_nonparametric = 'noparametric'
# cost_simulator(s,S,P,n_rep,is_nonparametric)
# inventory(s,S,P)


class DistributionKernel(Kern):
    def __init__(self, input_dim, variance=1., lengthscale1=1., lengthscale2=1., power=1.0, active_dims=None):
        super(DistributionKernel, self).__init__(
            input_dim, active_dims, 'distribution')
        # assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.variance = Param('variance', variance)
        self.lengthscale1 = Param('lengthscale1', lengthscale1)
        self.lengthscale2 = Param('lengthscale2', lengthscale2)  # ,Logexp()
        self.power = Param('power', power)
        self.link_parameters(
            self.variance, self.lengthscale1, self.lengthscale2)

    def K(self, X, X2):
        if X2 is None:
            X2 = X

        # print(X.shape)
        diff1 = np.expand_dims(X[:, 0:2], 0) - np.expand_dims(X2[:, 0:2], 1)
        dist1 = np.sum(diff1**2, axis=-1).T
        correlation1 = np.exp(-dist1/(2*self.lengthscale1))

        k = int((X.shape[1]-2)/2)
        # print(k)
        Distribution_p1 = np.ascontiguousarray(X[:, 2:(k+2)])
        Distribution_theta1 = np.ascontiguousarray(X[:, (k+2):])/1000
        Distribution_p2 = np.ascontiguousarray(X2[:, 2:(k+2)])
        Distribution_theta2 = np.ascontiguousarray(X2[:, (k+2):])/1000
        N1 = X.shape[0]
        N2 = X2.shape[0]
        dist2 = np.zeros((N1, N2))

        for i1 in range(N1):
            for i2 in range(N2):
                # print(i1,i2)
                dist2[i1, i2] = ot.wasserstein_1d(u_values=Distribution_theta1[i1, :],
                                                  v_values=Distribution_theta2[i2, :],
                                                  u_weights=Distribution_p1[i1, :],
                                                  v_weights=Distribution_p2[i2, :], p=2)

        dist2p = np.power(dist2, self.power)
#         diff2 = np.expand_dims(X[:,2:],0)- np.expand_dims(X2[:,2:],1)
#         dist2 = np.sum(diff2**2,axis = -1)
        correlation2 = np.exp(-dist2p/(2*self.lengthscale2))

        return self.variance * correlation1 * correlation2

    def Kdiag(self, X):
        return self.variance*np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):

        if X2 is None:
            X2 = X

        diff1 = np.expand_dims(X[:, 0:2], 0) - np.expand_dims(X2[:, 0:2], 1)
        dist1 = np.sum(diff1**2, axis=-1).T
        correlation1 = np.exp(-dist1/(2*self.lengthscale1))

        k = int((X.shape[1]-2)/2)
        Distribution_p1 = np.ascontiguousarray(X[:, 2:(k+2)])
        Distribution_theta1 = np.ascontiguousarray(X[:, (k+2):])/1000
        Distribution_p2 = np.ascontiguousarray(X2[:, 2:(k+2)])
        Distribution_theta2 = np.ascontiguousarray(X2[:, (k+2):])/1000
        N1 = X.shape[0]
        N2 = X2.shape[0]
        dist2 = np.zeros((N1, N2))

        for i1 in range(N1):
            for i2 in range(N2):
                dist2[i1, i2] = ot.wasserstein_1d(u_values=Distribution_theta1[i1, :],
                                                  v_values=Distribution_theta2[i2, :],
                                                  u_weights=Distribution_p1[i1, :],
                                                  v_weights=Distribution_p2[i2, :], p=2.0)


#         diff2 = np.expand_dims(X[:,2:],0)- np.expand_dims(X2[:,2:],1)
#         dist2 = np.sum(diff2**2,axis = -1)

        dist2p = np.power(dist2, self.power)
        correlation2 = np.exp(-dist2p/(2*self.lengthscale2))

        dvar = correlation1 * correlation2
        dl1 = self.variance * correlation1 * \
            correlation2 * 0.5 * dist1/(self.lengthscale1**2)
        dl2 = self.variance * correlation1 * correlation2 * \
            0.5 * dist2p/(self.lengthscale2**2)
        # dpower = self.variance * correlation1 * correlation2 * (-0.5)/self.lengthscale2 * dist2p * np.log(dist2)

        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale1.gradient = np.sum(dl1*dL_dK)
        self.lengthscale2.gradient = np.sum(dl2*dL_dK)
        # self.power.gradient = np.sum(dpower*dL_dK)


def Mu_G(x, gp, posterior):

    DistributionVector = distribution_to_vector(posterior)
    N_MC = DistributionVector.shape[0]
    D = np.array([np.concatenate((i, j), axis=0)
                 for i in x for j in DistributionVector])
    psi = gp.predict(D, full_cov=False)[0]
    mu = np.average(psi.reshape((x.shape[0], N_MC)), axis=1)

    return mu


def Mu_Std_G(x, gp, posterior):

    DistributionVector = distribution_to_vector(posterior)
    N_MC = DistributionVector.shape[0]
    mu = np.zeros(x.shape[0])
    sigma = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        D1 = np.array([np.concatenate((x[i, :], j), axis=0)
                      for j in DistributionVector])
        psi, k = gp.predict(D1, full_cov=True)
        sigma[i] = np.average(k)
        mu[i] = np.average(psi)
    sigma[sigma < 0] = 0
    std = np.sqrt(sigma)

    return mu, std


def Sigma_tilde(D, gp, posterior, dimension_x):
    # changed

    DistributionVector = distribution_to_vector(posterior)
    N_MC = DistributionVector.shape[0]
    Dim = D.shape[1]

    x = D[:, 0:dimension_x]
    Z = np.empty([N_MC, Dim])
    Z[:, 0:dimension_x] = x
    Z[:, dimension_x:] = DistributionVector

    # Z = np.concatenate((D,Z))
    # _, k = gp.predict(Z,full_cov=True,include_likelihood = True)
    # diag = k[0,0]
    # Sigma2_tilde1 = np.power(np.average(k[0,1:])/(np.sqrt(diag) + np.finfo(float).eps),2)
    # Sigma2_tilde2 = np.average(k[1:,1:] - k[0,1:].reshape(-1,1) @ k[0,1:].reshape(1,-1)/ k[0,0])
    # Sigma2_tilde = np.sqrt(Sigma2_tilde1+Sigma2_tilde2)

    mean, variance = gp.predict(D)
    print("mean, variance:", mean, variance)
    fitted_kernel = gp.kern
    covariance_matrix = fitted_kernel.K(D, Z)
    Sigma2_tilde = np.mean(covariance_matrix) / \
        (np.sqrt(variance) + np.finfo(float).eps)
    print("covariance mean", np.mean(covariance_matrix), "std", np.sqrt(variance),
          np.max(covariance_matrix),  np.min(covariance_matrix))
    # print(covariance_matrix)

    print("estimated std", Sigma2_tilde)
    return Sigma2_tilde


def EGO_PB(gp, posterior, X_update, D_, dimension_x, disp_plots=False):
    print("fitted_kernel")
    print(gp.kern)
    mu = Mu_G(X_update, gp, posterior)
    mu_min = np.min(mu)
    EGO_crit = np.empty(D_.shape[0])
    for i in range(D_.shape[0]):
        Sigma = Sigma_tilde(D_[i:(i+1)], gp, posterior, dimension_x)
        mu_hat = Mu_G(D_[i:(i+1)][:, 0:dimension_x], gp, posterior)
        delta = mu_min - mu_hat
        value = delta * \
            scipy.stats.norm.cdf(delta/Sigma) + Sigma * \
            scipy.stats.norm.pdf(delta/Sigma)
        print("EI_component", delta, scipy.stats.norm.cdf(
            delta/Sigma), Sigma, scipy.stats.norm.pdf(delta/Sigma))
        EGO_crit[i] = value
        print("EI", value)
    D_new = D_[np.argmax(EGO_crit), :]
    print("D_new", D_new)

    # Create a figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(D_[:,0], D_[:,1], EGO_crit, c='r', marker='o')
    # ax.set_title('EI')
    # ax.set_xlabel('s')
    # ax.set_ylabel('S')
    # ax.set_zlabel('EI')
    # plt.savefig('3d_scatter_plot.png')
    # plt.show()

    return D_new


def update_data_PB(D_new, D_old, Y_old, n_rep, method, n_k, f, dimension_x, dimension_p):
    # changed, add dimension_p
    # s = D_new[0]
    # S = D_new[1]
    # P = (D_new[2:(2+k)],D_new[(2+k):])
    # Y_new = np.atleast_2d(cost_simulator(s,S,P,n_rep,method)[0])

    xx = D_new[:dimension_x]
    P = []
    for i in range(dimension_p):
        P.append(
            (D_new[(dimension_x+2*i*n_k):(dimension_x+(2*i+1)*n_k)],
             D_new[(dimension_x+(2*i+1)*n_k):(dimension_x+(2*i+2)*n_k)])
        )

    Y_new = np.atleast_2d(f.evaluate(xx, P, n_rep, method))
    Y_update = np.concatenate((Y_old, Y_new))

    D_new = np.atleast_2d(D_new)
    D_update = np.concatenate((D_old, D_new), axis=0)

    return D_update, Y_update


def EGO(X_, X_update, mu, sigma, gp):
    mu_AEI = gp.predict(X_update, full_cov=False, include_likelihood=False)[0]
    mu_min = np.min(mu_AEI)
    EGO_crit = np.empty(X_.shape[0])
    sigma[sigma == 0] = np.finfo(float).eps
    delta = mu_min - mu
    EGO_crit = delta * \
        scipy.stats.norm.cdf(delta/sigma) + sigma * \
        scipy.stats.norm.pdf(delta/sigma)
    index = np.argmax(EGO_crit)
    X_new = X_[index]
    return X_new


def update_data(x_new, D_old, Y_old, P, n_rep, method, f):

    # s = x_new[0]
    # S = x_new[1]
    # Y_new = np.atleast_2d(cost_simulator(s,S,P,n_rep,method)[0])
    Y_new = np.atleast_2d(f.evaluate(x_new, P, n_rep, method))

    Y_update = np.concatenate((Y_old, Y_new), axis=0)
    D_new = np.atleast_2d(x_new)
    D_update = np.concatenate((D_old, D_new), axis=0)

    return D_update, Y_update


def distribution_to_vector(distribution):
    p_all, theta_all = distribution[0], distribution[1]
    N = p_all.shape[1]
    dimension_p = p_all.shape[0]
    DistributionList = list()
    for i in range(N):
        distribution_row_list = list()
        for p in range(dimension_p):
            value = np.concatenate(
                (p_all[p, i, :].reshape(-1), theta_all[p, i, :].reshape(-1)))
            distribution_row_list.append(value)
        DistributionList.append(np.array(distribution_row_list).reshape(-1))
    return np.array(DistributionList)


def inital_design(n_sample, seed, lower_bound, upper_bound):

    np.random.seed(seed)
    dimension_x = len(lower_bound)
    lhd = lhs(dimension_x, samples=n_sample, criterion="maximin")
    D1 = np.zeros((n_sample, dimension_x))
    for i in range(dimension_x):
        D1[:, i] = lhd[:, i]*(upper_bound[i]-lower_bound[i]) + lower_bound[i]
    return D1


def log_normal_parameter(mu, sigma):
    s = np.sqrt(np.log(sigma**2/mu**2+1))
    scale = np.exp(np.log(mu) - s**2/2)
    return s, scale


def real_world_data(xi_n, seed, dist):
    # changed real_world_data, dist becomes a list
    np.random.seed(seed)
    dimension_p = len(dist)
    xi = np.zeros((dimension_p, xi_n))
    for p in range(dimension_p):
        if (dist[p].get("dist") == "exp"):
            xi[p, :] = set_generate_exponential(xi_n, dist[p].get("rate"))
        elif (dist[p].get("dist") == "lognorm"):
            xi[p, :] = set_generate_mixture_of_log_normal(xi_n, mu=dist[p].get(
                "mu"), sigma=dist[p].get("sigma"), p_i=dist[p].get("p_i", [1]))
        elif (dist[p].get("dist") == "m-lognorm"):
            xi[p, :] = set_generate_mixture_of_log_normal(xi_n, mu=dist[p].get(
                "mu"), sigma=dist[p].get("sigma"), p_i=dist[p].get("p_i"))
        elif (dist[p].get("dist") == "discrete"):
            xi[p, :] = set_generate_discrete(xi_n, p_i=dist[p].get("p_i"))

    return xi


def set_input_distributions(dist):
    input_distribution = list()
    for p in range(len(dist)):
        if (dist[p].get("dist") == "exp"):
            input_distribution.append(
                set_exponential(dist[p].get("rate"))
            )
        elif (dist[p].get("dist") == "lognorm"):
            input_distribution.append(
                set_mixture_of_log_normal(mu=dist[p].get("mu"), sigma=dist[p].get(
                    "sigma"), p_i=dist[p].get("p_i", [1]))
            )
        elif (dist[p].get("dist") == "m-lognorm"):
            input_distribution.append(
                set_mixture_of_log_normal(mu=dist[p].get(
                    "mu"), sigma=dist[p].get("sigma"), p_i=dist[p].get("p_i"))
            )
        elif (dist[p].get("dist") == "discrete"):
            input_distribution.append(
                set_discrete(dist[p].get("p_i"))
            )
    return input_distribution


def fit_discrete(xi):
    import collections
    counter = collections.Counter(xi)
    theta = np.array(list(counter.keys()))
    p = list(counter.values())/sum(np.array(list(counter.values())))
    return (p, theta)


def fit_exponential(xi):
    _, scale = stats.expon.fit(xi, floc=0)
    rv = stats.expon(scale=scale)
    return rv


def fit_log_normal(xi):
    s, _, scale = stats.lognorm.fit(xi, floc=0)
    rv = stats.lognorm(s=s, scale=scale)
    return ([1], rv)


def set_discrete(p_i):
    theta = np.arange(1, len(p_i)+1)
    return (np.array(p_i), theta)


def set_exponential(rate):
    rv = stats.expon(scale=1/rate)
    return rv


def set_mixture_of_log_normal(mu, sigma, p_i=[1], plots=False):

    if not isinstance(mu, List):
        mu = [mu]
        sigma = [sigma]
        p_i = [1]

    mixture_n = len(mu)

    s_all = []
    scale_all = []
    distributions = [p_i]
    for i in range(mixture_n):
        s, scale = log_normal_parameter(mu[i], sigma[i])
        s_all.append(s)
        scale_all.append(scale)
        distributions.append(stats.lognorm(s=s_all[i], scale=scale_all[i]))

    if plots:
        for i in range(mixture_n):
            x = np.linspace(stats.lognorm.ppf(0.01, s=s_all[i], scale=scale_all[i]), stats.lognorm.ppf(
                0.99, s=s_all[i], scale=scale_all[i]), 100)
            # fig, ax = plt.subplots(1, 1)
            plt.plot(x, stats.lognorm.pdf(
                x, s=s_all[i], scale=scale_all[i]), 'r-', lw=5, alpha=0.6, label='lognorm pdf')
            mean, var, skew, kurt = stats.lognorm.stats(
                s=s_all[i], scale=scale_all[i], moments='mvsk')
            print("mean:", mean, "std:", np.sqrt(var))
        plt.show()

        x = np.linspace(0, x.max(), 100)
        density = [
            p_i[i]*stats.lognorm.pdf(x, s=s_all[i], scale=scale_all[i]) for i in range(mixture_n)]
        density = np.array(density).sum(axis=0)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, density, 'r-', lw=5, alpha=0.6, label='lognorm pdf')
        plt.show()

    return distributions


def generate_sample_discrete(xi_n, rv):
    p, theta = rv
    xi = np.random.choice(theta, size=xi_n, p=p)
    return xi


def generate_sample_general(xi_n, rv):
    xi = rv.rvs(size=(xi_n))
    return xi


def generate_sample_mixture_of_log_normal(xi_n, rv):
    p_i = rv[0]
    rv = rv[1:]
    xi = np.zeros(xi_n)
    for i in range(xi_n):
        z_i = np.argmax(np.random.multinomial(1, p_i))
        xi[i] = rv[z_i].rvs(size=1)
    return xi


def fit_and_generate_exponential(xi, size=1000):

    rv = fit_exponential(xi)
    xi = generate_sample_general(size, rv)

    return xi


def fit_and_generate_log_normal(xi, size=1000):

    _, rv = fit_log_normal(xi)
    xi = generate_sample_general(size, rv)

    return xi

    # if disp_plots:
    #     plt.hist(xi, bins=100, density=True, alpha = 0.3)
    #     plt.hist(xi_generate, bins=100, density=True, alpha = 0.3)
    #     x = np.linspace(stats.lognorm.ppf(0.01,s=s,scale=scale),stats.lognorm.ppf(0.99,s=s,scale=scale), 100)
    #     plt.plot(x, stats.lognorm.pdf(x, s=s,scale=scale), 'r-', lw=5, alpha=0.6, label='lognorm pdf')
    #     mean, var, skew, kurt = stats.lognorm.stats(s=s,scale=scale,moments='mvsk')
    #     print("mean:",mean,"std:",np.sqrt(var))
    #     plt.show();


def set_generate_discrete(xi_n, p_i):
    rv = set_discrete(p_i)
    xi = generate_sample_discrete(xi_n, rv)
    return xi


def set_generate_exponential(xi_n, rate):
    rv = set_exponential(rate)
    xi = generate_sample_general(xi_n, rv)
    return xi


def set_generate_mixture_of_log_normal(xi_n, mu=[5000,  10000], sigma=[5000, 5000], p_i=[0.5, 0.5], plots=False):
    rv = set_mixture_of_log_normal(mu, sigma, p_i, plots)
    xi = generate_sample_mixture_of_log_normal(xi_n, rv)
    if plots:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(xi, bins=100, density=True)
        plt.show()
    return xi


def GP_model(D, Y, method):

    if method == "NBRO":
        kernel = DistributionKernel(
            input_dim=D.shape[1], variance=1., lengthscale1=1., lengthscale2=1., power=1)
        gp = GPy.models.GPRegression(D, Y, kernel)  # , normalizer=True)
        gp[''].constrain_positive()
        # gp['distribution.power'].constrain_bounded(0,2)
        # gp['distribution.lengthscale1'].constrain_bounded(0,100)
        # gp['distribution.lengthscale2'].constrain_bounded(0,100)

        # power = 2 will give some numerical issue,
        # display(gp)
        gp.optimize(messages=False)
        gp.optimize_restarts(num_restarts=10, messages=False, verbose=False)
        # display(gp)
        # fig = gp.plot()
    else:
        kernel = GPy.kern.RBF(
            input_dim=D.shape[1], variance=1., lengthscale=1.)
        gp = GPy.models.GPRegression(D, Y, kernel)
        # display(gp)
        gp.optimize(messages=False)
        gp.optimize_restarts(num_restarts=10, messages=False, verbose=False)
        # display(gp)
        # fig = gp.plot()

    return gp


def stick_breaking(alpha, n_k):

    # betas = beta(1, alpha, n_k)
    # remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    # p = betas * remaining_pieces
    # return p/p.sum()
    betas = beta(1, alpha, n_k-1)
    remaining_pieces = np.append(1, np.cumprod(1 - betas))
    p = betas * (remaining_pieces[:-1])
    p = np.concatenate((p, [remaining_pieces[-1]]), axis=0)

    return p/p.sum()


def dirichlet_process_posterior(P0, n_k, alpha, observations):
    n = len(observations)
    theta = np.empty(n_k)
    for i in range(n_k):
        if np.random.random() < (1. * alpha / (alpha + n)):
            theta[i] = P0.rvs(size=1)  # P0.rvs(0,40000,size = 1)
        else:
            theta[i] = np.random.choice(observations, size=1)
    return theta


def posterior_generation(xi, N_MC, n_k, dimension_p, P0_list, alpha=10):
    # changed add dimension_p, P0_list

    p_all = np.zeros((dimension_p, N_MC, n_k))
    theta_all = np.zeros((dimension_p, N_MC, n_k))

    for p in range(dimension_p):
        if P0_list[p] is None:
            P0 = stats.uniform(loc=min(xi[p, :]), scale=max(xi[p, :]))
        else:
            P0 = P0_list[p]
        for i in range(N_MC):
            p_all[p, i, :] = stick_breaking(alpha, n_k)  # weight
            theta_all[p, i, :] = dirichlet_process_posterior(
                P0, n_k, alpha, xi[p, :])  # corresponding sampels

    return (p_all, theta_all)


def candidate_space(xi, lower_bound, upper_bound, N_candidate, n_k, dimension_p, P0_list, seed=None, alpha=10):

    X_ = inital_design(N_candidate, seed, lower_bound, upper_bound)

    posterior_candidate = posterior_generation(
        xi, N_candidate, n_k, dimension_p, P0_list, alpha)
    P_ = distribution_to_vector(posterior_candidate)
    # np.array([np.concatenate((i,j),axis = 0) for i in X_ for j in Y_])
    D_ = np.concatenate((X_, P_), axis=1)
    # print(X_.shape,P_.shape,D_.shape)

    return X_, D_


def results_save(file_pre, results, observations):

    file_name = file_pre + '_results.csv'
    if os.path.exists(file_name):
        results.to_csv(file_name, header=False, index=False, mode='a')
    else:
        results.to_csv(file_name, header=True, index=False)

    file_name = file_pre + '_observations.csv'
    if os.path.exists(file_name):
        observations.to_csv(file_name, header=False, index=False, mode='a')
    else:
        observations.to_csv(file_name, header=True, index=False)


def results_plot(GAP, n_sample, iteration):

    # Plots
    SimRegret = list()
    for i in range(len(GAP)):
        SimRegret.append(np.min(GAP[:(i+1)]))
    SimRegret = np.array(SimRegret).reshape(-1, 1)
    # AvgRegret = np.cumsum(GAP)/np.arange(1,(len(GAP)+1)).reshape(-1,1)
    CumRegret = np.cumsum(GAP)
    AvgRegret = np.cumsum(GAP)/np.arange(1, GAP.shape[0]+1)

    Budget = range(n_sample, n_sample + iteration + 1)
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 2, 1)
    plt.plot(Budget, CumRegret)
    plt.ylabel('CumRegret')
    plt.xlabel('Budget')
    plt.axhline(0, color='black', ls='--')
    # plt.xlim(n_sample,N_total)
    # plt.ylim(-0.1,4)
    ####################################
    plt.subplot(2, 2, 2)
    plt.plot(Budget, AvgRegret)
    plt.ylabel('AvgRegret')
    plt.xlabel('Budget')
    plt.axhline(0, color='black', ls='--')
    ####################################
    plt.subplot(2, 2, 3)
    plt.plot(Budget, SimRegret)
    plt.ylabel('SimRegret')
    plt.xlabel('Budget')
    plt.axhline(0, color='black', ls='--')
    ####################################
    plt.subplot(2, 2, 4)
    plt.plot(Budget, GAP)
    plt.ylabel('GAP')
    plt.xlabel('Budget')
    plt.axhline(0, color='black', ls='--')
    plt.show()


def output(file_pre, Y_update, D_update, minimizer, minimum_value, TIME_RE, f, x_star, f_star, seed, n_sample, iteration, method, n_xi, dimension_p=None, posterior=None, n_k=None, get_g_hat_x=False):

    # Prepare for the output of the results
    minimizer = np.array(minimizer)
    minimum_value = np.array(minimum_value)

    f_hat_x = np.zeros(minimizer.shape[0])
    for index in range(minimizer.shape[0]):
        f_hat_x[index] = f.evaluate_true(minimizer[index])

    if x_star is not None:
        xGAP = np.linalg.norm(minimizer - np.atleast_2d(x_star), axis=1)
    if f_star is not None:
        GAP = f_hat_x - f_star

    # Results save
    TIME_RE = np.array([np.nan] + TIME_RE)
    results = pd.DataFrame(minimizer)
    results.columns = ['x_' + str(i) for i in range(minimizer.shape[1])]
    results['f_hat_x'] = f_hat_x
    results['minimum_value'] = minimum_value
    results['iteration'] = np.arange(n_sample, n_sample + iteration + 1)
    results['time'] = TIME_RE
    results['seed'] = seed
    results['n_xi'] = n_xi

    if get_g_hat_x:
        g_hat_x = np.zeros(minimizer.shape[0])
        for index in range(minimizer.shape[0]):
            g_hat_x[index] = g(minimizer[index], posterior,
                               f, n_k, 'expectation')
        results['g_hat_x'] = g_hat_x

    observations = pd.DataFrame(np.concatenate((Y_update, D_update), axis=1))
    if method == "NBRO":
        observations.columns = ['y'] + ['x_' + str(i) for i in range(minimizer.shape[1])] + sum([['p_' + str(p) + '_' + str(
            i) for i in range(n_k)] + ['theta_' + str(p) + '_' + str(i) for i in range(n_k)] for p in range(dimension_p)], [])
    else:
        observations.columns = ['y'] + ['x_' +
                                        str(i) for i in range(minimizer.shape[1])]

    observations['seed'] = seed
    observations['n_xi'] = n_xi

    print(min(f_hat_x))

    results_save(file_pre, results, observations)

    results_plot(f_hat_x, n_sample, iteration)


# g(x)=E[f(x,\lambda)] is the objective function that is unknown
def g(x, posterior, f, n_k,  method='all', alpha=0.5, n_rep=100):

    values = []

    p_all, theta_all = posterior
    # print(p_all, theta_all)
    dimension_p = p_all.shape[0]
    size = p_all.shape[1]

    for i in range(size):
        P = []
        for p in range(dimension_p):
            P.append(
                (p_all[p, i, (2*p*n_k):((2*p+1)*n_k)],
                 theta_all[p, i, (2*p*n_k):(2*p+1)*n_k])
            )
        # print(P)
        value = f.evaluate(x, P, n_rep, 'NBRO')
        values.append(value)
    values = np.array(values)

    expectation = np.mean(values)

    if method == 'all':

        variance = np.std(values)
        mean_variance = expectation + 0.01*variance
        value_at_risk_1 = np.quantile(values, 1 - 0.1)
        value_at_risk_2 = np.quantile(values, 1 - 0.2)
        value_at_risk_3 = np.quantile(values, 1 - 0.3)
        value_at_risk_5 = np.quantile(values, 1 - 0.5)
        conditional_value_at_risk_1 = np.mean(
            values[values >= value_at_risk_1])
        conditional_value_at_risk_2 = np.mean(
            values[values >= value_at_risk_2])
        conditional_value_at_risk_3 = np.mean(
            values[values >= value_at_risk_3])
        conditional_value_at_risk_5 = np.mean(
            values[values >= value_at_risk_5])

        output = pd.DataFrame({
            'expectation': [expectation],
            'variance': [variance],
            'mean_variance': [mean_variance],
            'value_at_risk_1': [value_at_risk_1],
            'value_at_risk_2': [value_at_risk_2],
            'value_at_risk_3': [value_at_risk_3],
            'value_at_risk_5': [value_at_risk_5],
            'conditional_value_at_risk_1': [conditional_value_at_risk_1],
            'conditional_value_at_risk_2': [conditional_value_at_risk_2],
            'conditional_value_at_risk_3': [conditional_value_at_risk_3],
            'conditional_value_at_risk_5': [conditional_value_at_risk_5],
        })

        return output

    else:
        return expectation


def g_vector(X, posterior, f, n_k):
    N = X.shape[0]
    value_vector = pd.DataFrame()
    for i in range(N):
        value_vector = pd.concat(
            [value_vector, g(X[i, :], posterior, f, n_k)], ignore_index=True)
    return value_vector


def kl_divergence(p, q):
    """
    Calculate the KL divergence between two probability distributions.  \sum P(x) log P(x)/Q(x)

    Parameters:
    - p: array-like, first probability distribution
    - q: array-like, second probability distribution

    Returns:
    - kl_div: float, the KL divergence
    """
    return np.sum(rel_entr(p, q))


def calculate_ambiguity_set_size(empirical_data, quantiles, size=5000):

    # state = np.random.get_state()

    # np.random.seed(0)
    # Calculate empirical probabilities
    unique, counts = np.unique(empirical_data, return_counts=True)
    empirical_probs = counts / counts.sum()

    distance = np.zeros((size, 3))
    for i in range(size):
        epsilon_perturbed = np.array(np.random.normal(0, 0.1, unique.shape[0]))
        epsilon_perturbed = np.clip(epsilon_perturbed, -0.5, 0.5)
        perturbed_probs = empirical_probs * (1+epsilon_perturbed)
        perturbed_probs = np.clip(perturbed_probs, 0, 1)
        perturbed_probs /= np.sum(perturbed_probs)  # Normalize to sum to 1
        distance[i, 0] = ot.wasserstein_1d(u_values=unique,
                                           v_values=unique,
                                           u_weights=empirical_probs,
                                           v_weights=perturbed_probs,
                                           p=1)
        distance[i, 1] = ot.wasserstein_1d(u_values=unique,
                                           v_values=unique,
                                           u_weights=empirical_probs,
                                           v_weights=perturbed_probs,
                                           p=2)
        distance[i, 2] = kl_divergence(empirical_probs, perturbed_probs)

    # Compute the quantiles for each column
    quantile_values = np.quantile(distance, quantiles, axis=0)
    quantile_values = np.round(quantile_values, 4)
    quantile_values = np.clip(quantile_values, 0.0001, None)

    print(quantile_values)

    # np.random.set_state(state)

    return quantile_values


def calculate_ambiguity_set(empirical_data, epsilon, DRO_size=20, distance='wasserstein_1'):
    """
    Calculate the ambiguity set for a discrete distribution based on the Wasserstein distance.

    Parameters:
    - empirical_data: array-like, empirical data points
    - epsilon: float, the Wasserstein distance threshold
    - distance: wasserstein_1, wasserstein_2, KL

    Returns:
    - ambiguity_set: list of tuples, each tuple contains a data point and its probability
    """

    # Calculate empirical probabilities
    unique, counts = np.unique(empirical_data, return_counts=True)
    empirical_probs = counts / counts.sum()

    # Define the ambiguity set
    ambiguity_set = []

    # For simplicity, assume the ambiguity set includes all distributions within epsilon distance
    for i in range(10000):
        # epsilon_perturbed = np.array(np.random.uniform(-0.2, 0.2,unique.shape[0]))
        epsilon_perturbed = np.array(np.random.normal(0, 0.1, unique.shape[0]))
        epsilon_perturbed = np.clip(epsilon_perturbed, -0.5, 0.5)
        perturbed_probs = empirical_probs * (1+epsilon_perturbed)
        perturbed_probs = np.clip(perturbed_probs, 0, 1)
        perturbed_probs /= np.sum(perturbed_probs)  # Normalize to sum to 1

        if distance == 'wasserstein_1':
            if ot.wasserstein_1d(u_values=unique,
                                 v_values=unique,
                                 u_weights=empirical_probs,
                                 v_weights=perturbed_probs,
                                 p=1) <= epsilon:
                ambiguity_set.append((perturbed_probs, unique))

        elif distance == 'wasserstein_2':
            if ot.wasserstein_1d(u_values=unique,
                                 v_values=unique,
                                 u_weights=empirical_probs,
                                 v_weights=perturbed_probs,
                                 p=2) <= epsilon:
                ambiguity_set.append((perturbed_probs, unique))

        elif distance == 'KL':
            if kl_divergence(empirical_probs, perturbed_probs) <= epsilon:
                ambiguity_set.append((perturbed_probs, unique))

        if len(ambiguity_set) == DRO_size:
            break

    return ambiguity_set


def output_DRO(file_pre, minimizer, minimum_value, f, x_star, f_star, seed, n_xi,
               distance, quantile_selected, epsilon
               ):

    f_hat_x = f.evaluate_true(minimizer)
    minimizer = np.atleast_2d(minimizer)

    if x_star is not None:
        xGAP = np.linalg.norm(minimizer - np.atleast_2d(x_star), axis=1)
    if f_star is not None:
        GAP = f_hat_x - f_star

    results = pd.DataFrame(minimizer)
    results.columns = ['x_' + str(i) for i in range(minimizer.shape[1])]
    results['f_hat_x'] = f_hat_x
    results['minimum_value'] = minimum_value
    results['seed'] = seed
    results['n_xi'] = n_xi
    results["distance"] = distance
    results["quantile"] = quantile_selected
    results["epsilon"] = epsilon

    file_name = file_pre + '_DRO_results.csv'
    if os.path.exists(file_name):
        results.to_csv(file_name, header=False, index=False, mode='a')
    else:
        results.to_csv(file_name, header=True, index=False)

    return results
