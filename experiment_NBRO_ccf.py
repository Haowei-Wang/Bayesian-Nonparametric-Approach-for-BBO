
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats 
import time
import scipy.stats as stats
from nbro import *
from pyDOE import *

# %%
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 999


# %%
import ot
import GPy
from GPy.kern import Kern
import numpy as np
from GPy.core.parameterization import Param
from paramz.transformations import Logexp

### total variance
class DistributionKernel(Kern):
    def __init__(self,input_dim,variance= 0.1,lengthscale=1.0, power = 1.0,active_dims=None):
        super(DistributionKernel, self).__init__(input_dim, active_dims, 'distribution')
        #assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.variance = Param('variance', variance)
        self.lengthscale1x = Param('lengthscale1x', lengthscale)
        self.lengthscale2x = Param('lengthscale2x', lengthscale)
        self.lengthscale3x = Param('lengthscale3x', lengthscale)
        self.lengthscale1p = Param('lengthscale1p', lengthscale)
        self.lengthscale2p = Param('lengthscale2p', lengthscale)
        self.lengthscale3p = Param('lengthscale3p', lengthscale)
        self.lengthscale4p = Param('lengthscale4p', lengthscale)
        self.lengthscale5p = Param('lengthscale5p', lengthscale)
        self.lengthscale6p = Param('lengthscale6p', lengthscale)
        
        self.power = Param('power', power)
        self.link_parameters(self.variance, 
                             self.lengthscale1x, self.lengthscale2x, self.lengthscale3x,
                             self.lengthscale1p, self.lengthscale2p,
                             self.lengthscale3p, self.lengthscale4p,
                             self.lengthscale5p, self.lengthscale6p)
        self.dimension_x = 3
        self.dimension_p = 6
        
    def K(self,X,X2):
        
        #print(X.shape)
        dimension_x = self.dimension_x
        dimension_p = self.dimension_p
        
        if X2 is None: 
            X2 = X
        
        ## X dimension
        diff1x = np.expand_dims(X[:,0:1],0) - np.expand_dims(X2[:,0:1],1)
        dist1x = np.sum(diff1x**2,axis = -1).T
        correlation1x = np.exp(-dist1x/(2*self.lengthscale1x))
        
        diff2x = np.expand_dims(X[:,1:2],0) - np.expand_dims(X2[:,1:2],1)
        dist2x = np.sum(diff2x**2,axis = -1).T
        correlation2x = np.exp(-dist2x/(2*self.lengthscale2x))
        
        diff3x = np.expand_dims(X[:,2:3],0) - np.expand_dims(X2[:,2:3],1)
        dist3x = np.sum(diff3x**2,axis = -1).T
        correlation3x = np.exp(-dist3x/(2*self.lengthscale3x))
        
        ## P dimension
        k = int((X.shape[1]-dimension_x)/(2*dimension_p))
        N1 = X.shape[0]
        N2 = X2.shape[0]
        
        Distribution_p1 = np.ascontiguousarray(X[:,dimension_x:(k+dimension_x)])
        Distribution_theta1 = np.ascontiguousarray(X[:,(k+dimension_x):(2*k+dimension_x)])
        Distribution_p2 = np.ascontiguousarray(X2[:,dimension_x:(k+dimension_x)])
        Distribution_theta2 = np.ascontiguousarray(X2[:,(k+dimension_x):(2*k+dimension_x)])
        dist1p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist1p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta1[i1,:], 
                                                  v_values = Distribution_theta2[i2,:], 
                                                  u_weights=Distribution_p1[i1,:], 
                                                  v_weights= Distribution_p2[i2,:], p=2)       
        dist1p = np.power(dist1p,self.power)
        correlation1p = np.exp(-dist1p/(2*self.lengthscale1p))

        Distribution_p3 = np.ascontiguousarray(X[:,(2*k+dimension_x):(3*k+dimension_x)])
        Distribution_theta3 = np.ascontiguousarray(X[:,(3*k+dimension_x):(4*k+dimension_x)])
        Distribution_p4 = np.ascontiguousarray(X2[:,(2*k+dimension_x):(3*k+dimension_x)])
        Distribution_theta4 = np.ascontiguousarray(X2[:,(3*k+dimension_x):(4*k+dimension_x)])
        dist2p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist2p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta3[i1,:], 
                                                  v_values = Distribution_theta4[i2,:], 
                                                  u_weights=Distribution_p3[i1,:], 
                                                  v_weights= Distribution_p4[i2,:], p=2)       
        dist2p = np.power(dist2p,self.power)
        correlation2p = np.exp(-dist2p/(2*self.lengthscale2p))
        
        Distribution_p5 = np.ascontiguousarray(X[:,(4*k+dimension_x):(5*k+dimension_x)])
        Distribution_theta5 = np.ascontiguousarray(X[:,(5*k+dimension_x):(6*k+dimension_x)])
        Distribution_p6 = np.ascontiguousarray(X2[:,(4*k+dimension_x):(5*k+dimension_x)])
        Distribution_theta6 = np.ascontiguousarray(X2[:,(5*k+dimension_x):(6*k+dimension_x)])
        dist3p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist3p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta5[i1,:], 
                                                  v_values = Distribution_theta6[i2,:], 
                                                  u_weights=Distribution_p5[i1,:], 
                                                  v_weights= Distribution_p6[i2,:], p=2)       
        dist3p = np.power(dist3p,self.power)
        correlation3p = np.exp(-dist3p/(2*self.lengthscale3p))
        
        
        
        Distribution_p7 = np.ascontiguousarray(X[:,(6*k+dimension_x):(7*k+dimension_x)])
        Distribution_theta7 = np.ascontiguousarray(X[:,(7*k+dimension_x):(8*k+dimension_x)])
        Distribution_p8 = np.ascontiguousarray(X2[:,(6*k+dimension_x):(7*k+dimension_x)])
        Distribution_theta8 = np.ascontiguousarray(X2[:,(7*k+dimension_x):(8*k+dimension_x)])
        dist4p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist4p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta7[i1,:], 
                                                  v_values = Distribution_theta8[i2,:], 
                                                  u_weights=Distribution_p7[i1,:], 
                                                  v_weights= Distribution_p8[i2,:], p=2)       
        dist4p = np.power(dist4p,self.power)
        correlation4p = np.exp(-dist4p/(2*self.lengthscale4p))
        
        
        Distribution_p9 = np.ascontiguousarray(X[:,(8*k+dimension_x):(9*k+dimension_x)])
        Distribution_theta9 = np.ascontiguousarray(X[:,(9*k+dimension_x):(10*k+dimension_x)])
        Distribution_p10 = np.ascontiguousarray(X2[:,(8*k+dimension_x):(9*k+dimension_x)])
        Distribution_theta10 = np.ascontiguousarray(X2[:,(9*k+dimension_x):(10*k+dimension_x)])
        dist5p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist5p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta9[i1,:], 
                                                  v_values = Distribution_theta10[i2,:], 
                                                  u_weights=Distribution_p9[i1,:], 
                                                  v_weights= Distribution_p10[i2,:], p=2)       
        dist5p = np.power(dist5p,self.power)
        correlation5p = np.exp(-dist5p/(2*self.lengthscale5p))
        
        
        
        Distribution_p11 = np.ascontiguousarray(X[:,(10*k+dimension_x):(11*k+dimension_x)])
        Distribution_theta11 = np.ascontiguousarray(X[:,(11*k+dimension_x):(12*k+dimension_x)])
        Distribution_p12 = np.ascontiguousarray(X2[:,(10*k+dimension_x):(11*k+dimension_x)])
        Distribution_theta12 = np.ascontiguousarray(X2[:,(11*k+dimension_x):(12*k+dimension_x)])
        dist6p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist6p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta11[i1,:], v_values = Distribution_theta12[i2,:], 
                                                  u_weights=Distribution_p11[i1,:], v_weights= Distribution_p12[i2,:], p=2)       
        dist6p = np.power(dist6p,self.power)
        correlation6p = np.exp(-dist6p/(2*self.lengthscale6p))

        return self.variance * correlation1x * correlation2x* correlation3x * correlation1p * correlation2p * correlation3p * correlation4p * correlation5p * correlation6p


    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0])
    

    def update_gradients_full(self, dL_dK, X, X2):
        dimension_x = self.dimension_x
        dimension_p = self.dimension_p
        
        if X2 is None: 
            X2 = X
                              
        diff1x = np.expand_dims(X[:,0:1],0) - np.expand_dims(X2[:,0:1],1)
        dist1x = np.sum(diff1x**2,axis = -1).T
        correlation1x = np.exp(-dist1x/(2*self.lengthscale1x))
        
        diff2x = np.expand_dims(X[:,1:2],0) - np.expand_dims(X2[:,1:2],1)
        dist2x = np.sum(diff2x**2,axis = -1).T
        correlation2x = np.exp(-dist2x/(2*self.lengthscale2x))
        
        diff3x = np.expand_dims(X[:,2:3],0) - np.expand_dims(X2[:,2:3],1)
        dist3x = np.sum(diff3x**2,axis = -1).T
        correlation3x = np.exp(-dist3x/(2*self.lengthscale3x))
        
        k = int((X.shape[1]-dimension_x)/(2*dimension_p))
        N1 = X.shape[0]
        N2 = X2.shape[0]
        
        Distribution_p1 = np.ascontiguousarray(X[:,dimension_x:(k+dimension_x)])
        Distribution_theta1 = np.ascontiguousarray(X[:,(k+dimension_x):(2*k+dimension_x)])
        Distribution_p2 = np.ascontiguousarray(X2[:,dimension_x:(k+dimension_x)])
        Distribution_theta2 = np.ascontiguousarray(X2[:,(k+dimension_x):(2*k+dimension_x)])
        dist1p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist1p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta1[i1,:], 
                                                  v_values = Distribution_theta2[i2,:],
                                                  u_weights=Distribution_p1[i1,:], 
                                                  v_weights= Distribution_p2[i2,:], p=2.0) 
        dist1p = np.power(dist1p,self.power)
        correlation1p = np.exp(-dist1p/(2*self.lengthscale1p))
        
        Distribution_p3 = np.ascontiguousarray(X[:,(2*k+dimension_x):(3*k+dimension_x)])
        Distribution_theta3 = np.ascontiguousarray(X[:,(3*k+dimension_x):(4*k+dimension_x)])
        Distribution_p4 = np.ascontiguousarray(X2[:,(2*k+dimension_x):(3*k+dimension_x)])
        Distribution_theta4 = np.ascontiguousarray(X2[:,(3*k+dimension_x):(4*k+dimension_x)])
        dist2p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist2p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta3[i1,:], 
                                                  v_values = Distribution_theta4[i2,:], 
                                                  u_weights=Distribution_p3[i1,:], 
                                                  v_weights= Distribution_p4[i2,:], p=2)       
        dist2p = np.power(dist2p,self.power)
        correlation2p = np.exp(-dist2p/(2*self.lengthscale2p)) 
        
        Distribution_p5 = np.ascontiguousarray(X[:,(4*k+dimension_x):(5*k+dimension_x)])
        Distribution_theta5 = np.ascontiguousarray(X[:,(5*k+dimension_x):(6*k+dimension_x)])
        Distribution_p6 = np.ascontiguousarray(X2[:,(4*k+dimension_x):(5*k+dimension_x)])
        Distribution_theta6 = np.ascontiguousarray(X2[:,(5*k+dimension_x):(6*k+dimension_x)])
        dist3p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist3p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta5[i1,:], 
                                                  v_values = Distribution_theta6[i2,:], 
                                                  u_weights=Distribution_p5[i1,:], 
                                                  v_weights= Distribution_p6[i2,:], p=2)       
        dist3p = np.power(dist3p,self.power)
        correlation3p = np.exp(-dist3p/(2*self.lengthscale3p))

        # P dimension
        Distribution_p7 = np.ascontiguousarray(X[:,(6*k+dimension_x):(7*k+dimension_x)])
        Distribution_theta7 = np.ascontiguousarray(X[:,(7*k+dimension_x):(8*k+dimension_x)])
        Distribution_p8 = np.ascontiguousarray(X2[:,(6*k+dimension_x):(7*k+dimension_x)])
        Distribution_theta8 = np.ascontiguousarray(X2[:,(7*k+dimension_x):(8*k+dimension_x)])
        dist4p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist4p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta7[i1,:], 
                                                  v_values = Distribution_theta8[i2,:], 
                                                  u_weights=Distribution_p7[i1,:], 
                                                  v_weights= Distribution_p8[i2,:], p=2)       
        dist4p = np.power(dist4p,self.power)
        correlation4p = np.exp(-dist4p/(2*self.lengthscale4p))
        
        Distribution_p9 = np.ascontiguousarray(X[:,(8*k+dimension_x):(9*k+dimension_x)])
        Distribution_theta9 = np.ascontiguousarray(X[:,(9*k+dimension_x):(10*k+dimension_x)])
        Distribution_p10 = np.ascontiguousarray(X2[:,(8*k+dimension_x):(9*k+dimension_x)])
        Distribution_theta10 = np.ascontiguousarray(X2[:,(9*k+dimension_x):(10*k+dimension_x)])
        dist5p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist5p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta9[i1,:], 
                                                  v_values = Distribution_theta10[i2,:], 
                                                  u_weights=Distribution_p9[i1,:], 
                                                  v_weights= Distribution_p10[i2,:], p=2)       
        dist5p = np.power(dist5p,self.power)
        correlation5p = np.exp(-dist5p/(2*self.lengthscale5p))

        # P dimension
        Distribution_p11 = np.ascontiguousarray(X[:,(10*k+dimension_x):(11*k+dimension_x)])
        Distribution_theta11 = np.ascontiguousarray(X[:,(11*k+dimension_x):(12*k+dimension_x)])
        Distribution_p12 = np.ascontiguousarray(X2[:,(10*k+dimension_x):(11*k+dimension_x)])
        Distribution_theta12 = np.ascontiguousarray(X2[:,(11*k+dimension_x):(12*k+dimension_x)])
        dist6p = np.zeros((N1,N2))
        for i1 in range(N1):
            for i2 in range(N2):
                dist6p[i1,i2] = ot.wasserstein_1d(u_values = Distribution_theta11[i1,:], 
                                                  v_values = Distribution_theta12[i2,:], 
                                                  u_weights=Distribution_p11[i1,:], 
                                                  v_weights= Distribution_p12[i2,:], p=2)       
        dist6p = np.power(dist6p,self.power)
        correlation6p = np.exp(-dist6p/(2*self.lengthscale6p))       
        
        correlation = correlation1x * correlation2x* correlation3x * correlation1p * correlation2p * correlation3p * correlation4p * correlation5p * correlation6p      
        dvar = correlation
        dl1x = self.variance * correlation * 0.5 * dist1x/(self.lengthscale1x**2)
        dl2x = self.variance * correlation * 0.5 * dist2x/(self.lengthscale2x**2)
        dl3x = self.variance * correlation * 0.5 * dist3x/(self.lengthscale3x**2)
        dl1p = self.variance * correlation * 0.5 * dist1p/(self.lengthscale1p**2)
        dl2p = self.variance * correlation * 0.5 * dist2p/(self.lengthscale2p**2)
        dl3p = self.variance * correlation * 0.5 * dist3p/(self.lengthscale3p**2)
        dl4p = self.variance * correlation * 0.5 * dist4p/(self.lengthscale4p**2)
        dl5p = self.variance * correlation * 0.5 * dist5p/(self.lengthscale5p**2)
        dl6p = self.variance * correlation * 0.5 * dist6p/(self.lengthscale6p**2)
        #dpower = self.variance * correlation1 * correlation2 * (-0.5)/self.lengthscale2 * dist2p * np.log(dist2)
        
        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale1x.gradient = np.sum(dl1x*dL_dK)
        self.lengthscale2x.gradient = np.sum(dl2x*dL_dK)
        self.lengthscale3x.gradient = np.sum(dl3x*dL_dK)
        self.lengthscale1p.gradient = np.sum(dl1p*dL_dK)
        self.lengthscale2p.gradient = np.sum(dl2p*dL_dK)
        self.lengthscale3p.gradient = np.sum(dl3p*dL_dK)
        self.lengthscale4p.gradient = np.sum(dl4p*dL_dK)
        self.lengthscale5p.gradient = np.sum(dl5p*dL_dK)
        self.lengthscale6p.gradient = np.sum(dl6p*dL_dK)
        
        #self.power.gradient = np.sum(dpower*dL_dK)
        #print("distance","\n",dist1x,"\n",dist2x,"\n", dist3x,"\n", dist1p,"\n",dist2p, "\n")
        #print("correlation","\n",correlation1x,"\n",correlation2x,"\n", correlation3x,"\n", correlation1p,"\n",correlation2p, "\n")

# %%
def GP_model(D,Y, method = None):
    
    if method == "NBRO":
        #kernel = GPy.kern.RBF(input_dim=D.shape[1], variance=1., lengthscale=1.)
        kernel = DistributionKernel(input_dim= D.shape[1], variance=1., lengthscale=1.,power=1)
        gp = GPy.models.GPRegression(D,Y,kernel)
        
        gp[''].constrain_positive()
        gp['distribution.lengthscale1x'].constrain_bounded(0,100)
        gp['distribution.lengthscale2x'].constrain_bounded(0,100)
        gp['distribution.lengthscale3x'].constrain_bounded(0,100)
        gp['distribution.lengthscale1p'].constrain_bounded(0,100)
        gp['distribution.lengthscale2p'].constrain_bounded(0,100)
        gp['distribution.lengthscale3p'].constrain_bounded(0,100)
        gp['distribution.lengthscale4p'].constrain_bounded(0,100)
        gp['distribution.lengthscale5p'].constrain_bounded(0,100)
        gp['distribution.lengthscale6p'].constrain_bounded(0,100)
        
        # power 2 fitting is not stable; power 1 if no constraint, some theta can be very high which is not reasonable

        #display(gp)
        gp.optimize(messages=False)
        gp.optimize_restarts(num_restarts = 1, messages = False, verbose = False)
        #display(gp)
        #fig = gp.plot()
    
    else:
        kernel = GPy.kern.RBF(input_dim=D.shape[1], variance=1., lengthscale=1.)
        gp = GPy.models.GPRegression(D,Y,kernel)
        #display(gp)
        gp.optimize(messages=False)
        gp.optimize_restarts(num_restarts = 10, messages = False, verbose = False)
        #display(gp)
        #fig = gp.plot()       
    
    return gp

# %%
def inital_design(n_sample, seed):  
    np.random.seed(seed)
    B1_list = np.arange(10,15)
    B2_list = np.arange(3,8)
    B3_list = np.arange(10,25)
    candidate = np.array([[i,j,k] for i in B1_list for j in B2_list for k in B3_list])
    X_all = candidate[((candidate[:,0] + candidate[:,1]+ candidate[:,2]/2) <= 28) & ((candidate[:,0] + candidate[:,1]+ candidate[:,2]/2) >= 26),:]
    if n_sample is None:
        D1 = X_all
    else:
        D1 = X_all[np.random.choice(X_all.shape[0], n_sample, replace=False)]
    
    return D1

def candidate_space(xi, n_candidate, n_k, dimension_p, P0_list, seed = None, alpha = 10):
    X_ = inital_design(n_candidate, seed)
    posterior_candidate = posterior_generation(xi, n_candidate,n_k, dimension_p, P0_list, alpha)
    P_ = distribution_to_vector(posterior_candidate)
    print(X_.shape, P_.shape)
    D_ = np.concatenate((X_,P_),axis = 1)

    return X_, D_

# %%
def fit_distribution_hist(xi):
    P = []
    for p in range(dimension_p):
        P.append(xi[p,:])
    return P

def fit_distribution_parametric(xi):
    P = []
    for p in range(dimension_p):
        if p == 0:
            P.append(fit_discrete(xi[p,:]))
        elif p == 1:
            P.append(fit_exponential(xi[p,:]))
        else:
            P.append(fit_log_normal(xi[p,:]))
    return P

# %%
def EGO_plug_in(seed): 

    # real world data generation
    xi = real_world_data(n_xi, seed, true_dist)

    # this is different
    # initial experiment design
    D1 = inital_design(n_sample, seed)

    # this is different
    if method == 'hist':
        P = fit_distribution_hist(xi)
    elif method == 'parametric':
        P = fit_distribution_parametric(xi)
    
    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        Y_list.append(f.evaluate(xx, P, n_rep, method))

    Y1 = np.array(Y_list).reshape(-1,1)
    print(D1.shape,Y1.shape)

    # Fit to data using Maximum Likelihood Estimation of the parameters 
    gp0 = GP_model(D1, Y1, method) 
    # test = gp0.predict(D1,full_cov = True)[1]
    # print(np.mean(test),np.max(test),np.min(test))

    # this is different
    ## get prediction and obtain minimizer
    X_ = inital_design(n_candidate, None)
    mu_g,sigma_g = gp0.predict(X_,full_cov=False,include_likelihood = False)

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
        D_new = EGO(X_, X_update,mu_g,sigma_g,gp0)

        # 2. Add selected samples
        D_update, Y_update = update_data(D_new,D_update,Y_update,P, n_rep, method, f)
        X_update =  np.vstack({tuple(row) for row in D_update})

        # 3. Update GP model and make prediction 
        gp0 = GP_model(D_update, Y_update, method)
        mu_g,sigma_g = gp0.predict(X_,full_cov=False,include_likelihood = False) 

        # 4. Update x^* 
        minimizer.append(X_[np.argmin(mu_g)])
        minimum_value.append(min(mu_g))

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
    
    output(file_pre, Y_update, D_update, minimizer,minimum_value, TIME_RE, f, x_star, f_star, seed, n_sample, n_iteration, method, n_xi)

# %%
def NBRO(seed):
    
    # real world data generation
    xi = real_world_data(n_xi, seed, true_dist)
    print(xi.shape) #dimension_p, n_xi

    # this is different
    # obtain posterior and get n_MC samples
    P0_list = [
        stats.rv_discrete(values=(np.arange(1,5), (0.25, 0.25, 0.25, 0.25))),
        scipy.stats.uniform(0,1.4),
        scipy.stats.uniform(4.8,38.2),
        scipy.stats.uniform(0.3,17.1),
        scipy.stats.uniform(1.3,9.0),
        scipy.stats.uniform(11.1,25.2)
    ]
    posterior = posterior_generation(xi,n_MC,n_k, dimension_p, P0_list)

    # the ture minimizer and minimizer value 
    x_star = f.x_star
    f_star = f.f_star

    # this is different
    # initial experiment design
    X1, D1 = candidate_space(xi,  n_sample, n_k, dimension_p, P0_list,seed = seed, alpha = alpha)
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
        Y_list.append(f.evaluate(xx, P , n_rep, method))
    Y1 = np.array(Y_list).reshape(-1,1)
    print(D1.shape,Y1.shape)

    # Fit to data using Maximum Likelihood Estimation of the parameters 
    gp0 = GP_model(D1, Y1, method) 
    # test = gp0.predict(D1,full_cov = True)[1]
    # print(np.mean(test),np.max(test),np.min(test))

    # this is different
    ## get prediction and obtain minimizer
    X_, D_ = candidate_space(xi, n_candidate,n_k, dimension_p, P0_list,seed = None, alpha = alpha)
    #mu_g,sigma_g= Mu_Std_G(X_,gp0,posterior)
    mu_g = Mu_G(X_,gp0,posterior)
    minimizer = list([X_[np.argmin(mu_g)]])
    minimum_value = list([min(mu_g)])
    print(mu_g)

    # Prepare for iteration 
    TIME_RE = list()
    D_update = D1
    X_update = D_update[:,0:dimension_x]
    Y_update = Y1  

    for i in range(n_iteration):

        start = time.time()

        # 0. regenerate candidate space for each iteration
        X_, D_ = candidate_space(xi, n_candidate,n_k, dimension_p, P0_list,seed = None, alpha = alpha)

        # 1. Algorithm for x and \lambda
        D_new = EGO_PB(gp0,posterior,X_update,D_,dimension_x)
        print(D_new.shape)

        # 2. Add selected samples
        D_update, Y_update = update_data_PB(D_new,D_update,Y_update, n_rep, method,n_k,f, dimension_x, dimension_p)

        # 3. Update GP model
        gp0 = GP_model(D_update, Y_update, method)
        #mu_g, _ = Mu_Std_G(X_,gp0,posterior)
        mu_g = Mu_G(X_,gp0,posterior)
        print(mu_g)

        # 4. Update x^* 
        minimizer.append(X_[np.argmin(mu_g)])
        minimum_value.append(min(mu_g))

        # 5. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

    output(file_pre, Y_update, D_update, minimizer,minimum_value, TIME_RE, f, x_star, f_star, seed, n_sample, n_iteration, method, n_xi, dimension_p, posterior, n_k, get_g_hat_x)

# %%
def Experiment(seed):
    
    if method in ["parametric", 'hist']:
        EGO_plug_in(seed)
    elif method == 'NBRO':
        NBRO(seed) 

# %%
# %tb
import argparse
parser = argparse.ArgumentParser(description='NBRO-algo')
#parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

parser.add_argument('-t','--time', type=str, help = 'Date of the experiments, e.g. 20201109',default = '20240824')
parser.add_argument('-method','--method', type=str, help = 'NBRO/hist/exp/lognorm',default = "NBRO")
parser.add_argument('-problem','--problem', type=str, help = 'Function',default = 'ccf')

parser.add_argument('-smacro','--start_num',type=int, help = 'Start number of macroreplication', default = 12)
parser.add_argument('-macro','--repet_num', type=int, help = 'Number of macroreplication',default = 24)
parser.add_argument('-core','--n_cores', type=int, help = 'Number of Cores',default = 12)

parser.add_argument('-n_xi','--n_xi', type=int, help = 'Number of observations of the random variable',default = 10)
parser.add_argument('-n_sample','--n_sample',type=int, help = 'number of initial samples', default = 30)
parser.add_argument('-n_i','--n_iteration', type=int, help = 'Number of iteration',default = 30)

parser.add_argument('-n_rep','--n_replication', type=int, help = 'Number of replications at each point',default = 2)
parser.add_argument('-n_candidate','--n_candidate', type=int, help = 'Number of candidate points for each iteration',default = 100)
parser.add_argument('-n_MC','--n_MC', type=int, help = 'Number of Monte Carlo samples for the posterior',default = 50)
parser.add_argument('-n_k','--n_k', type=int, help = 'Number of DP truncation',default = 1000) #50 #100
parser.add_argument('-get_g_hat_x','--get_g_hat_x',help = 'get g_hat_x',action='store_true')
 
parser.add_argument('-alpha','--alpha', type=int, help = 'alpha',default = 10)

#cmd = []
#args = parser.parse_args(cmd)
args = parser.parse_args()
print(args)

time_run = args.time 
method = args.method 
problem = args.problem 

n_xi = args.n_xi
alpha = args.alpha
n_sample = args.n_sample # number of initial samples
n_iteration = args.n_iteration # Iteration

n_rep = args.n_replication  #number of replication on each point
n_candidate = args.n_candidate # Each iteration, number of the candidate points are regenerated+
n_MC = args.n_MC  # number of Monte Carlo samples
n_k = n_xi * 8 #args.n_k #Dirichlet Process Truncation
get_g_hat_x = args.get_g_hat_x

# true distribution of the demand distribution
true_dist = [
    {'dist': "discrete", 'p_i':[0.2, 0.55,0.2,0.05]},
    {'dist': "exp", 'rate': 3.3},
    # {'dist': "lognorm", 'mu': [15.0], 'sigma':[7] },
    # {'dist': "lognorm", 'mu': [3.4], 'sigma':[3.5]},
    # {'dist': "lognorm", 'mu': [3.8], 'sigma':[1.6]},
    # {'dist': "lognorm", 'mu': [17], 'sigma': [3]},
    {'dist': "m-lognorm", 'mu': [15.0,30], 'sigma':[7,7], 'p_i': [0.5,0.5]},
    {'dist': "m-lognorm", 'mu': [3.4,6.8], 'sigma':[3.5,3.5], 'p_i': [0.5,0.5]},
    {'dist': "m-lognorm", 'mu': [3.8,7.6], 'sigma':[1.6,1.6], 'p_i': [0.5,0.5]},
    {'dist': "m-lognorm", 'mu': [17, 34], 'sigma': [3,3], 'p_i': [0.5,0.5]},
]


# %%
## function
if problem == 'ccf':
    f = critical_care_facility_problem(true_dist)

dimension_x = f.dimension_x
dimension_p = f.dimension_p
lower_bound = f.lb
upper_bound = f.ub
f_star = f.f_star
x_star = f.x_star
print(f_star, x_star)

sample_space = f.set_sample_space()
sample_space.shape

# f.estimate_minimizer_minimum()
# # f_star = 352.1
# # x_star = [2.24445979, 2.67431892]

file_pre = '_'.join(["outputs/bo", time_run, "n_rep", str(n_rep), problem, method, true_dist[2].get('dist')])
print(file_pre)


if __name__ == '__main__':

    start_num = args.start_num  
    repet_num = args.repet_num # Macro-replication

    import multiprocessing
    starttime = time.time()
    pool = multiprocessing.Pool()
    pool.map(Experiment, range(start_num,repet_num))
    #pool.map(Experiment, [4,7,8,9])
    pool.close()
    print('That took {} secondas'.format(time.time() - starttime))

# %%
# time_run = "20230805"
# method = "parametric"
# problem  = "ccf"

# # Setting
# start_num = 0 
# repet_num = 1 # Macro-replication

# n_xi = 10
# n_sample = 10
# n_iteration = 2 # Iteration

# # Initial Design
# n_rep = 1 #number of replication on each point
# n_candidate = 100 
# n_MC = 50
# n_k = 30
# get_g_hat_x = False

# %%
# for method in ["hist", 'paramteric', 'NBRO']:
#     for seed in range(start_num,repet_num):
#         print(seed, method)
#         Experiment(seed)

# %%
# for method in ['hist']:
#     for seed in range(start_num,repet_num):
#         print(seed)
#         EGO_plug_in(seed)

# %%
# for method in ["NBRO"]:
#     for seed in range(start_num,repet_num):
#         print(seed)
#         NBRO(seed)

# %%


# %%
# if __name__ == "__main__":
#     pool = multiprocessing.Pool()
#     pool.map(EGO_Nonparametric, range(start_num,repet_num))
#     pool.close()

# %%
# p = 5
# xi = set_generate_mixture_of_log_normal(
#     1000,true_dist[p].get('mu'),true_dist[p].get('sigma'),true_dist[p].get('p_i'), plots=True)

# %%



