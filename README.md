# Bayesian-Nonparametric-Approach-for-BBO
This repository contains the implementation of the DaBNO-K algorithm described in the paper: "A Data-Driven Bayesian Nonparametric Approach for Black-Box Optimization"

### Code structure
`nbro.py` contains codes for the algorithm
`experiment_NBRO_numerical.py` contains codes for numerical problems, including Griewank and StybTang
`experiment_NBRO_inventory.py` contains codes for inventory problem
`experiment_NBRO_ccf.py` contains codes for CCF problem
`experiment_NBRO_online.py` contains codes for the online inventory problem

### Usage Example
Take the Griewank problem as an example, run for the experiment with the number of real world data being 10
```bash
python experiment_NBRO_numerical.py -method NBRO -n_xi 10 -problem Griewank
python experiment_NBRO_numerical.py -method hist -n_xi 10 -problem Griewank
python experiment_NBRO_numerical.py -method exp -n_xi 10 -problem Griewank
python experiment_NBRO_numerical.py -method lognorm -n_xi 10 -problem Griewank
```


