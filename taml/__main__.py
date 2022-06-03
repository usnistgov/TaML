import numpy as np
import pandas as pd
import os
from taml import mltools as mltools
from taml import mlmethods as mlmeth
from taml import mlmethodsGPRhetero as mlmethGPRhetero
from taml import mlmethodsGPRhomo as mlmethGPRhomo
from taml import GPRhetero

from sklearn.ensemble import RandomForestRegressor


########## define the theory ###########

def calc_rg(N):
    """
    Define the theory.
    """
    return N / 6

### check GPR with heteroscedastic noise ###


if (GPRhetero.CheckGPRhetero() != True):
    raise Exception(
        "GPR with heteroscedastic noise class is not working as expected. Aborting.")

########## get the data ################

path = os.path.dirname(__file__)

# will be updated with MIDAS path
df = pd.read_csv('https://raw.githubusercontent.com/usnistgov/TaML/main/data/rgmaindata.csv')  # regular data
dfho = pd.read_csv('https://raw.githubusercontent.com/usnistgov/TaML/main/data/rgoutlierdata.csv')  # outlier data

########## set up data for ML ##########

# main data
y_all = np.array(df['mean of log Rg^2'])
x_all = np.array(df[['log N', 'alpha']])
t_all = np.log(calc_rg(np.exp(x_all[:, 0]))).reshape(len(x_all[:, 0]))
a_all = np.array(df['var err of log Rg^2'])
dat = [x_all, y_all, t_all, a_all]

# outlier
y_out = np.array(dfho['mean of log Rg^2'])
x_out = np.array(dfho[['log N', 'alpha']])
t_out = np.log(calc_rg(np.exp(x_out[:, 0]))).reshape(len(x_out[:, 0]))
dat_out = [x_out, y_out, t_out]

seed = 1234

########## run GPR with heteroscedastic noise #############

# regular
mltools.run_datasets(dat, mlmethGPRhetero.direct, seed, 'hetero_direct.csv')
mltools.run_datasets(dat, mlmethGPRhetero.difference,
                     seed, 'hetero_difference.csv')
mltools.run_datasets(dat, mlmethGPRhetero.quotient,
                     seed, 'hetero_quotient.csv')
mltools.run_datasets(dat, mlmethGPRhetero.latentvariable,
                     seed, 'hetero_latentvariable.csv')
mltools.run_datasets(dat, mlmethGPRhetero.linearprior,
                     seed, 'hetero_linearprior.csv')
mltools.run_datasets(dat, mlmethGPRhetero.fixedprior,
                     seed, 'hetero_fixedprior.csv')
mltools.run_datasets(dat, mlmethGPRhetero.parameterization,
                     seed, 'hetero_parameterization.csv')

# outlier
mltools.run_datasets(dat, mlmethGPRhetero.direct, seed,
                     'hetero_out_direct.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhetero.difference, seed,
                     'hetero_out_difference.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhetero.quotient, seed,
                     'hetero_out_quotient.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhetero.latentvariable, seed,
                     'hetero_out_latentvariable.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhetero.linearprior, seed,
                     'hetero_out_linearprior.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhetero.fixedprior, seed,
                     'hetero_out_fixedprior.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhetero.parameterization, seed,
                     'hetero_out_parameterization.csv', dat_out)

#### run GPR with homoscedastic noise ###

# regular
mltools.run_datasets(dat, mlmethGPRhomo.direct, seed, 'homo_direct.csv')
mltools.run_datasets(dat, mlmethGPRhomo.difference,
                     seed, 'homo_difference.csv')
mltools.run_datasets(dat, mlmethGPRhomo.quotient, seed, 'homo_quotient.csv')
mltools.run_datasets(dat, mlmethGPRhomo.latentvariable,
                     seed, 'homo_latentvariable.csv')
mltools.run_datasets(dat, mlmethGPRhomo.linearprior,
                     seed, 'homo_linearprior.csv')
mltools.run_datasets(dat, mlmethGPRhomo.fixedprior,
                     seed, 'homo_fixedprior.csv')
mltools.run_datasets(dat, mlmethGPRhomo.parameterization,
                     seed, 'homo_parameterization.csv')

# outlier
mltools.run_datasets(dat, mlmethGPRhomo.direct, seed,
                     'homo_out_direct.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhomo.difference, seed,
                     'homo_out_difference.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhomo.quotient, seed,
                     'homo_out_quotient.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhomo.latentvariable, seed,
                     'homo_out_latentvariable.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhomo.linearprior, seed,
                     'homo_out_linearprior.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhomo.fixedprior, seed,
                     'homo_out_fixedprior.csv', dat_out)
mltools.run_datasets(dat, mlmethGPRhomo.parameterization, seed,
                     'homo_out_parameterization.csv', dat_out)

########## run theory ########

mltools.run_datasets(dat, mlmeth.theory, seed, 'theory.csv')
mltools.run_datasets(dat, mlmeth.theory, seed, 'out_theory.csv', dat_out)

######### run Random Forest #########

# hyperparameters determined from optimization
n_est = 1000
max_dep = None
 
RFmodel = RandomForestRegressor(n_estimators=n_est, max_depth=max_dep)
# One can add random_state=int for reproducible results to the above line.
 
# regular
filesuffix = str(n_est) + '_' + str(max_dep) + '.csv'

mltools.run_datasets(dat, mlmeth.direct, seed,
                     'rf_direct_' + filesuffix, model=RFmodel)
mltools.run_datasets(dat, mlmeth.difference, seed,
                     'rf_difference_' + filesuffix, model=RFmodel)
mltools.run_datasets(dat, mlmeth.quotient, seed,
                     'rf_quotient_' + filesuffix, model=RFmodel)
mltools.run_datasets(dat, mlmeth.latentvariable, seed,
                     'rf_latentvariable_' + filesuffix, model=RFmodel)
mltools.run_datasets(dat, mlmeth.multitask, seed,
                     'rf_multitask_' + filesuffix, model=RFmodel)

# outlier
mltools.run_datasets(dat, mlmeth.direct, seed,
                     'rf_out_direct_' + filesuffix, dat_out, model=RFmodel)
mltools.run_datasets(dat, mlmeth.difference, seed,
                     'rf_out_difference_' + filesuffix, dat_out, model=RFmodel)
mltools.run_datasets(dat, mlmeth.quotient, seed,
                     'rf_out_quotient_' + filesuffix, dat_out, model=RFmodel)
mltools.run_datasets(dat, mlmeth.latentvariable, seed,
                     'rf_out_latentvariable_' + filesuffix, dat_out, model=RFmodel)
mltools.run_datasets(dat, mlmeth.multitask, seed,
                     'rf_out_multitask_' + filesuffix, dat_out, model=RFmodel)
