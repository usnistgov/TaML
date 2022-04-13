import numpy as np
import pandas as pd
import mltools as mltools
import mlmethods as mlmeth
import mlmethodsGPRhetero as mlmethGPRhetero
import mlmethodsGPRhomo as mlmethGPRhomo
import GPRhetero

from sklearn.ensemble import RandomForestRegressor


########## define the theory ###########

def calc_rg(N):
    """
    Define the theory.
    """
    return N/6

### check GPR with heteroscedastic noise ###

if (GPRhetero.CheckGPRhetero() != True):
    raise Exception("GPR with heteroscedastic noise class is not working as expected. Aborting.")

########## get the data ################

df=pd.read_csv("../data/rgmaindata.csv") # regular data
dfho=pd.read_csv("../data/rgoutlierdata.csv") # outlier data

########## set up data for ML ##########

# main data
y_all = np.array(df['log Rg,mean'])
x_all = np.array(df[['log N','lambda']])
t_all = np.log(calc_rg(np.exp(x_all[:,0]))).reshape(len(x_all[:,0]))
a_all = np.array(df['log Rg, var err'])
dat=[x_all,y_all,t_all,a_all]

# outlier
y_out = np.array(dfho['log Rg,mean'])
x_out = np.array(dfho[['log N','lambda']])
t_out = np.log(calc_rg(np.exp(x_out[:,0]))).reshape(len(x_out[:,0]))
dat_out=[x_out,y_out,t_out]

seed=1234

########## run GPR with heteroscedastic noise #############

# regular
mltools.run_datasets(dat,mlmethGPRhetero.direct,seed,'hetero_direct.dat') 
mltools.run_datasets(dat,mlmethGPRhetero.difference,seed,'hetero_difference.dat') 
mltools.run_datasets(dat,mlmethGPRhetero.quotient,seed,'hetero_quotient.dat') 
mltools.run_datasets(dat,mlmethGPRhetero.latentvariable,seed,'hetero_latentvariable.dat') 
mltools.run_datasets(dat,mlmethGPRhetero.linearprior,seed,'hetero_linearprior.dat') 
mltools.run_datasets(dat,mlmethGPRhetero.fixedprior,seed,'hetero_fixedprior.dat') 
mltools.run_datasets(dat,mlmethGPRhetero.parameterization,seed,'hetero_parameterization.dat') 

# outlier
mltools.run_datasets(dat,mlmethGPRhetero.direct,seed,'hetero_out_direct.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhetero.difference,seed,'hetero_out_difference.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhetero.quotient,seed,'hetero_out_quotient.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhetero.latentvariable,seed,'hetero_out_latentvariable.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhetero.linearprior,seed,'hetero_out_linearprior.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhetero.fixedprior,seed,'hetero_out_fixedprior.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhetero.parameterization,seed,'hetero_out_parameterization.dat',dat_out)

#### run GPR with homoscedastic noise ###

# regular
mltools.run_datasets(dat,mlmethGPRhomo.direct,seed,'homo_direct.dat') 
mltools.run_datasets(dat,mlmethGPRhomo.difference,seed,'homo_difference.dat') 
mltools.run_datasets(dat,mlmethGPRhomo.quotient,seed,'homo_quotient.dat') 
mltools.run_datasets(dat,mlmethGPRhomo.latentvariable,seed,'homo_latentvariable.dat') 
mltools.run_datasets(dat,mlmethGPRhomo.linearprior,seed,'homo_linearprior.dat') 
mltools.run_datasets(dat,mlmethGPRhomo.fixedprior,seed,'homo_fixedprior.dat') 
mltools.run_datasets(dat,mlmethGPRhomo.parameterization,seed,'homo_parameterization.dat') 

# outlier
mltools.run_datasets(dat,mlmethGPRhomo.direct,seed,'homo_out_direct.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhomo.difference,seed,'homo_out_difference.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhomo.quotient,seed,'homo_out_quotient.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhomo.latentvariable,seed,'homo_out_latentvariable.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhomo.linearprior,seed,'homo_out_linearprior.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhomo.fixedprior,seed,'homo_out_fixedprior.dat',dat_out)
mltools.run_datasets(dat,mlmethGPRhomo.parameterization,seed,'homo_out_parameterization.dat',dat_out)

########## run theory ########

mltools.run_datasets(dat,mlmeth.theory,seed,'theory.dat') 
mltools.run_datasets(dat,mlmeth.theory,seed,'out_theory.dat',dat_out)

######### run Random Forest #########

# hyperparameters determined from optimization
n_est=1000
max_dep=None
 
mltools.run_datasets(dat,mlmeth.direct,seed,'rf_direct_'+str(n_est)+'_'+str(max_dep)+'.dat',model=RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.difference,seed,'rf_difference_'+str(n_est)+'_'+str(max_dep)+'.dat',model=RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.quotient,seed,'rf_quotient_'+str(n_est)+'_'+str(max_dep)+'.dat',model=RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.latentvariable,seed,'rf_latentvariable_'+str(n_est)+'_'+str(max_dep)+'.dat',model=RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.multitask,seed,'rf_multitask_'+str(n_est)+'_'+str(max_dep)+'.dat',model=RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
 
mltools.run_datasets(dat,mlmeth.direct,seed,'rf_direct_'+str(n_est)+'_'+str(max_dep)+'_out.dat',dat_out,RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.difference,seed,'rf_difference_'+str(n_est)+'_'+str(max_dep)+'_out.dat',dat_out,RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.quotient,seed,'rf_quotient_'+str(n_est)+'_'+str(max_dep)+'_out.dat',dat_out,RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.latentvariable,seed,'rf_latentvariable_'+str(n_est)+'_'+str(max_dep)+'_out.dat',dat_out,RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
mltools.run_datasets(dat,mlmeth.multitask,seed,'rf_multitask_'+str(n_est)+'_'+str(max_dep)+'_out.dat',dat_out,RandomForestRegressor(n_estimators=n_est,max_depth=max_dep))
