r"""
These are the helper tools for running machine learning. 
"""

import numpy as np
import random
import scipy

import gpflow
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary

def test_train(train_size,tot_size,seed):
    """
    Generates a test train split such that all of the data not used for
    testing is always used for training. Each data point is always used
    for at least one test split to ensure accurate statistics.
    """
    
    random.seed(a=seed)
    
    # Generate array of total size
    indices=np.arange(tot_size)
    random.shuffle(indices)
    doubleindices=np.concatenate((indices,indices)) # do this for lazy pbc
    
    # This determines the shift. A maximum train_size is 0.9 of tot_size.
    # If all data is used, his corresponds to 10-fold validation
    shift_size=int(train_size/9)
    if(shift_size==0):
        shift_size=1
    shift_value=0
    
    test=[]
    train=[]
    
    while (shift_value < tot_size):
        
        train_indx=doubleindices[shift_value:shift_value+train_size]
        test_indx= [i for i in indices if i not in train_indx]
        
        train.append(train_indx)
        test.append(test_indx)
        
        shift_value+=shift_size
    
    return(train,test)


def validation(size,data,metamodel,seed,data_out,model=None):
    """
    Validation splits the data into test and train and then runs
    the model, and computes statistics (average of the mean squared
    error and accompanying standard error) for a desired training
    set size.
    """
 
    x=data[0]
    y=data[1]
    t=data[2]
    a=data[3]

    mse=[]
    errar=[]
    
    # split test and train
    train,test=test_train(size,len(y),seed)
    
    for i in range(len(test)):
    
        train_index=train[i]
        test_index=test[i]
        
        print("TRAIN:",train_index)
        print("TEST:",test_index)

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]
        a_train, a_test = a[train_index], a[test_index]
        
        if(data_out!=False):
            x_test=data_out[0]
            y_test=data_out[1]
            t_test=data_out[2]
    
        if(model==None):
            result=metamodel(x_train,y_train,t_train,a_train,x_test,y_test,t_test)
        else:
            result=metamodel(x_train,y_train,t_train,a_train,x_test,y_test,t_test,model)
        print("mse is",result)
        mse.append(result)
        
    npmse=np.array(mse)
    print('************************************')
    print(npmse)
    print('************************************')
    
    return npmse.mean(), scipy.stats.sem(npmse, axis=0, ddof=1, nan_policy='propagate')


def run_datasets(data,metamodel,seed,filename,data_out=False,model=None):
    """
    Runs the validation across all dataset sizes. Data set sizes are chose on a log scale.
    """

    # choose dataset sizes
    num=10
    datasize=np.round(np.logspace(np.log10(5), np.log10(0.9*len(data[1])), num=num))
    print(datasize)
    mean=np.zeros(num)
    ste=np.zeros(num)
    
    # loop over dataset size
    for j, size in enumerate(datasize):
        
        mean[j],ste[j]=validation(int(size),data,metamodel,seed,data_out,model)

    print("datasize",datasize)
    print("mean", mean)
    print("ste",ste)
    print(np.hstack([datasize,mean,ste]))

    np.savetxt(filename,np.vstack([datasize, mean, ste]).transpose())

    return


def optimizeGPR(model,niter=5000):
    """
    Perform model (hyperparameter) optimization for GPR using Adam as implemented
    in TensorFlow. A stochastic optimizer helps to reduce getting stuck in a local
    minimum.
    """ 

    adam = tf.optimizers.Adam(learning_rate=0.01)
    
    minimum = model.training_loss().numpy()
    minmodelparam = gpflow.utilities.read_values(model)
    
    print_summary(model)
    
    @tf.function
    def step(i):
        adam.minimize(model.training_loss, model.trainable_variables)
    
    for i in tf.range(ci_niter(niter)):
        step(i)
    
    for i in tf.range(ci_niter(100)):
        step(i)
    
        # keep track of the actual minimum to eliminate error from unstable optimization
        if model.training_loss().numpy() < minimum:
            minimum = model.training_loss().numpy()
            minmodelparam = gpflow.utilities.read_values(model)
        
    gpflow.utilities.multiple_assign(model,minmodelparam)
    print_summary(model)
    print("Maximum Likelihood is",-1*model.training_loss().numpy())
    
    return model
