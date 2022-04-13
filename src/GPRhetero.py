import gpflow
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple

class GPRhetero(gpflow.models.GPR):
  r"""
  Gaussian Process Regression with heteroscedastic noise.
    
  This uses https://github.com/GPflow/GPflow/blob/v2.2.1/gpflow/models/gpr.py
  as the parent class and then additionally takes in a vector
  containing the noise variance associated with the (1D) output data

  The only difference is that when the noise is added to the covariance
  function instead of a single value (noise_variance) multiplied
  by the identity matrix, a matrix with the known vector (NoiseVar) running
  along the diagonal is used.   

  """
  def __init__(
    self,                                                
    data: gpflow.models.model.RegressionData,
    NoiseVar: gpflow.models.model.RegressionData,
    kernel: gpflow.kernels.Kernel,
    mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
    noise_variance: float = 1.0,
  ):
    super().__init__(data, kernel, mean_function, noise_variance)

    self.NoiseVar = gpflow.models.util.data_input_to_tensor(NoiseVar)
    # noise_variance is completely ignored as it is no longer a hyperparameter
    gpflow.utilities.set_trainable(self.likelihood.variance, False) 

    if (self.num_latent_gps != 1 or NoiseVar.shape[-1] != 1):
      raise Exception("Multi-output GPR with heteroscedastic noise not implemented")

  def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
    """
    Returns K + N, where N is a matrix with the vector 
    NoiseVar running along the diagonal
    """

    k_diag = tf.linalg.diag_part(K)
    s_diag = tf.reshape(self.NoiseVar, tf.shape(k_diag))
    return tf.linalg.set_diag(K, k_diag + s_diag)


def CheckGPRhetero():
    """Check to make sure that the GPRhetero code is working correctly. In which case, return True."""
    
    if (gpflow.__version__ != '2.2.1'):
        return False
    
    # Generate training data
    X=np.linspace(0,3,2).reshape(-1,1)
    Y=X*X-1
    NoiseVar=X*.05+.01
    
    # Test data and known results
    Xstar=np.linspace(1,2,2).reshape(-1,1)
    muref=np.array([[1.24694676],[5.14551933]])
    varref=np.array([[0.39587969],[0.45496805]])

    # Build model
    kernel = gpflow.kernels.SquaredExponential(lengthscales=1.3)
    model = GPRhetero((X,Y), NoiseVar, kernel=kernel, mean_function=gpflow.mean_functions.Constant(Y.mean()))

    mu, var = model.predict_f(Xstar)

    return np.all(np.concatenate((np.isclose(mu.numpy(),muref),np.isclose(var.numpy(),varref))))
