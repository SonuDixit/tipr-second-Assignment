from scipy.stats import truncnorm
import numpy as np
class Initialise:
    def __init__(self):
        pass
    @staticmethod
    def normal(shape,mean=0,std = 0.5, seed = 123):
        np.random.seed(seed)
        # return np.random.normal(mean,std,shape)
        return  std * np.random.randn(shape[0],shape[1])
    @staticmethod
    def uniform(shape,low=-0.25,high=0.25,seed=123):
        np.random.seed(seed)
        return np.random.uniform(low, high, shape)
    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def glorot_normal(shape,mean=0,seed=123):
        std = np.sqrt(2/(shape[0]+shape[1]))
        scale=1.0
        scale /= max(1,float(shape[0]+shape[1])/2)
        std = np.sqrt(scale)/.87962566103423978
        a = -2 * std
        b = 2 * std
        return truncnorm.rvs(a,b,mean, std, size=shape,random_state = seed)
