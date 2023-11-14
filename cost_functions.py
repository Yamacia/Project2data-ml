import autograd.numpy as np

def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):

    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func

def CostR2score(target):
    
    def func(X):
        ss_res = np.sum((target - X) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        #note we dont do 1 - (ss_res / ss_tot), such that the best fit is when r2_score returns 0
        r2_score = ss_res / ss_tot
        return r2_score
    return func



