import numpy as np

def L2_norm(y_true,y_pred):


    loss=np.mean((y_true-y_pred)**2)
    return loss

