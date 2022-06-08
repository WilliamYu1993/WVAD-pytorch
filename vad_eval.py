from __future__ import division
import numpy as np, pdb

def vad_evaluate(y_pred, y_true, threshold=0.5): 
    #pdb.set_trace()
    length = min(len(y_pred), len(y_true))
    crt = 0
    inc = 0
    fl = 0    
    #pdb.set_trace() 
    for i in range(length):        
        if y_pred[i,1]>=y_pred[i,0]:# or y_pred[i,1]>=np.float(threshold):# or y_pred[i,1]>=np.float(threshold):
        #if y_pred[i]>=np.float(threshold):# or y_pred[i][1]>=y_pred[i][0]:
            y_pred[i,1]=1
            y_pred[i,0]=0
        else:
            y_pred[i,1]=0
            y_pred[i,0]=1
        
    y_pred = y_pred.astype("int")
    #pdb.set_trace()
    
    for i in range(length):
        if y_pred[i,1]-y_true[i,1]==0:
            crt = crt + 1
        elif y_pred[i,1]-y_true[i,1]==1:
            inc = inc + 1    
        elif y_pred[i,1]-y_true[i,1]==-1:
            fl = fl + 1
        
    return crt/length, inc/length, fl/length
















