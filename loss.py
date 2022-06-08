"""STFT-based Loss modules."""

import torch, pdb
import torch.nn.functional as F

def crossentropy(y, x, epsilon):
    output = x.clone() 
    output /= torch.sum(output, len(output.shape)-1, keepdim=True, dtype=torch.float32)
    output = torch.clamp(output, epsilon, 1. - epsilon, out=None)
    return - torch.sum(y*torch.log(output), len(output.shape)-1, keepdim=False, dtype=torch.float32)

class categorical_crossentropy(torch.nn.Module):
    "Implemented as categorical_crossentropy loss in Keras==1.2.2"
    
    def __init__(self):

        super(categorical_crossentropy, self).__init__()
        self.epsilon = 1e-7
        
    def forward(self, y, x):
        return torch.mean(crossentropy(y, x, self.epsilon), dtype=torch.float32)



