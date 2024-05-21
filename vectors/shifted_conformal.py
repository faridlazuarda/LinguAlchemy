import torch
import torch.nn as nn

class ShiftedConformal(nn.Module):
    # Definition:
    #   X ∈ R^{N X dA}: Source Embedding Vectors
    #   Y ∈ R^{M X db}: Target Embedding Vectors 
    #   x ∈ R^{B X dA}: Source Embedding Batch Vectors
    #   y ∈ R^{B X db}: Target Embedding Batch Vectors
    #   Tx ∈ R^{dA}: Source Embedding Shift
    #   Ty ∈ R^{db}: Target Embedding Shift
    #   W ∈ R^{dA X dB}: Orthogonal Transformation
    #   S ∈ R: Scaling Factor
    
    # Objective:
    #   argmin || S.W.(x - Tx) - (y - Ty) ||
        
    def __init__(self, src_dim, tgt_dim, loss_fn=nn.MSELoss()):
        super(ShiftedConformal, self).__init__()      
        
        # Init Orthogonal transformation = eye matrix
        self.W = torch.nn.Linear(src_dim, tgt_dim, bias=False) 
        if src_dim > tgt_dim:
            self.W.weight.data[:tgt_dim,:tgt_dim] = torch.eye(tgt_dim)
            self.W.weight.data[:,tgt_dim:] = 0
        elif src_dim < tgt_dim:
            self.W.weight.data[:src_dim,:src_dim] = torch.eye(src_dim)
            self.W.weight.data[src_dim:,:] = 0
        else: # src_dim == tgt_dim
            self.W.weight.data = torch.eye(src_dim)
            
        # Init Source shift = 0, Target shift = 0, Scaling factor = 1
        self.Tx = nn.Parameter(torch.zeros(1, src_dim)) 
        self.Ty = nn.Parameter(torch.zeros(1, tgt_dim)) 
        self.S = nn.Parameter(torch.ones(1,1))
        
        # Init Loss Function
        self.loss_fn = loss_fn

    def forward(self, x, y=None):
        # Compute Batch Loss: S.W.(x - Tx) - (y - Ty)
        hx = x - self.Tx # Source Shift
        hx = self.S * self.W(hx) # Conformal
        hx = hx + self.Ty # Target Shift
        
        if y is not None:
            loss = self.loss_fn(hx, y) # Compute Loss
            return loss, hx
        return hx