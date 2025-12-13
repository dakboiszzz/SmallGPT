import torch
import torch.nn as nn
class RotEmb(nn.Module):
    def __init__(self, head_size, T, theta = 10000):
        super().__init__()
        # prepare frequency right at the init
        
        # freqs (scalar for each pair of positions) -> (head_size/2)
        freqs = 1.0 / (theta ** (torch.arrange(0,head_size,2).float() / head_size))
        # m (represents the postion in the time dim) -> (T)
        m = torch.arrange(T).float()
        # Outer product -> (T,head_size/2)
        angles = torch.outer(m,freqs)
        # Double each entries to make (T, head_size)
        # First I used the repeat() but actually it won't work because it will double but it's like [f0,f1,f0,f1,..]
        # But what I want was [f0,f0,f1,f1...] -> hence this
        emb = torch.cat((angles,angles),dim = -1)
        
        # For efficiency, those are constants so we need to register buffer for them
        # And also, creating additional dimension (for batches) and apply cos and sin
        self.register_buffer('cos_cached',emb.cos()[None,None, :,:]) # (1,1,T,D)
        self.register_buffer('sin_cached',emb.sin()[None,None, :,:])
    def forward(self,x):
        # D represents the head_size
        B, T, D = x.shape
        # Manipulate the input to perfrom the rotation trick
        # Split in two, even and odd
        x_ev = x[...,0::2]
        x_od = x[...,1::2]
        # Stack the EVEN (negative) on top of the ODD, and stack along the CHANNEL/HEAD_SIZE dimension 
        x_trick = torch.stack([-x_od, x_ev], dim = -1) 
        # x_trick (B,T,D/2,2)
        
        # Flatten back the x_trick to use for the rotation (across the D/2 dimension)
        x_rot = x_trick.flatten(-2)
        # x_rot (B,T,D)
        
        
        # Take the cos and sin (1,T,D)
        cos = self.cos_cached[:,:,:T,:]
        sin = self.cos_cached[:,:,:T,:]
        
        x_rotated = (x * cos) + (x_rot * sin)
        # x_rotated (B,T,D) -> same size as input
        return x_rotated 
        
