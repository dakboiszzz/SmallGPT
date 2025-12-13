import torch
class RotEmb():
    def __init__(self, head_size, T, theta = 10000):
        self.head_size = head_size
        self.theta = theta
        self.T = T
    def prepare_sin_cos(self):
        # freqs (scalar for each pair of positions) -> (head_size/2)
        freqs = 1.0 / (self.theta ** (torch.arrange(0,self.head_size,2).float() / self.head_size))
        # m (represents the postion in the time dim) -> (T)
        m = torch.arrange(self.T).float()
        # Outer product -> (T,head_size/2)
        angles = torch.outer(m,freqs)
        # Take the cos and sin, the results are both (T,head_size/2)
        cos_val = angles.cos()
        sin_val = angles.sin()
        # Add a dimension at the first shape to use in batches
        cos_val = cos_val.unsqueeze(0)
        sin_val = sin_val.unsqueeze(0)
        # Now both are (1,T,head_size/2)
        return cos_val,sin_val
    def apply_rope(self,x):
        # D represents the head_size
        B, T, D = x.shape
        # Manipulate the input to perfrom the rotation trick
        # Split in two, even and odd
        x_ev = x[...,0::2]
        x_od = x[...,1::2]
        # Stack the EVEN (negative) on top of the ODD, and stack along the CHANNEL/HEAD_SIZE dimension 
        x_trick = torch.stack([-x_ev, x_od], dim = -1) 
        # x_trick (B,T,D/2,2)
        
        # Flatten back the x_trick to use for the rotation (across the D/2 dimension)
        x_rot = x_trick.flatten(-2)
        # x_rot (B,T,D)
        
        
        # Take the cos and sin (1,T,D/2)
        cos,sin = self.prepare_sin_cos()
        # Make the dimensions fit -> want (1,T,D/2) to be (B,T,D) -> repeat along 3 dim with (B,1,2) times accordingly
        cos = cos[:,:T,:].repeat(B,1,2)
        sin = sin[:,:T,:].repeat(B,1,2)
        # Now our cos sin are both B,T,D -> Perform the rotation
        
        x_rotated = (x * cos) + (x_rot * sin)
        # x_rotated (B,T,D) -> same size as input
        return x_rotated 
        
