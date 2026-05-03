import torch
import torch.nn.functional as F

class WinCLIP:
    def __init__(self, model, device = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        

