import numpy as np
import torch
import math

d_model = 16
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
print(torch.sin(div_term))
print(torch.cos(div_term))