import pfns4bo
import torch
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping

# For HEBO+ 
model_path = pfns4bo.hebo_plus_model
# For BNN
# model_path = pfns4bo.bnn_model

# for correctly specified search spaces (e.g. correctly applied log transformations)
pfn_bo = TransformerBOMethod(torch.load(model_path), device='cuda:1')

# for mis-specified search spaces
pfn_bo = TransformerBOMethod(torch.load(model_path), fit_encoder=fit_input_warping, device='cuda:1')