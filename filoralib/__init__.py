from .layers import replace_one_with_filora, merge_filora_layers
from .dataprocess import (build_ag_loader, save_tensor_data_hdf5, load_tensor_data_hdf5, prepare_glue_dataset,
                          create_glue_dataloaders, evaluate)
from .utils import list_linear_layers, select_lora_modules, compute_AG, apply_filora, compute_UV_from_AG, get_UV
from .contrast_algorithm import compute_UV_LoRA_XS, compute_UV_random, replace_layers_with_vera

# Placeholder for future imports from layers.py
# from .layers import SvdLoraLinear

# Package-level metadata
__version__ = "0.1.0"
__author__ = "Cranky Monster"