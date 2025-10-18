import math
import torch
import os, random, numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from filoralib import prepare_glue_dataset, create_glue_dataloaders, select_lora_modules, build_ag_loader, compute_AG, get_UV, apply_filora, evaluate
import time

##############
# Reproducibility: Set Random Seed
##############
def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # :16:8, :4096:8
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


##############
# Hyperparameters & Task Config
##############
seed = int(os.getenv("SEED", 0))
epochs = 25
batch_size_train = 8
rank = 32
learn_ratio = 2e-4

num_labels = 1
batch_size_AG = 8
data_ratio_AG = 0.015
data_ROOT = r"E:/data/GLUE/STS-B"
task_name = 'sts-b'

set_seed(seed)


##############
# Tokenizer/Model Loading & Device Placement
##############
model_name = "E:/model/DeBERTa/DeBERTa_v3_base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_classify = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="regression")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_classify.to(device)


##############
# Training Loss Definition
##############
mse_loss = torch.nn.MSELoss()
def loss_fn(outputs, batch):
    logits = outputs.logits.squeeze(-1)
    labels = batch["labels"].to(dtype=logits.dtype)
    return mse_loss(logits, labels)


#######################################################################################################################
##############
# Dataset Preparation & AG Statistics (U/V)
##############
# stsb_dataset = load_dataset("glue", "stsb")
# stsb_dataset.save_to_disk(data_ROOT)

dataset = prepare_glue_dataset(load_from_disk(data_ROOT), task_name=task_name, tokenizer=tokenizer)
train_loader, val_loader = create_glue_dataloaders(dataset, task_name=task_name, tokenizer=tokenizer, batch_size=batch_size_train)
layer_names = select_lora_modules(model_classify, [0,11], ['key_proj', 'query_proj', 'value_proj', 'dense'])
start_time = time.time()
AG_get_loader = build_ag_loader(dataset["train"], ratio=data_ratio_AG, batch_size=batch_size_AG, tokenizer=tokenizer, seed=42)
stats_AG = compute_AG(model=model_classify, layer_refs=layer_names, dataloader=AG_get_loader, loss_fn=loss_fn, group_size=len(layer_names))
stats_UV = get_UV(stats_AG, rank)
end_time = time.time()
#######################################################################################################################
run_time = end_time - start_time
print(f"initialization time: {run_time} s")
