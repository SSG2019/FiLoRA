import csv
import math
import torch
import os, random, numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from filoralib import prepare_glue_dataset, create_glue_dataloaders, select_lora_modules, build_ag_loader, compute_AG, get_UV, apply_filora, evaluate


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
batch_size_train = 16
rank = 32
learn_ratio = 3e-4

num_labels = 2
batch_size_AG = 32
data_ratio_AG = 0.01
data_ROOT = r"E:/data/GLUE/CoLA"
task_name = 'cola'

set_seed(seed)


##############
# Tokenizer/Model Loading & Device Placement
##############
model_name = "E:/model/DeBERTa/DeBERTa_v3_base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_classify = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_classify.to(device)


##############
# Training Loss Definition
##############
def loss_fn(outputs, batch):
    logits = outputs.logits
    labels = batch["labels"]
    return torch.nn.functional.cross_entropy(logits, labels)


#######################################################################################################################
##############
# Dataset Preparation & AG Statistics (U/V)
##############
# mrpc_dataset = load_dataset("glue", "mrpc")
# mrpc_dataset.save_to_disk(data_ROOT)

dataset = prepare_glue_dataset(load_from_disk(data_ROOT), task_name=task_name, tokenizer=tokenizer)
train_loader, val_loader = create_glue_dataloaders(dataset, task_name=task_name, tokenizer=tokenizer, batch_size=batch_size_train)
layer_names = select_lora_modules(model_classify, [0,11], ['key_proj', 'query_proj', 'value_proj', 'dense'])
AG_get_loader = build_ag_loader(dataset["train"], ratio=data_ratio_AG, batch_size=batch_size_AG, tokenizer=tokenizer, seed=0)
stats_AG = compute_AG(model=model_classify, layer_refs=layer_names, dataloader=AG_get_loader, loss_fn=loss_fn, group_size=len(layer_names))
stats_UV = get_UV(stats_AG, rank)
#######################################################################################################################


##############
# Apply FiLoRA Adapters
##############
model_filora = apply_filora(model_classify, stats_UV, rank=rank)


##########################################################################################################################
##############
# Select Trainable Parameters (LoRA .R & Classifier)
##############
for n, p in model_filora.named_parameters():
    if n.endswith(".R"):
        p.requires_grad_(True)
    elif "classifier" in n:
        p.requires_grad_(True)
    else:
        p.requires_grad_(False)

trainable_params = [p for p in model_filora.parameters() if p.requires_grad]

print(f"Total parameters to train: {sum(p.numel() for p in trainable_params):,}")
print(f"Parameters in LoRA .R: {sum(p.numel() for n, p in model_filora.named_parameters() if n.endswith('.R') and p.requires_grad):,}")
print(f"Parameters in classifier: {sum(p.numel() for n, p in model_filora.named_parameters() if 'classifier' in n and p.requires_grad):,}")
assert len(trainable_params) > 0, "No parameters requiring training were found."


##############
# Optimizer & Scheduler (Cosine with Warmup)
##############
optimizer = torch.optim.AdamW(trainable_params, lr=learn_ratio, weight_decay=0.0)
total_steps = epochs*len(train_loader)
warmup_steps = int(total_steps * 0.03)
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


##############
# Training Loop & Periodic Evaluation
##############

train_log_file = f"result/train_loss_FiLoRA.csv"
eval_log_file = f"result/eval_score_FiLoRA.csv"
global_step = 0
train_loss_records = []   # [(step, loss)]
eval_score_records = []   # [(step, score)]

for epoch in range(1, epochs + 1):
    model_filora.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k not in ("labels", "label")}
        outputs = model_filora(**inputs)
        loss = loss_fn(outputs, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        step_loss = loss.item()
        total_loss += step_loss
        train_loss_records.append((global_step, step_loss))

        if global_step % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] Step {global_step} | Train Loss: {step_loss:.6f}")


        del outputs, loss
        torch.cuda.empty_cache()

    eval_score = evaluate(model_filora, val_loader, device, task_name=task_name, epoch=epoch)
    eval_score_records.append((epoch, eval_score["MCC"]))

with open(train_log_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["global_step", "train_loss"])
    writer.writerows(train_loss_records)

with open(eval_log_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eval_score"])
    writer.writerows(eval_score_records)