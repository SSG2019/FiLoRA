import h5py
import torch
import numpy as np
from datasets import DatasetDict
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

def build_ag_loader(dataset, ratio, batch_size, tokenizer, seed=0, shuffle=True, count=None):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    n = len(dataset)
    if count is not None:
        # 按“个数”采样，自动转 int，并夹紧范围
        k = int(count)
        if k < 1:
            k = 1
        if k > n:
            k = n
    else:
        # 走原有比例逻辑
        assert 0.0 < ratio <= 1.0, "`ratio` 必须在 (0, 1] 内"
        k = max(1, int(n * ratio))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:k].tolist()

    # 只“选择”子集（零拷贝的arrow切片，不会重复处理数据）
    subset = dataset.select(idx)

    # 构造 A/G 专用 DataLoader
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

def save_tensor_data_hdf5(data: Dict[str, Tuple[torch.Tensor, torch.Tensor]], filename: str = 'tensor_data.h5'):
    """
    使用HDF5格式保存字典数据。
    字典的键是字符串，值是包含两个torch.Tensor的元组。

    Args:
        data (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): 要保存的字典，
                                                             例如 {'dynamic_key_str': (torch.Tensor, torch.Tensor)}。
        filename (str): 输出文件名。
    """
    with h5py.File(filename, 'w') as f:
        for dynamic_key, tensor_tuple in data.items():
            # 确保值是包含两个张量的元组
            if not isinstance(tensor_tuple, tuple) or \
               len(tensor_tuple) != 2 or \
               not all(isinstance(t, torch.Tensor) for t in tensor_tuple):
                raise ValueError(
                    f"键 '{dynamic_key}' 对应的值格式不正确。期望一个包含两个torch.Tensor的元组。"
                )

            tensor1, tensor2 = tensor_tuple

            # 将PyTorch Tensor转换为NumPy数组进行保存
            np_tensor1 = tensor1.cpu().numpy()
            np_tensor2 = tensor2.cpu().numpy()

            # 使用动态键作为 HDF5 组名
            # HDF5 组名不能包含 '/', '.', '\' 等特殊字符，这里假设您的键符合规范
            group_name = dynamic_key
            grp = f.create_group(group_name)

            # 保存第一个张量
            grp.create_dataset('tensor_0', data=np_tensor1)
            grp['tensor_0'].attrs['dtype'] = str(np_tensor1.dtype)
            grp['tensor_0'].attrs['shape'] = np_tensor1.shape

            # 保存第二个张量
            grp.create_dataset('tensor_1', data=np_tensor2)
            grp['tensor_1'].attrs['dtype'] = str(np_tensor2.dtype)
            grp['tensor_1'].attrs['shape'] = np_tensor2.shape

    print(f"数据已保存到 {filename}")

def load_tensor_data_hdf5(filename: str = 'tensor_data.h5', to_torch: bool = True) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    从HDF5文件加载字典数据。
    加载的数据将恢复为 Dict[str, Tuple[torch.Tensor, torch.Tensor]] 格式。

    Args:
        filename (str): 输入文件名。
        to_torch (bool): 如果为True，则将加载的NumPy数组转换为PyTorch Tensor。
                         如果为False，则返回NumPy数组。

    Returns:
        Dict[str, Tuple[torch.Tensor, torch.Tensor]]: 加载的数据字典，
                每个值形如 (torch.Tensor, torch.Tensor) 或 (np.ndarray, np.ndarray)。
    """
    loaded_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    try:
        with h5py.File(filename, 'r') as f:
            for group_name in f.keys():
                grp = f[group_name]

                # 加载第一个张量
                np_tensor1 = grp['tensor_0'][()]
                # 加载第二个张量
                np_tensor2 = grp['tensor_1'][()]

                if to_torch:
                    tensor1 = torch.from_numpy(np_tensor1)
                    tensor2 = torch.from_numpy(np_tensor2)
                else:
                    tensor1 = np_tensor1
                    tensor2 = np_tensor2

                # 使用组名作为动态键构建字典
                loaded_data[group_name] = (tensor1, tensor2)
        return loaded_data
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return {}



def prepare_glue_dataset(
        raw_datasets: DatasetDict,
        task_name: str,
        tokenizer,
        max_seq_length: int = 256,
) -> DatasetDict:

    sentence_keys = {
        "mrpc": ("sentence1", "sentence2"),
        "sst-2": ("sentence", None),
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "sts-b": ("sentence1", "sentence2"),
    }

    text_feature_name1, text_feature_name2 = sentence_keys[task_name]

    raw_datasets = raw_datasets.rename_column("label", "labels")

    # 2. Tokenization
    def tokenize_function(examples):
        if text_feature_name2:
            return tokenizer(
                examples[text_feature_name1],
                examples[text_feature_name2],
                padding=False,
                truncation=True,
                max_length=max_seq_length
            )
        else:
            return tokenizer(
                examples[text_feature_name1],
                padding=False,
                truncation=True,
                max_length=max_seq_length
            )

    disable_progress_bar()
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=[
            col for col in raw_datasets["train"].column_names
            if col not in ["labels"]
        ],
        desc="Running tokenizer on dataset",
    )
    tokenized_datasets.set_format("torch")
    return tokenized_datasets

def create_glue_dataloaders(
    tokenized_datasets: DatasetDict,
    task_name: str,
    tokenizer,
    batch_size: int
) -> tuple:

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,
    )

    if task_name == "mnli":
        val_matched_loader = DataLoader(
            tokenized_datasets["validation_matched"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        val_mismatched_loader = DataLoader(
            tokenized_datasets["validation_mismatched"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        return train_loader, val_matched_loader, val_mismatched_loader
    else:
        val_loader = DataLoader(
            tokenized_datasets["validation"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        return train_loader, val_loader

def evaluate(model, val_loader, device, task_name: str, epoch: int):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # 对分类任务：取 argmax
            if task_name.lower() != "sts-b":
                preds = logits.argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
            else:
                # 对 STS-B 是回归任务 → 直接取数值预测
                preds = logits.squeeze().cpu().tolist()
                labels = batch["labels"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

    task = task_name.lower()
    results = {}

    if task == "cola":
        mcc = matthews_corrcoef(all_labels, all_preds)
        results["MCC"] = mcc
        print(f"[Epoch {epoch}] CoLA MCC = {mcc:.4f}")

    elif task in ["mrpc", "qqp"]:
        acc = accuracy_score(all_labels, all_preds)
        results["accuracy"] = acc
        print(f"[Epoch {epoch}] {task.upper()} Accuracy = {acc:.4f}")

    elif task == "sts-b":
        # Pearson
        pearson = pearsonr(all_preds, all_labels)[0]
        spearman = spearmanr(all_preds, all_labels)[0]
        avg = (pearson + spearman) / 2
        results["pearson"] = pearson
        results["spearman"] = spearman
        results["avg"] = avg
        print(f"[Epoch {epoch}] STS-B pearson = {pearson:.4f} spearman = {spearman:.4f} avg = {avg:.4f}")

    else:  # SST-2, QNLI, MNLI
        acc = accuracy_score(all_labels, all_preds)
        results["accuracy"] = acc
        print(f"[Epoch {epoch}] {task.upper()} Accuracy = {acc:.4f}")

    return results


