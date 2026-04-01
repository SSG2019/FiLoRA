import re
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Tuple, Pattern, Union, Optional
from filoralib.layers import replace_one_with_filora

def _format_indices(indices: List[int]) -> str:
    """
    Helper function to format a list of indices into a compact string.
    Example: [0, 1, 2, 5, 6, 10] -> "0-2, 5-6, 10"
    """
    if not indices:
        return ""

    indices = sorted(list(set(indices)))
    groups = []
    start = end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            if start == end:
                groups.append(f"{start}")
            else:
                groups.append(f"{start}-{end}")
            start = end = indices[i]

    if start == end:
        groups.append(f"{start}")
    else:
        groups.append(f"{start}-{end}")

    return ", ".join(groups)


def list_linear_layers(
        model: nn.Module,
        layer_pattern: Pattern = re.compile(r'\.(\d+)\.')
) -> None:
    """
    Analyzes a model's architecture to identify and group layers with identical
    structures of nn.Linear sub-modules, showing the full module path template.

    Args:
        model (nn.Module):
            The PyTorch model to be analyzed.
        layer_pattern (re.Pattern):
            A regular expression pattern to identify the layer number in a module name.
            The pattern must contain one capturing group for the number.
            Default: r'\\.(\\d+)\\.' (matches 'layers.15.attn').
    """
    potential_blocks = defaultdict(lambda: defaultdict(list))

    linear_modules = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}

    for name in linear_modules:
        match = layer_pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            prefix = name[:match.start()]
            # relative_path is the part AFTER 'prefix.number.'
            relative_path = name[match.end(0):]

            potential_blocks[prefix][layer_idx].append(relative_path)

    print("-" * 80)
    print(f"{'Model Architectural Analysis':^80}")
    print("-" * 80)

    if not potential_blocks:
        print("Could not identify any repeating layer structures based on the provided pattern.")
        print(f"Total nn.Linear modules found: {len(linear_modules)}")
        print("-" * 80)
        return

    total_layers_grouped = 0
    for prefix, layers in potential_blocks.items():
        print(f"\nAnalyzing Block Prefix: '{prefix}'")

        structure_map = defaultdict(list)
        for layer_idx, modules in layers.items():
            fingerprint = tuple(sorted(modules))
            structure_map[fingerprint].append(layer_idx)

        if not structure_map:
            continue

        # Step 3: Print the grouped results with the FULL path template
        for fingerprint, indices in structure_map.items():
            formatted_indices = _format_indices(indices)
            print(f"\n  [+] Layers {formatted_indices} ({len(indices)} layers) share this structure:")
            for module_part in fingerprint:
                # --- THIS IS THE KEY CHANGE ---
                # We construct the full path template using the prefix and a placeholder
                print(f"    - {prefix}.{{...}}.{module_part}")
            total_layers_grouped += len(indices)

    # Find and print ungrouped layers
    grouped_names = set()
    for prefix, layers in potential_blocks.items():
        for idx in layers:
            match_str = f"{prefix}.{idx}."
            for name in linear_modules:
                if name.startswith(match_str):
                    grouped_names.add(name)

    ungrouped_names = set(linear_modules.keys()) - grouped_names
    if ungrouped_names:
        print("\n  [+] Other nn.Linear modules (not part of a detected repeating structure):")
        for name in sorted(list(ungrouped_names)):
            print(f"    - {name}")

    print("-" * 80)
    print(f"Analysis complete. Found {total_layers_grouped} linear modules within repeating structures.")
    print("-" * 80)


def select_lora_modules(
        model: nn.Module,
        layer_indices: Union[List[int], int, str],  # 现在支持范围列表 [start, end]
        target_blocks: List[str] = ["q_proj", "v_proj"]
) -> List[str]:
    """
    Selects specific module names for LoRA application based on layer indices and block names.

    Args:
        model (nn.Module): The PyTorch model.
        layer_indices (Union[List[int], int, str]):
            The indices of the transformer layers to target.
            - Can be a list of two integers [start, end] to target a range.
            - Can be a single integer for one layer.
            - Can be the string "all" to target all layers.
        target_blocks (List[str]):
            A list of keywords identifying the specific blocks within the attention
            or MLP modules to adapt.

    Returns:
        List[str]: A list of fully-qualified module names selected for LoRA adaptation.

    Raises:
        ValueError: If layer_indices is invalid.
    """
    selected_modules = []

    # --- Determine the total number of layers for "all" keyword ---
    max_layer_idx = -1
    for name, _ in model.named_modules():
        matches = re.findall(r'\.(h|layer|block)\.(\d+)\.', name)
        if matches:
            for _, idx_str in matches:
                idx = int(idx_str)
                if idx > max_layer_idx:
                    max_layer_idx = idx

    # --- Process layer_indices for ranges ---
    if isinstance(layer_indices, str) and layer_indices.lower() == "all":
        if max_layer_idx == -1:
            print("Warning: Could not determine the number of layers automatically. No modules selected.")
            return []
        indices_to_target = list(range(max_layer_idx + 1))
    elif isinstance(layer_indices, int):
        indices_to_target = [layer_indices]
    elif isinstance(layer_indices, list) and len(layer_indices) == 2:
        # Handle [start, end] range
        start, end = layer_indices
        if start > end:
            raise ValueError(f"Invalid range: start index {start} cannot be greater than end index {end}.")
        indices_to_target = list(range(start, end + 1))
    elif isinstance(layer_indices, list):
        # Handle invalid list format
        raise ValueError(f"Invalid format for layer_indices: {layer_indices}. Expected a list with two integers.")
    else:
        raise ValueError(f"Unsupported type for layer_indices: {type(layer_indices)}")

    # --- Iterate and select modules ---
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check if the module name contains any of the target block keywords
        is_target_block = any(block in name for block in target_blocks)
        if not is_target_block:
            continue

        # Check if the module belongs to one of the target layer indices
        match = re.search(r'\.(h|layer|block)\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(2))
            if layer_idx in indices_to_target:
                selected_modules.append(name)

    return sorted(list(set(selected_modules)))  # Sort and unique

LayerRef = Union[str, nn.Linear]


def compute_AG(
    model: nn.Module,
    layer_refs: List[LayerRef],
    dataloader,
    loss_fn,
    group_size: int = 4,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:

    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    model.eval()

    def _resolve_layer(ref: LayerRef) -> Tuple[str, nn.Linear]:
        if isinstance(ref, nn.Linear):
            name = None
            for n, m in model.named_modules():
                if m is ref:
                    name = n
                    break
            return (name or f"<unnamed_linear_{id(ref)}>", ref)
        obj, name = model, ref
        for p in ref.split("."):
            obj = getattr(obj, p)
        if not isinstance(obj, nn.Linear):
            raise TypeError(f"{ref} is not nn.Linear")
        return name, obj

    def _flat2d(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t

    def _to_device_batch(b: dict) -> dict:
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    layers = [_resolve_layer(r) for r in layer_refs]
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for i in range(0, len(layers), group_size):
        group = layers[i:i+group_size]
        handles = []

        for name, lin in group:
            d_in, d_out = lin.in_features, lin.out_features
            A_sum = torch.zeros(d_in, d_in, device=device)
            G_sum = torch.zeros(d_out, d_out, device=device)
            N_t   = torch.zeros(1, device=device)

            def fwd_hook(mod, inp, out, A_acc=A_sum, G_acc=G_sum, N_cnt=N_t, layer_name=name):
                # print(inp[0].detach().shape)
                # is_pos = (inp[0].detach().size(0) == 1 and inp[0].detach().size(1) == 512 and inp[0].detach().size(2) == 768)
                # if is_pos:
                #     return
                x = _flat2d(inp[0].detach())     # [Ntok, d_in]
                # print(x.T @ x)
                A_acc.add_(x.T @ x)
                N_cnt.add_(x.size(0))
                # print(f"[fwd:{layer_name}] x={x.size(0)}, N={float(N_cnt.item())}")

                # print(x.T @ x)
                out.requires_grad_(True)
                def bwd_hook(grad_out):
                    delta = _flat2d(grad_out)    # [Ntok, d_out]
                    G_acc.add_(delta.T @ delta)
                    # print(f"[bwd:{layer_name}] x={delta.size(0)}")
                out.register_hook(bwd_hook)

            handles.append((name, A_sum, G_sum, N_t, lin.register_forward_hook(fwd_hook)))

        for batch in dataloader:
            batch = _to_device_batch(batch)
            model.zero_grad(set_to_none=True)

            inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**inputs)

            loss = loss_fn(outputs, batch)
            loss.backward()

            del outputs, loss
            torch.cuda.empty_cache()

        for name, A_sum, G_sum, N_t, h in handles:
            h.remove()
            N = max(1.0, float(N_t.item()))
            A = 0.5 * ((A_sum / N) + (A_sum / N).T)
            G = 0.5 * ((G_sum / N) + (G_sum / N).T)

            results[name] = (A, G)

        del group, handles
        torch.cuda.empty_cache()

    return results


def compute_UV_from_AG(
    A: torch.Tensor,
    G: torch.Tensor,
    rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    eval_A, evec_A = torch.linalg.eigh(A)
    idxA = torch.argsort(eval_A, descending=True)[:rank]
    V_r = evec_A[:, idxA].contiguous()

    eval_G, evec_G = torch.linalg.eigh(G)
    idxG = torch.argsort(eval_G, descending=True)[:rank]
    U_r = evec_G[:, idxG].contiguous()


    return U_r, V_r

def get_UV(stats, rank):
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_name, (A, G) in stats.items():
        U, V, = compute_UV_from_AG(A, G, rank)
        results[layer_name] = (U,V)
    return results


def apply_filora(model, stats, rank):
    if stats:
        first_layer_name = next(iter(stats.keys()))
        U, V = stats[first_layer_name]
        n_columns = U.size(1)
        assert rank <= n_columns, f"Rank {rank} must be <= number of columns {n_columns}"

    for layer_name, (U, V) in stats.items():
        U = U[:, :rank].contiguous()
        V = V[:, :rank].contiguous()
        # Replace this layer in the model
        replace_one_with_filora(model, layer_name, U, V)

    return model


