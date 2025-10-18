import os, re, glob, statistics
from typing import Dict, Tuple, List

def summarize_logs(log_dir: str, task: str) -> Tuple[int, float, float, Dict[str, float]]:
    t = task.lower()

    if t == "cola":
        pat = re.compile(r"cola\s+mcc\s*=\s*([0-9]*\.?[0-9]+)", re.I)
    elif t == "sts-b":
        pat = re.compile(r"sts-b\s+pearson\s*=\s*([0-9]*\.?[0-9]+)", re.I)
    else:
        pat = re.compile(r"accuracy\s*=\s*([0-9]*\.?[0-9]+)", re.I)

    def best_from_file(path: str):
        best = None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    v = float(m.group(1))
                    best = v if best is None or v > best else best
        return best

    files = sorted(glob.glob(os.path.join(log_dir, "seed_*.log")))
    if not files:
        raise FileNotFoundError(f"No log files matched: {os.path.join(log_dir, 'seed_*.log')}")

    per_file: Dict[str, float] = {}
    vals: List[float] = []
    for p in files:
        b = best_from_file(p)
        per_file[os.path.basename(p)] = b if b is not None else float("nan")
        if b is not None:
            vals.append(b)

    if not vals:
        raise RuntimeError("No metric values parsed from logs. Check task name or log format.")

    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return len(vals), mean, std, per_file

results = {}

def collect(task_label, path, task_key):
    r, m, s, d = summarize_logs(path, task_key)
    results[task_label] = {"runs": r, "mean": m, "std": s, "detail": d}
    print(task_label + ":", r, f"{m:.4f}", f"{s:.4f}", d)
    return m

t = 0
t += collect("MRPC", r'./experiment/GLUE_MRPC/log', "mrpc")
t += collect("CoLA", r'./experiment/GLUE_COLA/log', "cola")
t += collect("STS-B", r'./experiment/GLUE_STSB/log', "sts-b")
t += collect("SST-2", r'./experiment/GLUE_SST2/log', "sst-2")
# t += collect("QNLI", r'./experiment/GLUE_QNLI/log', "qnli")
t += collect("RTE",  r'./experiment/GLUE_RTE/log',  "rte")
print(f"All: {t/6:.4f}")