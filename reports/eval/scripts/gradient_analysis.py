"""Read-only weight-change and gradient analysis. No optimizer is created and no file
of the run is written; the script asserts the weights are bit-identical afterwards.
    python reports/eval/scripts/gradient_analysis.py <init.pt> <current.pt> <dataset_dir> <out.json>
"""
import hashlib, json, os, shutil, sys, tempfile
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from evaluate.checkpoint_io import load_model
from agent.tensorize import to_tensors

init_p, cur_p, data_dir, out_p = sys.argv[1:5]
tmp = tempfile.mkdtemp()
cur_copy = shutil.copy(cur_p, os.path.join(tmp, "cur.pt"))   # never touch the live file
torch.set_num_threads(2)
A, B = load_model(init_p).model, load_model(cur_copy).model
sa, sb = A.state_dict(), B.state_dict()

def group(name):
    p = name.split(".")
    return ".".join(p[:2]) if p[0] in ("policy_heads", "aux") else p[0]

def digest(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()): h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

before = digest(B)
# ---- 1. what training has changed so far (init -> current)
chg = {}
for k in sa:
    if not sa[k].dtype.is_floating_point: continue
    g = chg.setdefault(group(k), [0.0, 0.0, 0])
    g[0] += float((sb[k] - sa[k]).pow(2).sum()); g[1] += float(sa[k].pow(2).sum()); g[2] += sa[k].numel()
# ---- 2. gradient of the imitation loss on a fixed batch, weights frozen (no step)
rows = [json.loads(l) for l in open(os.path.join(data_dir, "index.jsonl"))]
rows = [r for r in rows if r["stage"] == "S1"][:24]
B.train(False)
for p in B.parameters(): p.grad = None
tot_steps = 0
for r in rows:
    z = np.load(os.path.join(data_dir, os.path.basename(r["file"])))
    T = min(len(z["action"]), 128)
    obs = {k: z[k][:T][:, None] for k in z.files if k not in ("action", "executed_action", "engine_reward", "done", "teacher_event", "driver")}
    obs = {k: v for k, v in to_tensors(obs).items()}
    start = torch.zeros(T, 1, dtype=torch.bool); start[0, 0] = True
    out, _ = B.forward_sequence(obs, B.initial_state(1), start)
    lab = torch.as_tensor(z["action"][:T]).long()
    loss = sum(torch.nn.functional.cross_entropy(out.logits[i][:, 0], lab[:, i], reduction="sum") for i in range(6))
    loss.backward(); tot_steps += T
grad = {}
for k, p in B.named_parameters():
    if p.grad is None: continue
    g = grad.setdefault(group(k), [0.0, 0.0])
    g[0] += float((p.grad / tot_steps).pow(2).sum()); g[1] += float(p.detach().pow(2).sum())
after = digest(B)
assert before == after, "weights changed"
res = {"init": init_p, "current": cur_p, "weights_unchanged_sha256": after, "imitation_batch_steps": tot_steps, "groups": {}}
for g in chg:
    c = chg[g]; gr = grad.get(g, [0.0, 1.0])
    res["groups"][g] = {"params": c[2], "rel_change": (c[0] ** .5) / max(c[1] ** .5, 1e-12), "grad_norm": gr[0] ** .5, "grad_over_weight": (gr[0] ** .5) / max(gr[1] ** .5, 1e-12)}
json.dump(res, open(out_p, "w"), indent=1)
print("weights unchanged:", before == after, "| steps", tot_steps)
for g, v in sorted(res["groups"].items(), key=lambda kv: -kv[1]["rel_change"]):
    print(f"{g:28s} params {v['params']:7d}  changed {100*v['rel_change']:6.2f}%  grad {v['grad_norm']:.4f}  grad/weight {v['grad_over_weight']:.5f}")
shutil.rmtree(tmp, ignore_errors=True)
