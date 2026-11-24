"""Train the 3.8M-parameter char-level TinyLlama used by the emulator.

  python experiments/emulator/train_char.py --data data/tinyshakespeare.txt --steps 700
"""
import argparse
import time

import torch
import torch.nn.functional as F

from mkparity.tinyllama import CFG, TinyLlama

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/tinyshakespeare.txt")
ap.add_argument("--out", default="data/ckpt.pt")
ap.add_argument("--steps", type=int, default=700)
args = ap.parse_args()

torch.manual_seed(0)
torch.set_num_threads(2)
text = open(args.data).read()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
n = int(0.9 * len(data))
train, val = data[:n], data[n:]
CFG["vocab"] = len(chars)
model = TinyLlama(CFG)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
T, B, steps = 128, 16, args.steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)


def batch(d):
    ix = torch.randint(len(d) - T - 1, (B,))
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


t0 = time.time()
for s in range(steps):
    x, y = batch(train)
    loss = F.cross_entropy(model(x).view(-1, CFG["vocab"]), y.view(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if s % 50 == 0 or s == steps - 1:
        with torch.no_grad():
            xv, yv = batch(val)
            vl = F.cross_entropy(model(xv).view(-1, CFG["vocab"]), yv.view(-1)).item()
        print(f"step {s} train {loss.item():.3f} val {vl:.3f} t={time.time()-t0:.0f}s", flush=True)
torch.save({"state": model.state_dict(), "cfg": CFG, "chars": chars}, args.out)
print("saved")
