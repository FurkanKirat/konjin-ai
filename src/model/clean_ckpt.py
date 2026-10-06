#!/usr/bin/env python3
"""Make a checkpoint safe and small for sharing: fp16 weights, and a meta block without local paths.

    python src\\model\\clean_ckpt.py models\\bp3.pt release\\konjin-boilerplate.pt
    python src\\model\\clean_ckpt.py models\\tr2.pt release\\konjin-turkish-chat.pt

The result loads with the normal --load / tr_chat.py / bp_demo.py (fp16 weights are converted back on load).
"""
import os
import sys

import torch

KEEP = ("date", "steps", "steps_done", "lr", "bs", "fim", "val_loss", "chars")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=True)
    meta = {k: v for k, v in (ck.get("meta") or {}).items() if k in KEEP}
    out = {"model": {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()},
           "cfg": ck["cfg"], "vocab": ck["vocab"], "meta": meta}
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(out, dst)
    print(f"{src} ({os.path.getsize(src) / 1e6:.0f} MB) -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")
    print("kept meta:", meta)


if __name__ == "__main__":
    main()
