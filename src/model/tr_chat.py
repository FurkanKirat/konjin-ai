#!/usr/bin/env python3
"""Talk to the Turkish chat model (needs minigpt.py in the same folder).

    python tr_chat.py --load models\\tr2.pt
    python tr_chat.py --load models\\tr2.pt --temp 0.6

Type /yeni to forget the conversation, an empty line to quit.
The prompt format is the one written by prep_tr.py chat (Kullanici:/Asistan: turns, then '// out:').
"""
import argparse
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d) for _d in ("model", "datagen", "eval")]

MARK = "// out:\n"


def fmt_prompt(turns):
    lines = [("Kullanici" if r == "user" else "Asistan") + ": " + t for r, t in turns]
    return "\n".join(lines) + "\n" + MARK


def fit_history(turns, budget):
    """Drop the oldest turns until the prompt fits `budget` bytes; history must start with a user turn."""
    turns = list(turns)
    while turns and (turns[0][0] != "user" or len(fmt_prompt(turns).encode("utf-8")) > budget):
        turns = turns[1:]
    return turns


def load(path):
    import torch
    import minigpt as mg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device, weights_only=True)
    model = mg.build(ck["cfg"], device)
    model.load_state_dict(ck["model"])
    vocab = ck["vocab"]

    def generate(prompt_bytes, n, temp):
        return mg.sample_bytes(model, vocab, prompt_bytes, n, temp, device).decode("utf-8", errors="replace")

    return generate, ck["cfg"]["T"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=300, help="max reply length in bytes")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    generate, T = load(a.load)
    history = []
    print("Model hazir. /yeni = yeni konusma, bos satir = cikis.\n")
    while True:
        try:
            s = input("Sen: ").strip()
        except EOFError:
            break
        if not s:
            break
        if s == "/yeni":
            history = []
            print("(konusma sifirlandi)\n")
            continue
        history.append(("user", s))
        history = fit_history(history, T - a.max_new - 8)
        if not history:
            print("(mesaj cok uzun)\n")
            continue
        reply = generate(fmt_prompt(history).encode("utf-8"), a.max_new, a.temp).strip()
        print("Model:", reply, "\n")
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()
