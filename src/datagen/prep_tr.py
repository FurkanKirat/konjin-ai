#!/usr/bin/env python3
"""Turn downloaded Hugging Face Turkish datasets into plain .txt files for minigpt.py (needs: pip install pyarrow).

Stage 1 - language (plain text):
    python prep_tr.py wiki --src hf\\wiki_tr --out data_tr\\wiki --max-mb 800

Stage 2 - chat. One file per (conversation so far -> next assistant reply). The file looks like
    Kullanici: Merhaba!
    Asistan: Merhaba, nasilsin?
    Kullanici: Iyiyim, sen?
    // out:
    <the assistant reply>
and minigpt.py --out-only trains only on the reply (+ EOF), so the model learns to answer and to stop.
    python prep_tr.py chat --src hf\\conv_tur --out data_tr\\chat\\conv
    python prep_tr.py chat --src hf\\instr_nr --out data_tr\\chat\\instr
Supported columns: 'messages' / 'conversations' (list of {role, content} or {from, value}) or 'instruction' (+ optional 'input') + 'output'.
Any other layout: the script prints the first record so the layout can be added.
"""
import argparse
import json
import os
import random
import sys

MARK = "// out:\n"


def find_files(src, exts):
    out = []
    for dp, _, fn in os.walk(src):
        for f in fn:
            if f.lower().endswith(exts):
                out.append(os.path.join(dp, f))
    return sorted(out)


def iter_records(path):
    low = path.lower()
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            yield from batch.to_pylist()
    elif low.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif low.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):  # {"data": [...]} style
            data = next((v for v in data.values() if isinstance(v, list)), [])
        for rec in data:
            if isinstance(rec, dict):
                yield rec


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ---------------------------------------------------------------- stage 1
def cmd_wiki(a):
    files = find_files(a.src, (".parquet",))
    if not files:
        raise SystemExit(f"No .parquet files under {a.src}")
    random.Random(a.seed).shuffle(files)  # shards in random order so a size cap does not keep only one alphabet range
    os.makedirs(a.out, exist_ok=True)
    cap, chunk_cap = int(a.max_mb * 1e6), int(a.chunk_kb * 1000)
    total = k = n_art = 0
    buf, buf_n = [], 0

    def flush():
        nonlocal buf, buf_n, k
        if buf:
            write_text(os.path.join(a.out, f"w{k:05d}.txt"), "".join(buf))
            k += 1
            buf, buf_n = [], 0

    for f in files:
        for rec in iter_records(f):
            title, text = (rec.get("title") or "").strip(), (rec.get("text") or "").strip()
            if len(text) < a.min_chars:
                continue
            art = f"{title}\n\n{text}\n\n\n".replace("\r\n", "\n")
            b = len(art.encode("utf-8"))
            buf.append(art)
            buf_n += b
            total += b
            n_art += 1
            if buf_n >= chunk_cap:
                flush()
            if total >= cap:
                break
        if total >= cap:
            break
    flush()
    print(f"wrote {k} files, {n_art} articles, {total / 1e6:.0f} MB to {a.out}")


# ---------------------------------------------------------------- stage 2
def norm_role(r):
    r = (r or "").lower()
    if r in ("user", "human", "kullanici"):
        return "user"
    if r in ("assistant", "gpt", "bot", "asistan", "model"):
        return "assistant"
    return None  # system etc. are dropped


def record_turns(rec):
    """-> list of (role, text) or None if the layout is unknown."""
    for key in ("messages", "conversations", "conversation"):
        if isinstance(rec.get(key), list):
            turns = []
            for m in rec[key]:
                if not isinstance(m, dict):
                    return None
                role = norm_role(m.get("role", m.get("from")))  # OpenAI style or ShareGPT style
                text = str(m.get("content", m.get("value")) or "").strip()
                if role and text:
                    turns.append((role, text))
            return turns
    if rec.get("instruction") is not None and rec.get("output") is not None:
        q = str(rec["instruction"]).strip()
        if rec.get("input"):
            q += "\n" + str(rec["input"]).strip()
        return [("user", q), ("assistant", str(rec["output"]).strip())]
    return None


def fmt_prompt(turns):
    lines = [("Kullanici" if r == "user" else "Asistan") + ": " + t.replace("\r\n", "\n") for r, t in turns]
    return "\n".join(lines) + "\n" + MARK


def examples_from(turns, max_bytes):
    """Every assistant turn becomes one example (history -> reply). Oldest turns are dropped to fit max_bytes."""
    out = []
    for i, (role, reply) in enumerate(turns):
        if role != "assistant" or i == 0:
            continue
        hist = turns[:i]
        while hist and (hist[0][0] != "user" or len((fmt_prompt(hist) + reply).encode("utf-8")) > max_bytes):
            hist = hist[1:]
        if hist and hist[-1][0] == "user" and MARK not in reply:
            out.append(fmt_prompt(hist) + reply.replace("\r\n", "\n"))
    return out


def cmd_chat(a):
    files = find_files(a.src, (".parquet", ".jsonl", ".json"))
    if not files:
        raise SystemExit(f"No .parquet/.jsonl/.json files under {a.src}")
    examples, n_rec, skipped = [], 0, 0
    for f in files:
        try:
            for rec in iter_records(f):
                turns = record_turns(rec)
                if turns is None:
                    if n_rec == 0 and skipped == 0:
                        print(f"Unknown layout in {f}. First record:\n{json.dumps(rec, ensure_ascii=False)[:800]}")
                    skipped += 1
                    continue
                n_rec += 1
                examples += examples_from(turns, a.max_bytes)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"skipping {f}: {e}")
    if not examples:
        raise SystemExit("No examples produced (unknown column layout?). See the record printed above.")
    random.Random(a.seed).shuffle(examples)  # random order so the every-20th validation split is a fair sample
    os.makedirs(a.out, exist_ok=True)
    for i, ex in enumerate(examples):
        write_text(os.path.join(a.out, f"c{i:06d}.txt"), ex)
    avg = sum(len(e.encode("utf-8")) for e in examples) / len(examples)
    print(f"{n_rec} records ({skipped} with unknown layout) -> {len(examples)} examples, avg {avg:.0f} bytes, in {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wiki", help="Wikipedia-style parquet (title, text) -> chunked plain text")
    w.add_argument("--src", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--max-mb", type=float, default=800)
    w.add_argument("--chunk-kb", type=float, default=200, help="size of each output file")
    w.add_argument("--min-chars", type=int, default=500, help="skip very short articles")
    w.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("chat", help="chat/instruction data -> one file per (history -> reply)")
    c.add_argument("--src", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-bytes", type=int, default=1000, help="prompt+reply limit; must fit the model block (1024)")
    c.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd_wiki(a) if a.cmd == "wiki" else cmd_chat(a)


if __name__ == "__main__":
    main()