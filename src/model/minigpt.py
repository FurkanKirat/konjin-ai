r"""
minigpt.py'nin PyTorch + CUDA sürümü. Aynı mimari (pre-LN GPT), ama:
  - GPU'da çalışır, autograd kullanır,
  - bayt seviyesinde çalışır (metin UTF-8 baytlarına çevrilir): Türkçe karakter, BOM, her şey sorunsuz,
  - eğitim verisini komut satırından verdiğin klasörlerden okur.

Kullanım:
    # kendi C# projelerin (alt klasörler dahil, bin/obj/Library gibi klasörler atlanır)
    python minigpt_torch.py --data C:\Projeler\Proje1 C:\Projeler\Proje2

    # kendi kodun + Python standart kütüphanesi (ekstra veri olarak)
    python minigpt_torch.py --data C:\Projeler\Proje1 --stdlib

    # başka uzantılar
    python minigpt_torch.py --data C:\Projeler\Proje1 --ext .cs .shader .py

    # --data vermezsen Python standart kütüphanesiyle eğitir
    python minigpt_torch.py

    # eğitilmiş modelden üret
    python minigpt_torch.py --load model.pt --prompt "public class "

    # yarım kalmış çok satırlı kodu dosyadan ver, deterministik devam ettir
    python minigpt_torch.py --load model.pt --prompt-file yarim.cs --temp 0 --tokens 150

    # sonraki-satır isabet testi (doğrulama dosyalarından; eğitimdeki --data/--ext/--stdlib/--chars ile AYNI ver)
    python minigpt_torch.py --load model.pt --data C:\Projeler\Proje1 --eval-lines 200

    # mevcut modelden devam et (yeni veri karışımıyla ince ayar); eskisinin üstüne YAZMAZ, --out farklı olmalı
    python minigpt_torch.py --init bcl.pt --out bcl2.pt --data C:\Projeler\Proje1 C:\Yeni\Repo --lr 2e-4 --steps 8000

    # düzenli yapı: KÖK/datasets/<kaynak>, KÖK/models/*.pt; hangi veri hangi modelde kullanıldı
    python minigpt_torch.py --list "C:\Users\furki\Documents\Model Eğitim"

    # FIM: prompt dosyasında boşluğun yerine <FILL> yaz (önek ve sonek modele verilir, arası doldurulur)
    python minigpt_torch.py --load model.pt --prompt-file bosluk.cs --temp 0.2 --tokens 150
    # FIM isabet testi (eğitimdeki veri argümanlarıyla aynı)
    python minigpt_torch.py --load model.pt --data C:\Projeler\Proje1 --eval-fim 200

    # bir modelin hangi modelden, hangi veriyle üretildiğini göster
    python minigpt_torch.py --info bcl2.pt

GPU yoksa küçük ayarlarla dene:
    python minigpt_torch.py --embd 64 --layers 2 --heads 4 --block 64 --bs 32 --steps 2000
"""
import argparse
import contextlib
import math
import os
import shutil
import sys
import sysconfig
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- veri
SKIP_DIRS = {"bin", "obj", "library", "temp", "packages", "node_modules",
             ".git", ".vs", ".idea", "__pycache__"}
STDLIB_SKIP = {"test", "tests", "idlelib", "lib2to3", "turtledemo",
               "site-packages", "dist-packages"}
GENERATED = (".designer.cs", ".g.cs", ".g.i.cs", ".generated.cs")
MAX_FILE_BYTES = 1_000_000
# UTF-8'de hiç geçmeyen baytlar = özel işaretler (vocab 256 kalır, eski checkpoint'ler uyumlu)
EOF_B, PRE_B, SUF_B, MID_B = 0xF8, 0xF9, 0xFA, 0xFB


def collect_files(roots, exts, skip, g0=0):
    """Klasörleri (alt klasörler dahil) gezer, uzantısı eşleşen dosyaları okur.
    Döner: [(yol, utf8_bayt, grup)]. Her klasör kendi grubunu alır (g0 + sıra numarası)."""
    out, seen = [], set()
    for k, root in enumerate(roots):
        if not os.path.isdir(root):
            raise SystemExit(f"Folder not found: {root}")
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d.lower() not in skip)
            for f in sorted(files):
                low = f.lower()
                if not low.endswith(exts) or low.endswith(GENERATED):
                    continue
                path = os.path.realpath(os.path.join(dirpath, f))
                if path in seen:
                    continue
                try:
                    if os.path.getsize(path) > MAX_FILE_BYTES:
                        continue
                    # utf-8-sig: BOM'u atar. Metin modu: \r\n -> \n. UTF-8 olmayan dosya atlanır.
                    with open(path, encoding="utf-8-sig") as fh:
                        s = fh.read()
                except (UnicodeDecodeError, OSError):
                    continue
                seen.add(path)
                out.append((path, s.encode("utf-8"), g0 + k))
    return out


def pick_val_ids(files):
    """Doğrulamaya ayrılan dosyaların indeksleri: her grupta (klasörde) her 20. dosya,
    20'den az dosya varsa son dosya. Tek dosyalı grup tamamen eğitimde kalır."""
    val_ids = set()
    for g in {f[2] for f in files}:
        ids = [i for i, f in enumerate(files) if f[2] == g]
        if len(ids) > 1:
            val_ids.update(ids[19::20] or ids[-1:])
    return val_ids


def split_files(files):
    """Dosya seviyesinde böl (bkz. pick_val_ids). Böylece doğrulama kaybı her kaynaktan
    (senin projelerin dahil) örnek içerir."""
    val_ids = pick_val_ids(files)
    eof = bytes([EOF_B])  # her dosyanın sonuna "dosya bitti" işareti
    train = b"".join(f[1] + eof for i, f in enumerate(files) if i not in val_ids)
    val = b"".join(f[1] + eof for i, f in enumerate(files) if i in val_ids)
    return train, val


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, C, nh, p):
        super().__init__()
        self.nh, self.p = nh, p
        self.ln1 = nn.LayerNorm(C)
        self.qkv = nn.Linear(C, 3 * C)
        self.proj = nn.Linear(C, C)
        self.ln2 = nn.LayerNorm(C)
        self.fc = nn.Linear(C, 4 * C)
        self.out = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nh, C // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))
        return x


class GPT(nn.Module):
    def __init__(self, V, C, L, nh, T, p):
        super().__init__()
        self.T = T
        self.wte = nn.Embedding(V, C)
        self.wpe = nn.Embedding(T, C)
        self.drop = nn.Dropout(p)
        self.blocks = nn.ModuleList(Block(C, nh, p) for _ in range(L))
        self.lnf = nn.LayerNorm(C)
        self.head = nn.Linear(C, V, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        for name, prm in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, 0, 0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def sample_bytes(model, vocab, ctx, n, temp, device, stop=EOF_B, no_empty=False):
    """ctx (bayt) sonrasını üretir, üretilen baytları döner. `stop` baytı gelirse durur (dahil değil).
    no_empty: boşluk-dışı ilk bayt üretilene kadar `stop` yasak (FIM'de 'hiçbir şey yazmama' kaçışını engeller)."""
    model.eval()
    to_id = {b: i for i, b in enumerate(vocab)}
    ids = [to_id[b] for b in ctx if b in to_id] or [to_id[10]]
    x = torch.tensor([ids], device=device)
    out = []
    seen = not no_empty
    for _ in range(n):
        logits, _ = model(x[:, -model.T:])
        lg = logits[0, -1].float()
        if not seen and stop is not None:
            lg[to_id[stop]] = float("-inf")
        if temp <= 0:  # deterministik: her adımda en olası bayt
            nxt = lg.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(lg / temp, dim=-1), 1)
        if stop is not None and vocab[int(nxt)] == stop:
            break
        out.append(vocab[int(nxt)])
        seen = seen or vocab[int(nxt)] not in (9, 10, 13, 32)
        x = torch.cat([x, nxt[None]], dim=1)
    return bytes(out)


def generate(model, vocab, prompt, n, temp, device):
    """vocab: id -> bayt değeri listesi. Boş prompt satır başından başlar. Prompt + devamı döner (EOF'ta durur)."""
    p = prompt.encode("utf-8")
    return (p + sample_bytes(model, vocab, p, n, temp, device)).decode("utf-8", errors="replace")


def fill_bytes(model, vocab, prefix, suffix, n, temp, device):
    """FIM: önek ve sonek arasındaki boşluğu doldurur (EOF'a kadar). Bağlam sığmazsa önek soldan, sonek sağdan kırpılır."""
    n = min(n, max(1, model.T // 2))
    budget = model.T - 3 - n
    pre_b = budget * 2 // 3 if len(prefix) + len(suffix) > budget else len(prefix)
    pre = prefix[-pre_b:] if pre_b > 0 else b""
    suf = suffix[:max(0, budget - len(pre))]
    ctx = bytes([PRE_B]) + pre + bytes([SUF_B]) + suf + bytes([MID_B])
    return sample_bytes(model, vocab, ctx, n, temp, device, no_empty=True)


def trim_balanced(mid, own_line):
    """Doldurulan kısım açtığından fazla '}' kapatmaya başladıysa (yani sonekin işini yapıyorsa) orada kes.
    own_line: kesim bir önceki satır sonuna çekilir (yarım satır kalmasın)."""
    depth = 0
    for i, c in enumerate(mid):
        depth += (c == 0x7B) - (c == 0x7D)
        if depth < 0:
            mid = mid[:i]
            if own_line:
                mid = mid[:mid.rfind(b"\n") + 1]
            break
    return mid


def fill_text(model, vocab, pre, suf, n, temp, device):
    """<FILL> kullanımı: boşluk kendi satırındaysa (öncesi \n+girinti, sonrası \n) eğitimdeki satır-hizalı biçime çevirir
    (önek satır başında biter, sonek satır başında başlar, girintiyi model yazar). Döner: (önek, orta, sonek)."""
    own = pre.rstrip(b" \t").endswith(b"\n") and suf.startswith(b"\n")
    if own:
        pre, suf = pre.rstrip(b" \t"), suf[1:]
    mid = trim_balanced(fill_bytes(model, vocab, pre, suf, n, temp, device), own)
    if own:
        mid = (mid.rstrip(b" \t\n") + b"\n") if mid.strip() else b""
    return pre, mid, suf


@torch.no_grad()
def complete_line(model, vocab, ctx, max_new, device):
    """Greedy: ctx baytlarından sonra gelen satırı (satır sonuna kadar) üretir, bayt olarak döner."""
    model.eval()
    to_id = {b: i for i, b in enumerate(vocab)}
    ids = [to_id[b] for b in ctx if b in to_id] or [to_id[10]]
    x = torch.tensor([ids], device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -model.T:])
        nxt = int(logits[0, -1].argmax())
        if vocab[nxt] == 10:
            break
        out.append(vocab[nxt])
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return bytes(out)


def eval_lines(model, vocab, val_files, n, device, seed=1):
    """Sonraki-satır isabet testi: doğrulama dosyalarında rastgele bir satırın öncesini ver,
    modele (greedy) o satırı yazdır, gerçeğiyle karşılaştır. Önemsiz satırlar ({, }, boş) sayılmaz.
    Karşılaştırma ölçütü: 'önceki satırı aynen tekrarla'."""
    rng = np.random.default_rng(seed)
    T = model.T
    done = hit = hit_copy = tries = 0
    frac_sum = 0.0
    shown = []
    while done < n and tries < 50 * n:
        tries += 1
        data = val_files[int(rng.integers(len(val_files)))][1]
        lines = data.split(b"\n")
        if len(lines) < 8:
            continue
        i = int(rng.integers(4, len(lines)))
        target = lines[i].rstrip()
        if len(target.strip()) < 6:
            continue
        ctx = (b"\n".join(lines[:i]) + b"\n")[-(T - 1):]
        k = ctx.find(b"\n")
        if 0 <= k < len(ctx) - 1:  # baştaki yarım satırı at
            ctx = ctx[k + 1:]
        gen = complete_line(model, vocab, ctx, 200, device).rstrip()
        m = 0
        for p, q in zip(gen, target):
            if p != q:
                break
            m += 1
        done += 1
        hit += int(gen == target)
        hit_copy += int(lines[i - 1].rstrip() == target)
        frac_sum += m / len(target)
        if len(shown) < 6:
            shown.append((b"\n".join(ctx.split(b"\n")[-3:-1]), target, gen))
    if done == 0:
        raise SystemExit("No usable lines found in the validation files.")
    print(f"validation files: {len(val_files)}, lines tried: {done}")
    print(f"line exactly right:           %{100 * hit / done:.1f}")
    print(f"copy-previous-line baseline:  %{100 * hit_copy / done:.1f}")
    print(f"correct prefix of the line:    %{100 * frac_sum / done:.1f} (mean)")
    print("\nexamples (last 2 context lines, real line, model output):")
    for c, t, g in shown:
        print("-" * 60)
        print(c.decode("utf-8", "replace"))
        print("  real :", t.decode("utf-8", "replace").strip())
        print("  model :", g.decode("utf-8", "replace").strip())
    return {"n": done, "exact": hit / done, "copy": hit_copy / done, "prefix": frac_sum / done}


def eval_fim(model, vocab, val_files, n, device, seed=1):
    """FIM isabet testi: doğrulama dosyasında rastgele bir satırı boşluk yap (önceki kod = önek, sonraki satırlar = sonek),
    modele (greedy) doldurt, gerçeğiyle karşılaştır. Aynı satırlar eval_lines ile (sadece önek) da ölçülebilir."""
    rng = np.random.default_rng(seed)
    T = model.T
    done = hit = tries = 0
    frac_sum = 0.0
    shown = []
    while done < n and tries < 50 * n:
        tries += 1
        data = val_files[int(rng.integers(len(val_files)))][1]
        lines = data.split(b"\n")
        if len(lines) < 8:
            continue
        i = int(rng.integers(4, len(lines) - 3))
        target = lines[i].rstrip()
        if len(target.strip()) < 6:
            continue
        prefix = b"\n".join(lines[:i]) + b"\n"
        suffix = b"\n".join(lines[i + 1:])
        gen = fill_bytes(model, vocab, prefix[-T:], suffix[:T], 200, 0.0, device).split(b"\n")[0].rstrip()
        m = 0
        for p, q in zip(gen, target):
            if p != q:
                break
            m += 1
        done += 1
        hit += int(gen == target)
        frac_sum += m / len(target)
        if len(shown) < 6:
            shown.append((b"\n".join(prefix.split(b"\n")[-3:-1]), target, gen))
    if done == 0:
        raise SystemExit("No usable lines found in the validation files.")
    print(f"validation files: {len(val_files)}, lines tried: {done}")
    print(f"FIM: line exactly right:           %{100 * hit / done:.1f}")
    print(f"FIM: correct prefix of the line:    %{100 * frac_sum / done:.1f} (mean)")
    print("\nexamples (previous 2 lines, real line, what the model filled in):")
    for c, t, g in shown:
        print("-" * 60)
        print(c.decode("utf-8", "replace"))
        print("  real :", t.decode("utf-8", "replace").strip())
        print("  model :", g.decode("utf-8", "replace").strip())
    return {"n": done, "exact": hit / done, "prefix": frac_sum / done}


OUT_MARK = np.frombuffer(b"// out:\n", dtype=np.uint8)


def out_region_mask(seq):
    """seq: 1-D uint8. True for bytes after an '// out:\\n' marker up to and including the next EOF byte."""
    n, m = len(seq), len(OUT_MARK)
    mask = np.zeros(n, dtype=bool)
    if n < m:
        return mask
    win = np.lib.stride_tricks.sliding_window_view(seq, m)
    starts = np.flatnonzero((win == OUT_MARK).all(axis=1)) + m  # first byte of each output
    eofs = np.flatnonzero(seq == EOF_B)
    for s in starts:
        k = int(np.searchsorted(eofs, s))
        mask[s:(int(eofs[k]) + 1 if k < len(eofs) else n)] = True
    return mask


def make_seq(d, n1, fim_p, rng):
    """d: uint8 dizi. n1 uzunluğunda bir eğitim dizisi döner. fim_p olasılıkla pencere FIM'e çevrilir:
    PRE önek SUF sonek MID orta EOF  (toplam uzunluk yine n1; EOF 'boşluk burada bitti' demek)."""
    i = int(rng.integers(len(d) - n1))
    if rng.random() >= fim_p:
        return d[i:i + n1]
    w = d[i:i + n1 - 4]
    L = len(w)
    r = rng.random()
    a = b = None
    if r < 0.5:  # satır hizalı, 1-6 satırlık boşluk (gerçek kullanım: bir gövde/ifade doldurmak)
        starts = np.flatnonzero(w[:-1] == 10) + 1
        if len(starts) >= 2:
            k, m = int(rng.integers(0, len(starts) - 1)), int(rng.integers(1, 7))
            a = int(starts[k])
            b = int(starts[k + m]) if k + m < len(starts) else L
    elif r < 0.75:  # bayt seviyesinde kısa boşluk (<= 80 bayt)
        a = int(rng.integers(0, L + 1))
        b = a + int(rng.integers(0, min(80, L - a) + 1))
    if a is None:  # bayt seviyesinde rastgele (uzun olabilir)
        a, b = sorted(int(v) for v in rng.integers(0, L + 1, 2))
    return np.concatenate((np.array([PRE_B], np.uint8), w[:a], np.array([SUF_B], np.uint8), w[b:],
                           np.array([MID_B], np.uint8), w[a:b], np.array([EOF_B], np.uint8)))


def resolve_data(a):
    """--data klasörleri + --datasets altındaki her alt klasör (sıralı). Kendi projeler --data'da kalsın: önce onlar okunur."""
    out = list(a.data or [])
    if a.datasets:
        if not os.path.isdir(a.datasets):
            raise SystemExit(f"--datasets folder not found: {a.datasets}")
        out += [os.path.join(a.datasets, n) for n in sorted(os.listdir(a.datasets))
                if os.path.isdir(os.path.join(a.datasets, n)) and not n.startswith(".")]
    return out


def used_data(meta):
    """Checkpoint'in ve üst modellerinin eğitildiği klasörler (normalize edilmiş yollar)."""
    seen = set()
    while meta:
        seen.update(os.path.normcase(os.path.abspath(d)) for d in meta.get("data") or [])
        meta = meta.get("parent")
    return seen


def list_root(root):
    """root/datasets/* ve root/models/*.pt: hangi veri kaynağı hangi modellerde kullanıldı."""
    ds_dir, md_dir = os.path.join(root, "datasets"), os.path.join(root, "models")
    sets = sorted(n for n in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, n)) and not n.startswith(".")) \
        if os.path.isdir(ds_dir) else []
    models = sorted(n for n in os.listdir(md_dir) if n.endswith(".pt")) if os.path.isdir(md_dir) else []
    used, nometa = {}, []
    for m in models:
        ck = torch.load(os.path.join(md_dir, m), map_location="cpu", weights_only=True)
        if not ck.get("meta"):
            nometa.append(m)
        used[m] = used_data(ck.get("meta"))
    print(f"data sources ({ds_dir}):")
    for n in sets:
        key = os.path.normcase(os.path.abspath(os.path.join(ds_dir, n)))
        who = [m for m in models if key in used[m]]
        print(f"  {n:30s} {'trained: ' + ', '.join(who) if who else 'NOT TRAINED'}")
    if not sets:
        print("  (none)")
    print(f"models ({md_dir}): {', '.join(models) or '(none)'}")
    if nometa:
        print(f"note: {', '.join(nometa)} have no record (old checkpoint), training data unknown.")


def cap_groups(files, first_capped, cap, seed=0):
    """group >= first_capped olan her grup için en fazla `cap` bayt: dosyalar sabit tohumla karıştırılıp seçilir
    (alfabetik sıradaki ilk klasörler değil, her yerden örnek gelsin). Sıra ve ölçüm tekrarlanabilir kalır."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, f in enumerate(files):
        by.setdefault(f[2], []).append(i)
    keep = set()
    for g, ids in by.items():
        if g < first_capped or g >= 100:
            keep.update(ids)
            continue
        total = 0
        for i in rng.permutation(ids):
            if total + len(files[i][1]) > cap:
                continue
            total += len(files[i][1])
            keep.add(int(i))
    return [f for i, f in enumerate(files) if i in keep]


def load_files(a):
    """--data/--datasets/--ext/--stdlib/--chars argümanlarından dosya listesini kurar (eğitim ve ölçüm aynısını kullanır)."""
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in a.ext)
    folders = resolve_data(a)
    files = collect_files(folders, exts, SKIP_DIRS)
    if a.per_dir > 0:  # --datasets kaynaklarına bayt sınırı (kendi --data klasörlerin sınırsız)
        files = cap_groups(files, len(a.data or []), a.per_dir * 1_000_000, a.per_dir_seed)
    if a.stdlib or not folders:
        files += collect_files([sysconfig.get_path("stdlib")], (".py",),
                               SKIP_DIRS | STDLIB_SKIP, g0=100)
    total = 0
    for k, f in enumerate(files):  # --chars sınırı: önce kendi klasörlerin, sonra ekstralar
        total += len(f[1])
        if total >= a.chars:
            files = files[:k + 1]
            break
    if not files:
        raise SystemExit("No files found. Check the --data path and the --ext extensions.")
    return files


def print_meta(meta, indent=0):
    """Checkpoint'in 'meta' kaydını (veri, üst model zinciri) okunur biçimde yazar."""
    pad = "  " * indent
    if meta is None:
        print(f"{pad}(no record: old checkpoint)")
        return
    print(f"{pad}date: {meta.get('date')}  steps: {meta.get('steps_done')}/{meta.get('steps')}  "
          f"lr: {meta.get('lr')}  val loss: {meta.get('val_loss')}")
    print(f"{pad}data: {meta.get('n_files')} files, {meta.get('train_bytes'):,} training bytes")
    for d in meta.get("data") or []:
        print(f"{pad}  - {d}")
    if meta.get("stdlib"):
        print(f"{pad}  - (Python stdlib)")
    if meta.get("init_from"):
        print(f"{pad}continued from: {meta['init_from']}")
        print_meta(meta.get("parent"), indent + 1)


def build(cfg, device):
    return GPT(cfg["V"], cfg["C"], cfg["L"], cfg["nh"], cfg["T"], cfg["p"]).to(device)


def get_prompt(a):
    """--prompt-file varsa metni dosyadan okur. \r\n -> \n (model sadece \n görerek eğitildi)."""
    text = a.prompt
    if a.prompt_file:
        with open(a.prompt_file, encoding="utf-8-sig") as fh:  # metin modu \r\n'yi \n yapar
            text = fh.read()
    return text.replace("\r\n", "\n")


# ---------------------------------------------------------------- ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None,
                    help="training data folders (subfolders included), several allowed")
    ap.add_argument("--datasets", default=None,
                    help="add every subfolder as a separate data source (read AFTER --data)")
    ap.add_argument("--per-dir", type=int, default=0, metavar="MB",
                    help="max MB from EACH source under --datasets (0=unlimited); your own --data folders are unlimited")
    ap.add_argument("--per-dir-seed", type=int, default=0,
                    help="which file subset --per-dir picks; change it between consecutive runs to see different slices "
                         "(give the same value as in training for --eval-lines)")
    ap.add_argument("--list", default=None, metavar="ROOT",
                    help="scan ROOT/datasets and ROOT/models: which data was used in which model")
    ap.add_argument("--ext", nargs="+", default=[".cs"],
                    help="extensions to read in the --data folders (default: .cs)")
    ap.add_argument("--stdlib", action="store_true",
                    help="also add the Python standard library (used anyway when --data is not given)")
    ap.add_argument("--chars", type=int, default=50_000_000, help="max bytes to read")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--embd", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--load", default=None)
    ap.add_argument("--init", default=None,
                    help="continue training from this checkpoint's weights (architecture is taken from it, "
                         "--embd/--layers/--heads/--block are ignored); must differ from --out")
    ap.add_argument("--info", default=None, help="print where/with which data a checkpoint was produced")
    ap.add_argument("--prompt", default="", help="start text for generation (empty: from the start of a line)")
    ap.add_argument("--prompt-file", default=None,
                    help="read the start text from a file (multi-line partial code; overrides --prompt; put <FILL> for fill-in-the-middle)")
    ap.add_argument("--temp", type=float, default=0.8,
                    help="sampling temperature; 0 = most likely byte at every step (deterministic)")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--eval-lines", type=int, default=0,
                    help="with --load: next-line accuracy test on this many validation lines "
                         "(use the same --data/--datasets/--per-dir/--ext/--stdlib/--chars as in training)")
    ap.add_argument("--fim", type=float, default=0.5,
                    help="fraction of training windows converted to fill-in-the-middle (0 = off); "
                         "an EOF byte is always appended to every file")
    ap.add_argument("--eval-fim", type=int, default=0,
                    help="with --load: FIM accuracy test on this many validation lines "
                         "(like --eval-lines, same data arguments as in training)")
    ap.add_argument("--out-only", action="store_true",
                    help="boilerplate data (gen_boilerplate.py): windows start at a file start and the loss counts only "
                         "the bytes after '// out:' (up to and including EOF); the printed val loss is then output-only too")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation: effective batch = bs * accum (steps then count optimizer updates)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # Windows konsolunda Türkçe karakter çökmesin
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    prompt = get_prompt(a)

    if a.list:
        list_root(a.list)
        return

    if a.info:
        ck = torch.load(a.info, map_location="cpu", weights_only=True)
        c = ck["cfg"]
        print(f"{a.info}: embd={c['C']} layers={c['L']} heads={c['nh']} block={c['T']}")
        print_meta(ck.get("meta"))
        return

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.load:
        ck = torch.load(a.load, map_location=device, weights_only=True)
        model = build(ck["cfg"], device)
        model.load_state_dict(ck["model"])
        # eski (karakter seviyeli) checkpoint'lerde "vocab" yok, "chars" var
        vocab = ck["vocab"] if "vocab" in ck else [ord(c) for c in ck["chars"]]
        if a.eval_lines > 0 or a.eval_fim > 0:
            files = load_files(a)
            val_files = [files[i] for i in sorted(pick_val_ids(files))]
            if not val_files:
                raise SystemExit("No validation files (too few files).")
            print("Note: validation files are recomputed from the SAME --data/--datasets/--per-dir/--ext/--stdlib/--chars as in training; "
                  "if they differ, the result is misleading.")
            if a.eval_lines > 0:
                eval_lines(model, vocab, val_files, a.eval_lines, device, a.seed)
            if a.eval_fim > 0:
                eval_fim(model, vocab, val_files, a.eval_fim, device, a.seed)
            return
        if "<FILL>" in prompt:  # FIM: prompt dosyasında boşluğun yerine <FILL> yaz
            pre, suf = (x.encode("utf-8") for x in prompt.split("<FILL>", 1))
            pre, mid, suf = fill_text(model, vocab, pre, suf, a.tokens, a.temp, device)
            print((pre + mid + suf).decode("utf-8", errors="replace"))
            print("\n--- the model filled in only this part:\n" + mid.decode("utf-8", errors="replace"))
            return
        print(generate(model, vocab, prompt, a.tokens, a.temp, device))
        return

    init_ck = None
    if a.init:
        if os.path.abspath(a.init) == os.path.abspath(a.out):
            raise SystemExit("--out must differ from --init (do not overwrite the old model). Give another --out.")
        init_ck = torch.load(a.init, map_location=device, weights_only=True)
        a.block = init_ck["cfg"]["T"]  # konum gömmeleri bu uzunluğa göre eğitildi
        print(f"initial weights: {a.init} (architecture taken from it)")

    if device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found, using CPU. The default settings are very slow on CPU;"
              " see the usage notes at the top of the file for small settings.")
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())

    # --- veri: klasörler -> dosyalar -> UTF-8 baytları -> eğitim/doğrulama
    files = load_files(a)
    train_b, val_b = split_files(files)
    if min(len(train_b), len(val_b)) <= a.block + 1:
        raise SystemExit("Training or validation data is too short. Give more files or reduce --block.")
    n_own = sum(1 for f in files if f[2] < 100)
    print(f"data: {len(files)} files ({n_own} own, {len(files) - n_own} stdlib), "
          f"train {len(train_b):,} bytes, validation {len(val_b):,} bytes")

    train = torch.from_numpy(np.frombuffer(train_b, dtype=np.uint8).copy())
    val = torch.from_numpy(np.frombuffer(val_b, dtype=np.uint8).copy())
    vocab = list(range(256))  # her bayt bir token

    if init_ck:
        cfg = dict(init_ck["cfg"], p=a.dropout)
        model = build(cfg, device)
        model.load_state_dict(init_ck["model"])
    else:
        cfg = dict(V=len(vocab), C=a.embd, L=a.layers, nh=a.heads, T=a.block, p=a.dropout)
        model = build(cfg, device)
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "data": [os.path.abspath(d) for d in resolve_data(a)],
            "ext": list(a.ext), "stdlib": bool(a.stdlib or not resolve_data(a)), "chars": a.chars,
            "n_files": len(files), "train_bytes": len(train_b), "steps": a.steps, "steps_done": 0,
            "lr": a.lr, "bs": a.bs, "fim": a.fim, "val_loss": None,
            "init_from": os.path.basename(a.init) if a.init else None,
            "parent": (init_ck.get("meta") if init_ck else None)}
    nparams = sum(p.numel() for p in model.parameters())
    print(f"parameters={nparams:,}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.99))

    rng = np.random.default_rng(a.seed)
    train_np, val_np = train.numpy(), val.numpy()

    def starts_of(arr):  # window starts: the byte right after each EOF (= the first byte of a file)
        e = np.flatnonzero(arr == EOF_B) + 1
        return e[e + a.block + 1 <= len(arr)]

    out_starts = {True: starts_of(train_np), False: starts_of(val_np)} if a.out_only else None
    if a.out_only and min(len(v) for v in out_starts.values()) == 0:
        raise SystemExit("--out-only: no usable file starts found (files must be shorter than --block).")

    def get_batch(d, fim_p=0.0):
        is_train = d is train
        arr = train_np if is_train else val_np
        if a.out_only:
            st = out_starts[is_train]
            seq = np.stack([arr[i:i + a.block + 1] for i in st[rng.integers(len(st), size=a.bs)]])
            keep = torch.from_numpy(np.stack([out_region_mask(s)[1:] for s in seq]))
            t = torch.from_numpy(seq).long()
            y = t[:, 1:].clone()
            y[~keep] = -100  # F.cross_entropy ignores -100 targets
            return t[:, :-1].to(device, non_blocking=True), y.to(device, non_blocking=True)
        seq = np.stack([make_seq(arr, a.block + 1, fim_p, rng) for _ in range(a.bs)])
        t = torch.from_numpy(seq).long()
        return t[:, :-1].to(device, non_blocking=True), t[:, 1:].to(device, non_blocking=True)

    @torch.no_grad()
    def eval_loss(d, n=20):
        model.eval()
        tot = 0.0
        for _ in range(n):
            x, y = get_batch(d)
            with autocast:
                _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / n

    def save(done):
        meta["steps_done"] = done
        tmp = a.out + ".tmp"  # write to a temp file first: a crash or a concurrent read never sees a half-written .pt
        torch.save({"model": model.state_dict(), "cfg": cfg, "vocab": vocab, "meta": meta}, tmp)
        os.replace(tmp, a.out)

    print(f"initial loss ~ {math.log(len(vocab)):.3f} (random guess)")
    model.train()
    t0 = time.time()
    run = torch.zeros((), device=device)
    best, best_path = float("inf"), os.path.splitext(a.out)[0] + "_best.pt"
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train, a.fim)
            with autocast:
                _, l = model(x, y)
            (l / a.accum).backward()
            loss = loss + l.detach() / a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.detach()
        if step % 100 == 0 or step == 1:
            n = 1 if step == 1 else 100
            el = time.time() - t0
            msg = f"step {step:5d}/{a.steps}  loss {run.item() / n:.3f}  lr {lr:.5f}  {el:6.0f}s"
            if step >= 100:
                msg += f"  remaining ~{el / step * (a.steps - step) / 60:.1f} min"
            run.zero_()
            if step % a.eval_every == 0:
                v = eval_loss(val)
                msg += f"  val {v:.3f}"
                save(step)
                if v < best:  # keep the best weights too: a later divergence must not destroy them
                    best = v
                    shutil.copyfile(a.out, best_path)
                    msg += "  (best)"
            print(msg, flush=True)

    final = eval_loss(val, 50)
    meta["val_loss"] = round(final, 4)
    print(f"\nfinal val loss: {final:.3f}")
    save(a.steps)
    print(f"model saved: {a.out}\n")
    for temp in (0.6, 0.9):
        print(f"--- sample (temperature {temp}) " + "-" * 40)
        print(generate(model, vocab, prompt, a.tokens, temp, device))
        print()


if __name__ == "__main__":
    main()