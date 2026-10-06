# konjin-ai

> *Konjin* = Turkish "konuşan cin" (a talking jinn), also a nod to Japanese *jinkō chinō* (artificial intelligence). A personal learning project; not affiliated with any company or product of a similar name.

A byte-level GPT-style language model written from scratch in PyTorch, trained on a single laptop GPU (RTX 4060 Laptop, 8 GB).
Two small experiments, built mostly to understand how these models are trained and where they break:

1. **C# boilerplate generator** - a 26M-parameter model that writes `constructor`, `ToString`, `Equals + GetHashCode` and `Clone` for a given class. **98.3% exact match (118/120)** on held-out classes with identifiers it never saw in training.
2. **Small Turkish chat model** - a 38M-parameter model pre-trained on Turkish Wikipedia, then fine-tuned on Turkish chat/instruction data. It holds a basic conversation. It is also frequently wrong and sometimes nonsensical; it is a toy.

No tokenizer library, no training framework: one file (`src/model/minigpt.py`) holds the model, data pipeline, training loop and sampling.

## Model

Decoder-only transformer, pre-LayerNorm, causal scaled-dot-product attention, tied input/output embeddings, byte vocabulary (256 + a few special bytes that cannot occur in UTF-8). AdamW, cosine LR schedule with warmup, bf16 autocast, gradient clipping, optional gradient accumulation.

| Model | Params | Layers x width | Context | Data |
|---|---|---|---|---|
| Boilerplate | 26M | 8 x 512 | 1024 bytes | 150k generated examples |
| Turkish chat | 38M | 12 x 512 | 1024 bytes | 760 MB Turkish Wikipedia, then ~60k chat examples |

Loss on the answer only: for task data (`// out:` marker) and chat data the loss is masked so the model is trained only on the output part (`--out-only`).

## Results

**Boilerplate (held-out exact match):** 118/120 = 98.3%. The test classes use identifiers drawn from a word pool that is disjoint from training, so the model has to copy names rather than recall them.
Not verified: the generated C# was compared to the reference output as text; it was not compiled.

**Turkish model:** validation loss 0.777 nats/byte after pre-training; 0.605 on replies after chat fine-tuning. Example (temperature 0.6):

```
Sen: merhaba
Model: Merhaba! Bugün size nasıl yardımcı olabilirim?
Sen: günde kaç kere yemek yenir
Model: Günde 20 kere yemek yapmak çok önemlidir.
```

Grammar and style are plausible; facts and reasoning are not. That is what ~40M byte-level parameters give you.

## What went wrong (and what it taught me)

- **Shortcut learning.** The first boilerplate generator used ~170 identifier words. The model memorised the vocabulary and failed on new names. Fix: 5000 words mined from real code plus 10% random pseudo-words, evaluated on a separate held-out word pool.
- **Mean loss hides progress.** With random identifiers in the input, the average loss stayed flat while the model was learning the task. Fix: compute loss on the output region only.
- **A divergence destroyed the best checkpoint.** The first Turkish run reached validation loss 0.90, then blew up around step 70k (lr 5e-4, tiny batch) and never recovered; the run only kept the latest weights. Fix: lower LR, gradient accumulation (effective batch x4), and a `_best.pt` checkpoint saved whenever validation improves. The rerun finished at 0.777.
- **Atomic checkpoints.** A crash during `torch.save` left a corrupt file. Checkpoints are now written to a temp file and renamed.
- **Spilling past VRAM.** With the GPU memory full, Windows can spill CUDA memory into system RAM and training crawls. Turn off the sysmem fallback in the NVIDIA control panel to get an out-of-memory error instead of a stall.

## Layout

```
src/model/    minigpt.py (train / sample), tr_chat.py (chat REPL)
src/datagen/  gen_boilerplate.py (rule-based task data), prep_tr.py (Turkish data prep)
src/eval/     bp_check.py (exact-match scoring), bp_demo.py (interactive demo), bp_eval.py
data/         training/, tests/, prompts/, raw/   (not in the repo)
models/       checkpoints                          (not in the repo)
```

## Usage

Boilerplate data and model (adjust paths and hyper-parameters to your setup):

```
python src/datagen/gen_boilerplate.py --out data/bp --train 150000 --test 300 --words-from <folder with .cs files> --pseudo 0.1
python src/model/minigpt.py --out models/bp.pt --data data/bp/train --ext .cs --embd 512 --layers 8 --heads 8 --block 1024 --bs 8 --steps 20000 --out-only --fim 0
python src/eval/bp_demo.py --load models/bp.pt
```

Turkish model:

```
pip install pyarrow
python src/datagen/prep_tr.py wiki --src <wikipedia parquet folder> --out data/training/tr/wiki --max-mb 800
python src/datagen/prep_tr.py chat --src <chat dataset folder> --out data/training/tr/chat/instr

python src/model/minigpt.py --out models/tr1.pt --data data/training/tr/wiki --ext .txt --embd 512 --layers 12 --heads 8 --block 1024 --bs 8 --accum 4 --lr 3e-4 --dropout 0.0 --steps 36000 --eval-every 2000 --fim 0
python src/model/minigpt.py --init models/tr1_best.pt --out models/tr2.pt --data data/training/tr/chat/instr --ext .txt --out-only --lr 1e-4 --dropout 0.1 --steps 15000 --fim 0
python src/model/tr_chat.py --load models/tr2.pt --temp 0.6
```

Training time on the RTX 4060 Laptop: the 26M model does ~0.1 s/step (20k steps in about 33 minutes); the 38M model ~0.15 s/step (about 6 hours for the Wikipedia stage).

## Limitations

- Toy scale. The Turkish model is a demonstration of the pipeline, not a useful assistant.
- Boilerplate outputs were checked against a rule-based reference, not compiled.
- Byte-level modelling is simple but data- and compute-hungry: a byte model sees about a quarter of the text a token model sees for the same context length.

## Credits

Turkish Wikipedia and Turkish chat/instruction datasets from Hugging Face (check each dataset's licence before reusing the trained weights).
