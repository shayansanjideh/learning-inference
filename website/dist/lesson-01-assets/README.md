# Lesson 01: run a model, then establish a baseline

This lesson runs SmolLM2-135M on the VM’s two CPU cores. You’ll operate a small inference server, understand prefill and decode, and reproduce two controlled sequence-length benchmarks.

## 1. Predict before reading the results (10 minutes)

- With a 128-token prompt, will generating 64 tokens take exactly four times as long as generating 16?
- With 32 output tokens, what happens when the prompt grows from 64 to 256 tokens?
- Is `output tokens / total generation time` the same as decode throughput? Why not?

Sketch your predictions. Then open `results/baseline.png` and `LESSON.md`.

## 2. Run the server (15 minutes)

From this lesson directory, use the environment already installed on this VM:

```bash
.venv/bin/python lab.py serve
```

In a second terminal:

```bash
.venv/bin/python client.py
```

The server binds to **127.0.0.1:8001** and exposes `GET /health` and `POST /generate`. It loads the model once, accepts one request at a time, and uses Hugging Face `.generate()`. This teaching server handles one request at a time. Stop it with Ctrl-C before benchmarking so two copies do not compete for CPU and memory.

Try another prompt:

```bash
curl http://127.0.0.1:8001/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_new_tokens":16}'
```

Outputs are forced to the requested length for controlled timing. This base model continues the supplied text; the fixed output limit controls where it stops.

## 3. Read the smallest useful code path (20 minutes)

Read `load_model`, `generate`, then `serve` in `lab.py`. Find:

1. Where weights are loaded once.
2. Where input text becomes token IDs.
3. Where gradient tracking is disabled.
4. Where greedy decoding and caching are enabled.
5. The exact boundaries of the timer.

Prefill processes the prompt and produces logits used to select the first output token. Decode uses cached state to produce subsequent tokens. This lesson times both together through `.generate()`. Separating those timings is the next lesson.

## 4. Reproduce the experiment (15–30 minutes)

```bash
.venv/bin/python lab.py benchmark
.venv/bin/python plot.py
```

The script performs two warmups per shape, then five measured repetitions of six distinct conditions in a shuffled order. It writes exact input token IDs, raw timing rows, summaries, samples, and an environment manifest to `results/`. Re-running replaces those result files, so copy the directory first if you want to keep this run.

- Output sweep: 8, 16, 32, 64 generated tokens; prompt fixed at 128 tokens.
- Prompt sweep: 64, 128, 256 input tokens; output fixed at 32 tokens.
- Batch 1, FP32, two CPU threads, eager attention, KV cache enabled.
- Model loading, tokenization, detokenization, and HTTP overhead are outside the benchmark timer.
- Repeated synthetic prefixes control token length; this keeps input length controlled.
- Total generated tokens divided by total generation time is **end-to-end generation throughput**, not isolated decode throughput.
- The cache assertion checks actual tensor bytes against the conventional KV formula.

## 5. Make one change yourself (20 minutes)

Add a 512-token prompt condition, predict its effect, and rerun. Keep everything else fixed. Explain the measured change without claiming you have isolated prefill or decode. Then inspect `results/samples.json`: would this text be useful to a user? Why can a performance baseline still be useful when model quality is limited?

## Install on a fresh machine

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python lab.py benchmark
.venv/bin/python plot.py
```

`model-lock.json` pins the exact Hugging Face revision. The downloader reuses that revision on subsequent machines and refreshes the local snapshot path. `environment-freeze.txt` records all dependency versions used for this run; `requirements.txt` specifies the direct dependencies.

## Reading tied to the build

- [Scaling book: inference](https://jax-ml.github.io/scaling-book/inference/): read the opening prefill/generation explanation, then identify both phases inside this workload.
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): compare its loading example with `load_model` and explain why this VM uses FP32 CPU execution.
- [Transformers generation](https://huggingface.co/docs/transformers/v4.48.3/en/main_classes/text_generation): explain `do_sample`, `use_cache`, `min_new_tokens`, and `max_new_tokens` in the experiment.

Done means: you can start the server, reproduce the data, identify what the timer includes, and explain one limitation. Use the exercises to connect the code with the plotted results.
