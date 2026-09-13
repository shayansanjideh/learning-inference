"""Lesson 01: explicit, reproducible CPU generation baseline (no custom loop yet)."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
MODEL = 'HuggingFaceTB/SmolLM2-135M'


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads((ROOT / 'model-lock.json').read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['path'], local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager').eval()
    return torch, tokenizer, model, time.perf_counter() - start


def generate(torch, model, tokens, count):
    # Fixed output length: EOS is intentionally suppressed by min_new_tokens.
    # This isolates output-length cost; it is not a natural-stop/chat benchmark.
    with torch.inference_mode():
        return model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                              min_new_tokens=count, max_new_tokens=count,
                              do_sample=False, use_cache=True,
                              pad_token_id=model.config.eos_token_id)


def swap_counters():
    values = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {k: int(values[k]) for k in ['pswpin', 'pswpout']}


def benchmark():
    import psutil
    import transformers
    torch, tokenizer, model, load_s = load_model()
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    corpus = ('Inference engineering studies how language models turn a prompt into tokens. '
              'A useful experiment fixes the hardware, precision, and workload before measuring latency. ') * 60
    ids = tokenizer.encode(corpus, add_special_tokens=False)
    cases = [(64, 32), (128, 32), (256, 32), (128, 8), (128, 16), (128, 64)]
    prompts = {n: torch.tensor([ids[:n]], dtype=torch.long) for n, _ in cases}
    (results / 'prompts.json').write_text(json.dumps({str(n): t[0].tolist() for n,t in prompts.items()}, indent=2))
    print(f'Loaded {MODEL} in {load_s:.3f}s; warming every shape twice', flush=True)
    for n, count in cases:
        for _ in range(2):
            generate(torch, model, prompts[n], count)
    rows = []
    swap_before = swap_counters()
    samples = {}
    start_time = datetime.now(timezone.utc).isoformat()
    # Interleave conditions to reduce bias from time drift on a shared VM.
    schedule = [(repeat,n,k) for repeat in range(5) for n,k in cases]
    random.Random(42).shuffle(schedule)
    for repeat,n,count in schedule:
        start = time.perf_counter()
        out = generate(torch, model, prompts[n], count)
        elapsed = time.perf_counter() - start
        actual = out.shape[1] - n
        assert actual == count, (actual,count)
        digest = hashlib.sha256(out.numpy().tobytes()).hexdigest()
        key = f'{n}:{count}'
        if key in samples:
            assert samples[key]['output_sha256'] == digest, 'Non-deterministic greedy output'
        else:
            samples[key] = {'output_sha256':digest,'completion':tokenizer.decode(out[0,n:],skip_special_tokens=True)}
        rows.append({'repeat':repeat,'input_tokens':n,'output_tokens':actual,
                     'latency_s':elapsed,'output_tokens_per_s':actual/elapsed,
                     'rss_after_mib':psutil.Process().memory_info().rss/2**20})
        print(f'{n:3d} in / {actual:2d} out: {elapsed:.3f}s, {actual/elapsed:.2f} tok/s',flush=True)
    swap_after = swap_counters()
    with (results / 'raw.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for n,count in cases:
        group=[r for r in rows if r['input_tokens']==n and r['output_tokens']==count]
        ts=[r['latency_s'] for r in group]
        summary.append({'input_tokens':n,'output_tokens':count,'repeats':len(ts),
                        'median_latency_s':statistics.median(ts),'min_latency_s':min(ts),
                        'max_latency_s':max(ts),'stdev_latency_s':statistics.stdev(ts),
                        'median_output_tokens_per_s':statistics.median(r['output_tokens_per_s'] for r in group)})
    c=model.config
    params=sum(p.numel() for p in model.parameters())
    head_dim=c.hidden_size//c.num_attention_heads
    kv_per_token=2*c.num_hidden_layers*c.num_key_value_heads*head_dim*4
    with torch.inference_mode():
        cache=model(prompts[64],use_cache=True).past_key_values
    actual_cache=sum(t.numel()*t.element_size() for layer in cache for t in layer)
    assert actual_cache==64*kv_per_token,(actual_cache,kv_per_token)
    manifest={'utc_started':start_time,'model':MODEL,'model_lock':json.loads((ROOT/'model-lock.json').read_text()),
              'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,
              'transformers':transformers.__version__,'device':'cpu','cuda_available':torch.cuda.is_available(),
              'cpu_count':os.cpu_count(),'cpu_description':subprocess.check_output(['lscpu'],text=True),
              'system_ram_gib':psutil.virtual_memory().total/2**30,'torch_threads':torch.get_num_threads(),
              'dtype':'float32','attention_backend':'eager','batch_size':1,'use_cache':True,
              'timing_scope':'model.generate wall time; excludes loading, tokenization and detokenization; includes generation orchestration',
              'fixed_output_length':True,'warmups_per_shape':2,'measured_repeats_per_shape':5,'order_seed':42,
              'model_tokenizer_load_s':load_s,'download_excluded':True,
              'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'process_swap_mib':psutil.Process().memory_full_info().swap/2**20,
              'system_swap_counter_delta':{k:swap_after[k]-swap_before[k] for k in swap_before},
              'parameters_unique':params,'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
              'layers':c.num_hidden_layers,'kv_heads':c.num_key_value_heads,'head_dim':head_dim,
              'kv_bytes_per_cached_token_batch1':kv_per_token,'actual_kv_bytes_64_tokens':actual_cache,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name,data in [('summary',summary),('environment',manifest),('samples',samples)]:
        (results/f'{name}.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'peak_rss_mib':manifest['process_peak_rss_mib'],'swap_delta':manifest['system_swap_counter_delta'],'summary':summary},indent=2))


def serve(port):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    torch, tokenizer, model, _ = load_model()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,payload):
            data=json.dumps(payload).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.reply(200,{'status':'ready','model':MODEL,'device':'cpu'}) if self.path=='/health' else self.reply(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/generate': return self.reply(404,{'error':'not found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384: raise ValueError('Body must be 1–16384 bytes')
                body=json.loads(self.rfile.read(size));prompt=body['prompt'];count=body.get('max_new_tokens',32)
                if not isinstance(prompt,str) or not prompt.strip(): raise ValueError('prompt must be nonempty text')
                if type(count) is not int or not 1<=count<=64: raise ValueError('max_new_tokens must be an integer from 1 to 64')
                tokens=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).input_ids
                if tokens.shape[1]>256: raise ValueError('prompt must contain at most 256 tokens')
            except (ValueError,KeyError,TypeError) as e: return self.reply(400,{'error':str(e)})
            t=time.perf_counter();out=generate(torch,model,tokens,count);elapsed=time.perf_counter()-t
            self.reply(200,{'completion':tokenizer.decode(out[0,tokens.shape[1]:],skip_special_tokens=True),
                            'input_tokens':tokens.shape[1],'output_tokens':count,'generation_s':elapsed,
                            'output_tokens_per_s':count/elapsed,'fixed_length':True})
    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'Lesson server ready: http://127.0.0.1:{port}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['benchmark','serve']);parser.add_argument('--port',type=int,default=8001)
    args=parser.parse_args()
    benchmark() if args.mode=='benchmark' else serve(args.port)
