"""Run after starting lab.py serve; validates a real generation and bad input."""
import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
base='http://127.0.0.1:8001'
with urlopen(base+'/health',timeout=10) as response: assert json.load(response)['status']=='ready'
request=Request(base+'/generate',data=json.dumps({'prompt':'Gravity is','max_new_tokens':16}).encode(),headers={'Content-Type':'application/json'})
with urlopen(request,timeout=120) as response: result=json.load(response)
assert result['output_tokens']==16 and result['generation_s']>0
Path('results/server-smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
request=Request(base+'/generate',data=b'{"prompt":"hello","max_new_tokens":1000}',headers={'Content-Type':'application/json'})
try: urlopen(request,timeout=10);raise AssertionError('Oversized output must be rejected')
except HTTPError as e: assert e.code==400
print('PASS: health, real generation, exact length, invalid request rejection')
