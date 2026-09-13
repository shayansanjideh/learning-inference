import json
import os
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi, snapshot_download
lock=root/'model-lock.json'
revision=json.loads(lock.read_text())['revision'] if lock.exists() else HfApi().model_info('HuggingFaceTB/SmolLM2-135M').sha
path=snapshot_download('HuggingFaceTB/SmolLM2-135M',revision=revision,cache_dir=str(root/'.cache/huggingface/hub'),allow_patterns=['*.json','*.safetensors','*.txt','*.model'],max_workers=2)
lock.write_text(json.dumps({'model':'HuggingFaceTB/SmolLM2-135M','revision':revision,'path':path},indent=2))
print('Pinned model:',revision)
