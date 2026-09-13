"""Build lesson 01 from the maintained Markdown article and measured artifacts.
Run with a Python environment containing requirements-build.txt.
"""
from pathlib import Path
import html,re,shutil,zipfile,tempfile
import markdown
site=Path(__file__).resolve().parents[1]
lab=site.parent/'lessons/01-baseline'
assets=site/'dist/lesson-01-assets'
# A source checkout can rebuild from the bundled experiment as well.
if not lab.exists():
    temporary=tempfile.TemporaryDirectory()
    with zipfile.ZipFile(assets/'lesson-01.zip') as bundle:
        bundle.extractall(temporary.name)
    lab=Path(temporary.name)/'lesson-01'
source=site/'content/lesson-01.md'
article=source.read_text()
md=markdown.Markdown(extensions=['fenced_code','tables','toc'])
body=md.convert(article)
words=len(re.findall(r'\b[\w’-]+\b',re.sub(r'<[^>]*>',' ',article)))
read_minutes=round(words/220)
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Serving and benchmarking a small language model · Learning inference</title><meta name="description" content="An illustrated walkthrough of a minimal LLM server, prefill and decode, and controlled sequence-length benchmarks on a CPU VM."><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="lesson.css"><link rel="stylesheet" href="postcard.css?v=borderless-2"><link rel="icon" type="image/svg+xml" href="favicon.svg"></head><body><a class="skip" href="#lesson">Skip to lesson</a><main class="article-shell"><aside class="article-nav" aria-label="On this page"><span>IN THIS LESSON</span>'''+md.toc+'''<a class="nav-back" href="index.html#journal">← All lessons</a></aside><article id="lesson" class="reading"><div class="article-resources"><a href="lesson-01-assets/lesson-01.zip" download>Code & exercises ↓</a><a href="lesson-01-assets/raw.csv" download>Raw measurements ↓</a><a href="lesson-01-assets/environment.json">Experiment setup</a></div>'''+body.replace('</h1>', '</h1><figure class="lesson-postcard"><img src="postcards/lesson-01-coast.png" width="1536" height="1024" alt="An ink drawing of a coastal path, windswept pines and blue sea."></figure>', 1)+'''<div class="next-lesson"><span>NEXT IN THE SERIES · LESSON 02</span><h3>Implementing the prefill and decode loop</h3><p>A manual generation loop, output-parity checks, and separate timings for each phase.</p><a href="index.html#week-2">View the build brief ↗</a></div></article></main></body></html>'''
(site/'dist/lesson-01.html').write_text(page)
for name in ['LESSON.md','article.md']:(lab/name).write_text(article)
(lab/'report.py').write_text('''"""Refresh LESSON.md from the maintained article.md. Charts come from plot.py."""\nfrom pathlib import Path\np=Path(__file__).resolve().parent\n(p/'LESSON.md').write_text((p/'article.md').read_text())\nprint('Refreshed lesson from article.md')\n''')
assets.mkdir(exist_ok=True)
if (lab/'thread-images').exists():shutil.copytree(lab/'thread-images',assets/'thread-images',dirs_exist_ok=True)
for p in (lab/'results').iterdir():
 if p.is_file():shutil.copy2(p,assets/p.name)
for name in ['LESSON.md','README.md','THREAD-DRAFT.md']:shutil.copy2(lab/name,assets/name)
files=['README.md','LESSON.md','article.md','THREAD-DRAFT.md','lab.py','client.py','download_model.py','plot.py','report.py','requirements.txt','environment-freeze.txt','model-lock.json']
with zipfile.ZipFile(assets/'lesson-01.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(lab/name,'lesson-01/'+name)
 for p in sorted((lab/'thread-images').glob('*.png')):z.write(p,'lesson-01/thread-images/'+p.name)
 for p in (lab/'results').iterdir():
  if p.is_file():z.write(p,'lesson-01/results/'+p.name)
 # Article graphics use the same relative paths in the downloaded Markdown.
 for name in ['input-scaling.png','output-scaling.png','input-scaling-mobile.png','output-scaling-mobile.png']:z.write(lab/'results'/name,'lesson-01/lesson-01-assets/'+name)
print(f'Built article: {words} words including code, labels and captions; {read_minutes} min read.')
