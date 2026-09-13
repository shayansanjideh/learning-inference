import json
import os
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(root/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=json.loads((root/'results/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#394354','text.color':'#192334','axes.edgecolor':'#bcc5d4'})
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,rows,key,title,xlabel in [
 (axs[0],sorted([r for r in s if r['input_tokens']==128],key=lambda r:r['output_tokens']),'output_tokens','Longer output, fixed 128-token prompt','Generated tokens'),
 (axs[1],sorted([r for r in s if r['output_tokens']==32],key=lambda r:r['input_tokens']),'input_tokens','Longer prompt, fixed 32-token output','Prompt tokens')]:
 x=[r[key] for r in rows]; y=[r['median_latency_s'] for r in rows]
 ax.errorbar(x,y,yerr=[[r['median_latency_s']-r['min_latency_s'] for r in rows],[r['max_latency_s']-r['median_latency_s'] for r in rows]],color='#244dce',marker='o',capsize=4,lw=2)
 ax.set(title=title,xlabel=xlabel,ylabel='Generation wall time (seconds)',ylim=(0,None));ax.set_xticks(x);ax.grid(axis='y',alpha=.18)
 for xx,yy in zip(x,y):ax.annotate(f'{yy:.2f}s',(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
fig.suptitle('Lesson 01 · CPU inference baseline',fontsize=18,fontweight='bold')
fig.supxlabel('SmolLM2-135M · FP32 · batch 1 · 2 CPU threads · median of 5 runs; whiskers show min–max',fontsize=10)
fig.savefig(root/'results/baseline.png',dpi=180)
fig.savefig(root/'results/baseline.svg')

# Publication figures: both summaries and individual observations stay visible.
import csv
raw=list(csv.DictReader((root/'results/raw.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
rows=sorted([r for r in s if r['input_tokens']==128],key=lambda r:r['output_tokens'])
for ax,metric,label,color in [(axs[0],'median_latency_s','Generation latency (seconds)','#244dce'),(axs[1],'median_output_tokens_per_s','Output tokens per second','#227c76')]:
 x=[r['output_tokens'] for r in rows];y=[r[metric] for r in rows]
 vals=[[float(p['latency_s'] if metric=='median_latency_s' else p['output_tokens_per_s']) for p in raw if int(p['input_tokens'])==128 and int(p['output_tokens'])==r['output_tokens']] for r in rows]
 ax.errorbar(x,y,yerr=[[yy-min(v) for yy,v in zip(y,vals)],[max(v)-yy for yy,v in zip(y,vals)]],marker='o',capsize=4,color=color,lw=2)
 ax.set(xlabel='Output tokens',ylabel=label,ylim=(0,None));ax.set_xticks(x);ax.grid(axis='y',alpha=.15)
 for xx,yy in zip(x,y):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(4,12),textcoords='offset points',fontsize=11,color=color)
axs[0].set_title('More output → more total time',loc='left',pad=18,fontsize=13)
axs[1].set_title('Initial cost spread across more tokens',loc='left',pad=18,fontsize=13)
fig.supxlabel('128-token prompt · CPU · FP32 · batch 1 · 5 runs per condition · median and min–max',fontsize=10)
for ext in ['png','svg']:fig.savefig(root/f'results/output-scaling.{ext}',dpi=180)
fig,ax=plt.subplots(figsize=(10,5.333),layout='constrained')
rows=sorted([r for r in s if r['output_tokens']==32],key=lambda r:r['input_tokens'])
for i,r in enumerate(rows):
 vals=[float(p['latency_s']) for p in raw if int(p['output_tokens'])==32 and int(p['input_tokens'])==r['input_tokens']]
 offsets=[-.10,-.05,0,.05,.10]
 ax.scatter([i+v for v in offsets],vals,s=48,color='#8dabea',edgecolors='white',linewidths=.8,zorder=3,label='Individual run' if i==0 else None)
 ax.plot([i-.18,i+.18],[r['median_latency_s']]*2,color='#244dce',lw=2,zorder=4)
 ax.scatter([i],[r['median_latency_s']],s=85,marker='D',color='#244dce',zorder=5,label='Median' if i==0 else None)
 ax.annotate(f"{r['median_latency_s']:.2f}s",(i,r['median_latency_s']),xytext=(17,-20),textcoords='offset points',fontsize=12,color='#244dce')
ax.annotate('10.16s · slowest observed run',(2,10.1622054),xytext=(-200,-5),textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#9aa9c0'},fontsize=11)
ax.set(xticks=[0,1,2],xticklabels=['64','128','256'],xlabel='Prompt tokens',ylabel='Generation latency (seconds)',ylim=(0,11.6),xlim=(-.5,2.6))
ax.grid(axis='y',alpha=.15);ax.legend(loc='upper left',frameon=False)
fig.supxlabel('32 output tokens · CPU · FP32 · batch 1 · all 15 measurements shown',fontsize=10)
for ext in ['png','svg']:fig.savefig(root/f'results/input-scaling.{ext}',dpi=180)

# Narrow-screen figures preserve readable labels in the article.
rows=sorted([r for r in s if r['input_tokens']==128],key=lambda r:r['output_tokens'])
fig,axs=plt.subplots(2,1,figsize=(4.6,8),layout='constrained')
for ax,metric,label,color in [(axs[0],'median_latency_s','Latency (seconds)','#244dce'),(axs[1],'median_output_tokens_per_s','Output tokens/s','#227c76')]:
 x=[r['output_tokens'] for r in rows];y=[r[metric] for r in rows]
 vals=[[float(p['latency_s'] if metric=='median_latency_s' else p['output_tokens_per_s']) for p in raw if int(p['input_tokens'])==128 and int(p['output_tokens'])==r['output_tokens']] for r in rows]
 ax.errorbar(x,y,yerr=[[yy-min(v) for yy,v in zip(y,vals)],[max(v)-yy for yy,v in zip(y,vals)]],marker='o',capsize=4,color=color,lw=2)
 ax.set(xlabel='Output tokens',ylabel=label,ylim=(0,None),xlim=(3,73));ax.set_xticks(x);ax.grid(axis='y',alpha=.15)
 for xx,yy in zip(x,y):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,9),textcoords='offset points',fontsize=10,color=color)
fig.savefig(root/'results/output-scaling-mobile.png',dpi=180)
fig,ax=plt.subplots(figsize=(4.6,4.8),layout='constrained')
rows=sorted([r for r in s if r['output_tokens']==32],key=lambda r:r['input_tokens'])
for i,r in enumerate(rows):
 vals=[float(p['latency_s']) for p in raw if int(p['output_tokens'])==32 and int(p['input_tokens'])==r['input_tokens']]
 ax.scatter([i+v for v in [-.10,-.05,0,.05,.10]],vals,s=45,color='#8dabea',edgecolors='white',zorder=3,label='Run' if i==0 else None)
 ax.scatter([i],[r['median_latency_s']],s=75,marker='D',color='#244dce',zorder=4,label='Median' if i==0 else None)
 ax.annotate(f"{r['median_latency_s']:.2f}s",(i,r['median_latency_s']),xytext=(0,-23),textcoords='offset points',ha='center',fontsize=11)
ax.set(xticks=[0,1,2],xticklabels=['64','128','256'],xlabel='Prompt tokens',ylabel='Latency (seconds)',ylim=(0,11.5),xlim=(-.45,2.45));ax.grid(axis='y',alpha=.15);ax.legend(loc='upper left',frameon=False,fontsize=10)
fig.savefig(root/'results/input-scaling-mobile.png',dpi=180)
