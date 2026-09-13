# Lesson 1 — Twitter thread draft

Five posts, one image per post. Copy each post body and attach its corresponding PNG.

## 1/5

I want to get better at inference engineering, and I think a good way to learn AI is with AI itself.

I’ve been using Astra to work through a lesson plan with me. I’ll share what I learn every few days as I finish each lesson.

First up: running and benchmarking a small LLM.

**Attach:** [Image 1](thread-images/01-introduction.png)

## 2/5

Lesson 1: a minimal inference server using SmolLM2-135M on my two-core CPU VM.

I worked through:
- how a prompt becomes tokens
- prefill vs. decode
- how input and output lengths affect generation time

The guide includes the code, measurements, and exercises.

**Attach:** [Image 2](thread-images/02-request-path.png)

## 3/5

With a 128-token prompt, generating 16 tokens took 2.16s; generating 64 took 7.53s.

4× the output, about 3.5× the time.

These are medians of five runs after warmup. The timer covers generation, with model loading and tokenization outside it.

**Attach:** [Image 3](thread-images/03-output-scaling.png)

## 4/5

Keeping output at 32 tokens, increasing the prompt from 64 to 256 tokens raised median generation time from 3.43s to 4.75s.

My initial intuition was that parallel prompt processing might keep the time constant. This experiment helped me see why that was too simple.

**Attach:** [Image 4](thread-images/04-input-scaling.png)

## 5/5

Next: writing my own prefill + decode loop and measuring the two phases separately.

The goal is to learn the ins and outs of inference engineering, one build at a time.

Lesson 1:
https://shayansanjideh.github.io/learning-inference/lesson-01.html

Code:
https://github.com/shayansanjideh/learning-inference

**Attach:** [Image 5](thread-images/05-prefill-decode.png)

## Publishing notes

Draft only. Images are screenshots of the guide and its recorded measurements. Post 3 and post 4 report total generation time; the next lesson separates prefill and decode.
