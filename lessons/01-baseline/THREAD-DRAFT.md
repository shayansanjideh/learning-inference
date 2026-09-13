# Lesson 1 — Twitter thread draft

## 1/5

I’m learning inference engineering by building and measuring the pieces.

Lesson 1: running a small LLM server, benchmarking prompt and output lengths, and understanding prefill vs. decode.

I put the code, results, and walkthrough into a public guide.

## 2/5

My setup: SmolLM2-135M on a two-core CPU VM.

I kept batch size at 1, warmed up each condition, then measured it five times.

The timer covers generation. Model loading, tokenization, and HTTP overhead are outside it.

## 3/5

With a 128-token prompt, going from 16 to 64 output tokens increased median generation time from 2.16s to 7.53s.

4× the output, about 3.5× the time.

Output tokens per second rose too: the initial prompt-processing cost was spread over more generated tokens.

## 4/5

With output fixed at 32 tokens, increasing the prompt from 64 to 256 tokens raised median generation time from 3.43s to 4.75s.

Processing a prompt in parallel still takes work.

Next I’ll separate prefill and decode timings to see where the extra time goes.

## 5/5

Next lesson: writing my own prefill + decode loop and checking it against .generate().

Lesson 1 has the server, charts, raw measurements, and exercises:
https://shayansanjideh.github.io/learning-inference/lesson-01.html

Code:
https://github.com/shayansanjideh/learning-inference

## Attachments

- Post 1: screenshot of the lesson introduction.
- Post 3: `results/output-scaling.png`.
- Post 4: `results/input-scaling.png`.

Draft only; nothing has been posted to Twitter. Results are median total-generation timings; separate phase measurements belong to the next lesson.
