# Lesson 1 — Twitter thread draft

Five posts, one image per post. Image 1 is a standalone lesson overview with the setup and measured results.

## 1/5

I’m on a mission to learn inference engineering, working through a lesson plan with Astra and building as I go. I’ll use X to build in public, document what I learn, and put together a game plan others can follow. First up: running and benchmarking a small LLM.

**Attach:** [Image 1](thread-images/01-introduction.png)

## 2/5

Lesson 1 uses SmolLM2-135M on a two-core CPU VM to explore tokenization, prefill, decode, and generation latency.

I ran two experiments: varying output length with a fixed prompt, then varying prompt length with a fixed output. Batch size stayed at 1 throughout.

**Attach:** [Image 2](thread-images/02-request-path.png)

## 3/5

With a 128-token prompt, increasing output from 16 to 64 tokens raised median generation time from 2.16s to 7.53s: 4× the output, approximately 3.5× the time.

Each condition had two warmup runs and five measured runs. Model loading and tokenization were outside the timer.

**Attach:** [Image 3](thread-images/03-output-scaling.png)

## 4/5

With output fixed at 32 tokens, increasing the prompt from 64 to 256 tokens raised median generation time from 3.43s to 4.75s.

Parallel prompt processing still has a cost. These total-generation timings motivate measuring prefill and decode separately in the next lesson.

**Attach:** [Image 4](thread-images/04-input-scaling.png)

## 5/5

Next: implementing a prefill + decode loop, checking its output against .generate(), and timing each phase separately.

Lesson 1 includes the server, benchmark code, raw measurements, charts, and exercises:

https://shayansanjideh.github.io/learning-inference/lesson-01.html

**Attach:** [Image 5](thread-images/05-prefill-decode.png)

## Publishing notes

Draft only. All timings are median total-generation times. Separate phase timings are planned for the next lesson.
