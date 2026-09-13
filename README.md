# Learning inference

Learn how language models run by building the pieces yourself. Twelve lessons move from a small model server through generation loops, KV caches, batching, quantization, speculative decoding, profiling, and GPU kernels.

**[Read the guide](https://shayansanjideh.github.io/learning-inference/)** · **[Start lesson 1](https://shayansanjideh.github.io/learning-inference/lesson-01.html)**

Lesson 1 is complete. Lessons 2–12 are build plans in the website roadmap.

## Repository layout

- `lessons/01-baseline/`: runnable server, benchmark scripts, environment record, raw results, charts, exercises, and `THREAD-DRAFT.md`.
- `website/content/lesson-01.md`: maintained article source.
- `website/scripts/build_lesson.py`: renders the article and refreshes its downloadable code/data bundle.
- `website/dist/`: website, diagrams, artwork, and lesson downloads.
- `.github/workflows/pages.yml`: rebuilds and publishes the guide on each push to `main`.

Future lessons follow the same numbered-folder pattern, with a corresponding article and thread draft. Each gets a distinct nontechnical ink illustration when the lesson is built; future artwork is not generated ahead of time.

## Read or run lesson 1

Follow [the lesson README](lessons/01-baseline/README.md) for installation, the server, experiments, and exercises. The recorded benchmark uses SmolLM2-135M, FP32, batch 1, and two CPU threads. Results include all measured repetitions and exact timing boundaries.

## Build the website

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r website/requirements-build.txt
.venv/bin/python website/scripts/build_lesson.py
python3 -m http.server 8000 --directory website/dist
```

Open http://localhost:8000. Rebuilding the website uses the recorded data; it does not run inference or download model weights.

Edit the article in `website/content/lesson-01.md`, experimental code and thread in `lessons/01-baseline/`, and site styles in `website/dist/`. Build before committing. Website introductions address the reader; personal threads use I/me.

## Hosting

GitHub Pages serves the guide over HTTPS and updates through GitHub Actions. To use a custom domain, add it in repository Settings → Pages, configure the domain’s DNS for GitHub Pages, and enable HTTPS. See [GitHub’s custom-domain instructions](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site).
