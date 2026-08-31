# 📊 Benchmark Results

We evaluated two incremental clustering strategies — the NLI-based algorithm `bloc_core.add_to_cluster`
ships today, and the pure centroid-cosine approach it replaced — against `KMeans` and `MiniBatchKMeans`
baselines on a synthetic streaming dataset with drifting cluster centroids and stochastic cluster spawning.
Quality: adjusted Rand index (ARI), purity, cosine silhouette. Speed: throughput (ops/sec).

## Setup
- Stream generator (`make_stream`): starts with `k_true` Gaussian cluster centroids that drift
  slightly each step, occasionally spawning a new cluster (`p_new`) — no text, no `--text-emb`.
- `Incremental (BLOC, NLI)`: today's `bloc_core.add_to_cluster`, run through the same streaming
  interface as the baselines.
- `Incremental (cosine, legacy)`: `SimpleIncrementalClusterer` in this benchmark file — nearest-centroid
  cosine matching with EMA updates, standing in for `add_to_cluster`'s pre-NLI design.
- Environment: Local CPU, Python 3.13.

## Why both incremental methods are run

`add_to_cluster` was rewritten (2026-06-08) to decide cluster absorption via zero-shot NLI against
**existing cluster names**, instead of centroid cosine distance. That decision only fires when
`memory['content']` is a real string — feed it a raw embedding vector with no text and `content` is
`""`, so the NLI path never engages at all. This benchmark's stream is vectors only (no text), which
is exactly the setup the *original* pre-NLI benchmark (and its "0.703 ARI, beats KMeans" result) used —
so running both algorithms side by side on it shows precisely what the NLI rewrite gave up for
vector-only input.

## Results — run 2026-08-31

```bash
python Benchmarks/bench_bloc.py --n 2000 --thresh 0.42 --out Benchmarks/bloc_run.json
```

| Method                         | ops/sec | ARI       | Purity | Silhouette |
| ------------------------------ | ------: | --------: | -----: | ---------: |
| **Incremental (cosine, legacy)** |  63,225 | **1.000** |  1.000 |      0.342 |
| KMeans (global)                 |   2,064 |     0.937 |  0.973 |      0.321 |
| MiniBatchKMeans                 |     231 |     0.839 |  0.931 |      0.275 |
| Incremental (BLOC, NLI)         |     306 |     0.000 |  1.000 |        nan |

Observed clusters — cosine: 14 (0% singleton, matching the 14 true labels exactly) | NLI: 2000
(100% singleton — every point became its own cluster) | Peak RAM ~442 MB

Raw JSON: [`bloc_run.json`](./bloc_run.json).

## Key Takeaways

- **The pre-NLI cosine approach is excellent on this benchmark**: perfect ARI, recovers the exact
  true cluster count (14), and does it at ~30x the throughput of batch `KMeans` and ~270x
  `MiniBatchKMeans` — no retraining, one pass, streaming. This is the capability the original
  "beats KMeans" claim was describing, and it holds up (in fact beats the historical 0.703 ARI
  figure outright, likely due to threshold/parameter differences between then and now).
- **Today's shipped NLI-based algorithm cannot cluster this stream at all** (ARI 0.000, every
  point its own cluster) — not because it's worse at the same task, but because it's solving a
  different one: it needs real text content to reason about via NLI, and this stream has none.
  This is a genuine capability gap versus the pre-NLI design for any vector-only / pre-embedded
  use case, not a benchmark artifact.
- Civilizism's own production usage (`memCore.py`) always passes real memory text to
  `add_to_cluster`, so this gap doesn't currently bite there — but it means `bloc_core` (the
  standalone package) is no longer a drop-in replacement for "cluster arbitrary embedding
  vectors," which the README's centroid-only description used to imply.

## Reproducing the Benchmarks

```bash
python Benchmarks/bench_bloc.py --n 2000 --thresh 0.42 --out Benchmarks/bloc_run.json
```
Add `--text-emb` to run on real sentence embeddings instead of synthetic Gaussian vectors — but
note the synthetic text generator in this harness is a small fixed combinatorial vocabulary
(9 words × topics), not continuous drift, so it isn't a like-for-like comparison to the table above.
