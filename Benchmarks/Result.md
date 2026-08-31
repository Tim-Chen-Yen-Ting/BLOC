# 📊 Benchmark Results

We evaluated **BLOC**'s incremental clustering (`add_to_cluster`) against `KMeans` and
`MiniBatchKMeans` baselines on a synthetic streaming dataset, using adjusted Rand index
(ARI), purity, and cosine silhouette as quality metrics, and throughput (ops/sec) for speed.

## Setup
- Stream generator (`make_stream`): starts with `k_true` Gaussian cluster centroids that
  drift slightly each step, occasionally spawning a new cluster (`p_new`).
- With `--text-emb`, each point is instead a sentence `"news {noise} about topic_{lab}"`
  (`noise` drawn from a 9-word pool) embedded via STEME's `embed()`, so BLOC clusters real
  text content rather than raw vectors.
- Environment: Local CPU inference, Python 3.13, `all-MiniLM-L6-v2` for `--text-emb` runs.

## ⚠️ Important caveat: this harness predates the NLI-based rewrite

`bloc_core.add_to_cluster` was rewritten to decide absorption via zero-shot NLI against
**existing cluster names**, not centroid cosine distance (see README). That decision only
engages when `memory['content']` is a non-empty string — with plain vectors (no `--text-emb`),
`content` is always `""`, so every point falls through to "no cluster matched" and gets
its own cluster. Civilizism's old `bloc_run_sanity.json` run (20k points, no `--text-emb`)
shows exactly this: 936 clusters, 95.7% singleton rate. **That isn't a regression — it's the
new algorithm correctly refusing to cluster on vectors with no text to reason about.** Don't
run this benchmark without `--text-emb` and expect meaningful clustering.

Even with `--text-emb`, this harness's `IncrementalClusterer` adapter (`Benchmarks/bench_bloc.py`)
hardcodes `meta_tag_pool = []`, so new-cluster naming falls through to BLOC's last resort
(name the cluster after the raw memory content) rather than the curated concept-label pool
`add_to_cluster` is actually designed around (see the README example — `"Freedom"`, `"Order"`,
etc.). Cluster identity in this benchmark is therefore "does NLI think this sentence entails an
existing raw sentence used as a previous cluster's name" — a real code path, but not the
intended production usage. Treat the numbers below as a smoke test of the streaming/absorption
mechanics, not a measure of real-world clustering quality.

## Latest run — `--text-emb`, run 2026-08-31

```
python Benchmarks/bench_bloc.py --n 2000 --thresh 0.42 --text-emb --out Benchmarks/bloc_run.json
```

| Method             | ops/sec | ARI   | Purity | Silhouette |
| ------------------ | ------: | ----: | -----: | ---------: |
| Incremental (BLOC)  |     166 | 0.182 |  1.000 |      1.000 |
| KMeans (global)     |     404 | 0.182 |  1.000 |      1.000 |
| MiniBatchKMeans     |     601 | 0.182 |  1.000 |      1.000 |

Observed clusters: 81 | singletons: 0.0% | avg size: 24.7 | p99 size: 35 | Peak RAM ~525 MB

All three methods land on identical ARI/purity/silhouette. That's not a coincidence: the
9-word noise pool × 9 true topics means only 81 distinct sentences ever get generated
(`embed()` caches by exact string), so `X` has just 81 unique row vectors each repeated
~24.7 times — a highly degenerate, discrete input rather than continuous semantic variation.
Purity of 1.0 is a predictable consequence of 81 clusters mapping onto 9 true labels (more
clusters than labels makes purity easy), and the low ARI (0.182) reflects that BLOC — like
both KMeans baselines here — is over-segmenting relative to the 9 ground-truth topics on
this particular synthetic setup, not that one method beats another.

Raw JSON: [`bloc_run.json`](./bloc_run.json).

## Key Takeaways

- The benchmark harness needs `--text-emb` to exercise the current (NLI-based) algorithm at
  all; the vector-only mode is not a valid way to benchmark this version of BLOC.
- Even with `--text-emb`, the harness's empty `meta_tag_pool` means it's testing raw-sentence-vs-
  raw-sentence NLI entailment, not BLOC's intended curated-label clustering — a harness rewrite
  (real `meta_tag_pool`, richer/continuous synthetic text) would be needed for a benchmark that
  actually reflects production usage. Flagging this rather than fixing it here, since it's a
  harness redesign, not a sync task.
- Ops/sec for the incremental method (166/sec) is well below both KMeans baselines here, but
  those baselines get to see the whole batch upfront (or a large batch at a time) — this run
  doesn't isolate a fair per-item streaming comparison.

## Reproducing the Benchmarks

```bash
python Benchmarks/bench_bloc.py --n 2000 --thresh 0.42 --text-emb --out Benchmarks/bloc_run.json
```
Drop `--text-emb` only if you specifically want to exercise the vector-only fallback path
(new-cluster-per-point, by design — see caveat above).
