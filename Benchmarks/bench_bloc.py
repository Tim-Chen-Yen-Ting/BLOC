# Benchmarks/bench_bloc.py
import argparse, json, time, os, sys, platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import psutil
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans

print("bench_bloc starting")

# Optional: use your sentence embeddings for realism
try:
    from steme_core import setModel, embed
    HAVE_EMBED = True
except Exception:
    HAVE_EMBED = False
    embed = None

from bloc_core import add_to_cluster

# Benchmark-only adapter: gives add_to_cluster() a sklearn-style incremental
# clustering interface (.update()/.n_clusters()) so it can be measured
# alongside KMeans/MiniBatchKMeans. Not used anywhere in production —
# runtime code calls add_to_cluster() directly.
class IncrementalClusterer:
    def __init__(self, dim, thresh=0.35):
        self.tag_pool = []
        self.meta_tag_pool = []  # Your STEME tag dictionary
        self.name_to_id = {}  # Maps cluster names to integer IDs
        self.id_to_name = {}  # Maps integer IDs to cluster names
        self.next_id = 0
        self.thresh = thresh

    def update(self, x):
        # If input is already a vector, use as is; otherwise embed it
        if isinstance(x, np.ndarray):
            vec = x.astype(np.float32)
        else:
            vec = embed(x).astype(np.float32)

        # Normalize to match all centroids
        vec /= np.linalg.norm(vec) + 1e-12

        memory = {'content': vec, 'importance': 1.0}

        cluster_name = add_to_cluster(
            memory=memory,
            tag_pool=self.tag_pool,
            meta_tag_pool=self.meta_tag_pool,
            threshold=self.thresh
        )

        # Ensure deterministic integer ID per cluster name
        if cluster_name not in self.name_to_id:
            cluster_id = self.next_id
            self.name_to_id[cluster_name] = cluster_id
            self.id_to_name[cluster_id] = cluster_name
            self.next_id += 1

        return self.name_to_id[cluster_name]

    def n_clusters(self):
        return len(self.name_to_id)

    def get_cluster_name_by_index(self, index):
        """Returns the cluster name for a given integer index."""
        return self.id_to_name.get(index, None)

    def get_cluster_index_by_name(self, name):
        """Returns the integer index for a given cluster name."""
        return self.name_to_id.get(name, None)

IncrementalImpl = IncrementalClusterer
# -----------------------------
# Synthetic streaming generator
# -----------------------------
sigma = 0.045 # spread around centroid
def make_stream(n=50000, dim=384, k_true=8, drift=0.002, p_new=0.001, seed=7,
                use_text_embeddings=False, model_name="all-MiniLM-L6-v2"):
    """
    Yields (vector, true_label). Start with k_true clusters; centroids drift.
    With prob p_new, spawn a new cluster.
    If use_text_embeddings and steme_core is available, generate short texts & embed;
    otherwise sample from Gaussians around moving centroids.
    """
    rng = np.random.default_rng(seed)
    labels = list(range(k_true))
    centroids = rng.normal(0, 1, size=(k_true, dim)).astype("float32")
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    if use_text_embeddings and HAVE_EMBED:
        setModel(model_name)
    for i in range(n):
        # occasionally spawn a new cluster
        if rng.random() < p_new:
            labels.append(len(labels))
            c = rng.normal(0, 1, size=(dim,)).astype("float32")
            c /= (np.linalg.norm(c) + 1e-12)
            centroids = np.vstack([centroids, c])

        lab = int(rng.integers(0, len(labels)))
        c = centroids[lab]

        if use_text_embeddings and HAVE_EMBED:
            noise = rng.choice(["update","policy","market","match","series","earnings","summit","chip","coach"])
            text = f"news {noise} about topic_{lab}"
            v = embed(text)
            x = v.astype("float32")
        else:
            x = (c + rng.normal(0, sigma, size=c.shape)).astype("float32")

        # drift all centroids slightly (simulate non-stationarity)
        delta = rng.normal(0, drift, size=centroids.shape).astype("float32")
        centroids = centroids + delta
        centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

        yield x, lab


# -----------------------------
# Simple reference incremental clusterer (fallback)
# -----------------------------
class SimpleIncrementalClusterer:
    """
    Cosine-based online clusterer:
    - keep running centroids (L2-normalized)
    - if max sim < thresh -> start new cluster (with small epsilon slack)
    - else merge into nearest (EMA update)
    Deterministic and fast; not sophisticated.
    """
    def __init__(self, dim, thresh=0.6, ema=0.5, near_eps=0.02):
        self.dim = dim
        self.thresh = float(thresh)
        self.ema = float(ema)
        self.near_eps = float(near_eps)
        self.centroids = np.zeros((0, dim), dtype=np.float32)
        self.counts = []

    def _norm(self, v):
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def update(self, x):
        x = self._norm(x.astype(np.float32))
        if len(self.centroids) == 0:
            self.centroids = x.reshape(1, -1)
            self.counts = [1]
            return 0
        sims = self.centroids @ x
        j = int(np.argmax(sims))
        if sims[j] < self.thresh:
            # near-miss? allow small slack to avoid uncontrolled cluster growth
            if sims[j] >= self.thresh - self.near_eps:
                cj = self.centroids[j]
                cj = self._norm(self.ema * cj + (1 - self.ema) * x)
                self.centroids[j] = cj
                self.counts[j] += 1
                return j
            # otherwise open new cluster
            self.centroids = np.vstack([self.centroids, x])
            self.counts.append(1)
            return len(self.counts) - 1
        # merge into j
        cj = self.centroids[j]
        cj = self._norm(self.ema * cj + (1 - self.ema) * x)
        self.centroids[j] = cj
        self.counts[j] += 1
        return j

    def n_clusters(self):
        return len(self.counts)


def get_incremental(dim, thresh):
    if IncrementalImpl is not None: #not
        try:
            return IncrementalImpl(dim=dim, thresh=thresh)  # adapt if your ctor differs
        except Exception:
            pass
    return SimpleIncrementalClusterer(dim=dim, thresh=thresh)


# -----------------------------
# Metrics
# -----------------------------
def purity(labels_true, labels_pred):
    #Majority-vote purity (higher is better).
    clusters = defaultdict(list)
    for t, p in zip(labels_true, labels_pred):
        clusters[p].append(t)
    num = sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return num / max(1, len(labels_true))

def cluster_diagnostics(labels_pred):
    #n_clusters, singleton_rate, avg_size, p99_size.
    sizes = Counter(labels_pred)
    n_clu = len(sizes)
    singleton_rate = sum(1 for s in sizes.values() if s == 1) / max(1, n_clu)
    avg_sz = len(labels_pred) / max(1, n_clu)
    p99_sz = float(np.quantile(list(sizes.values()), 0.99))
    return n_clu, singleton_rate, avg_sz, p99_sz

def safe_silhouette(X, labels, metric="cosine", max_samples=5000, seed=0):
    if len(X) <= 1 or len(set(labels)) < 2:
        return float("nan")
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        Xs = X[idx]
        ys = np.asarray(labels)[idx]
    else:
        Xs, ys = X, np.asarray(labels)
    try:
        return float(silhouette_score(Xs, ys, metric=metric))
    except Exception:
        return float("nan")


# -----------------------------
# Benchmark driver
# -----------------------------
def run_bloc_bench(n=50000, dim=384, thresh=0.6, drift=0.002, p_new=0.001,
                   use_text_embeddings=False, model_name="all-MiniLM-L6-v2",
                   kmeans_batch=4096, seed=7, out="Benchmarks/bloc_run.json",
                   kmeans_subsample=20000):
    process = psutil.Process(os.getpid())
    stream = list(make_stream(n=n, dim=dim, drift=drift, p_new=p_new,
                              use_text_embeddings=use_text_embeddings,
                              model_name=model_name, seed=seed))
    X = np.stack([v for v, _ in stream], axis=0)
    y_true = np.array([lab for _, lab in stream])
    print("True labels:", len(set(y_true)))


    # Incremental
    inc = get_incremental(dim=dim, thresh=thresh)
    t0 = time.perf_counter()
    y_pred_inc = np.empty(n, dtype=int)
    for i, (v, _) in enumerate(stream):
        y_pred_inc[i] = inc.update(v)
    inc_elapsed = time.perf_counter() - t0
    inc_ops_sec = n / max(inc_elapsed, 1e-9)

    # Quality + diagnostics for incremental
    inc_ari = adjusted_rand_score(y_true, y_pred_inc)
    inc_purity = purity(y_true, y_pred_inc)
    inc_sil = safe_silhouette(X, y_pred_inc, metric="cosine")
    n_clu, singleton_rate, avg_sz, p99_sz = cluster_diagnostics(y_pred_inc)
    # ---- Baselines ----
    # Cap K so baselines are valid and reasonably fast
    # tie cap to batch size; e.g., <= 1024 when batch=4096
    k_cap = max(2, kmeans_batch // 4)

    # MiniBatchKMeans: true streaming via partial_fit
    def run_minibatch_kmeans_stream():
        k_local = min(max(2, n_clu), k_cap)
        est = MiniBatchKMeans(n_clusters=k_local,
                              batch_size=min(1024, kmeans_batch),
                              n_init=1, random_state=seed)
        y_pred = np.empty(n, dtype=int)
        t0 = time.perf_counter()
        for start in range(0, n, kmeans_batch):
            end = min(n, start + kmeans_batch)
            est.partial_fit(X[start:end])
            y_pred[start:end] = est.predict(X[start:end])
        elapsed = time.perf_counter() - t0
        return y_pred, elapsed

    # KMeans: quality anchor on a subsample, then predict all
    def run_kmeans_subsample(m=kmeans_subsample):
        k_local = min(max(2, n_clu), k_cap)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(n, m), replace=False)
        est = KMeans(n_clusters=k_local, n_init=10, random_state=seed)
        t0 = time.perf_counter()
        est.fit(X[idx])
        y_pred_all = est.predict(X)
        elapsed = time.perf_counter() - t0
        return y_pred_all, elapsed

    y_mb, mb_elapsed = run_minibatch_kmeans_stream()
    y_km, km_elapsed = run_kmeans_subsample()

    mb_ops_sec = n / max(mb_elapsed, 1e-9)
    km_ops_sec = n / max(km_elapsed, 1e-9)

    km_ari = adjusted_rand_score(y_true, y_km)
    mb_ari = adjusted_rand_score(y_true, y_mb)
    km_purity = purity(y_true, y_km)
    mb_purity = purity(y_true, y_mb)
    km_sil = safe_silhouette(X, y_km, metric="cosine")
    mb_sil = safe_silhouette(X, y_mb, metric="cosine")

    peak_mb = getattr(process.memory_info(), "rss", 0) / (1024 * 1024)

    # Console table
    def row(name, ops, ari, pur, sil):
        return f"{name:<18} | {ops:10.0f} | {ari:6.3f} | {pur:6.3f} | {sil:7.3f}"
    print("BLOC benchmark (n={}, sigma = {}, dim={}, thresh={}, drift={}, p_new={}, text_emb={})"
          .format(n, sigma, dim, thresh, drift, p_new, use_text_embeddings))
    print("Method             |   ops/sec  |   ARI  | Purity | Silhouette")
    print("-----------------------------------------------------------------")
    print(row("Incremental(BLOC)", inc_ops_sec, inc_ari, inc_purity, inc_sil))
    print(row("KMeans (global)",   km_ops_sec,  km_ari,  km_purity,  km_sil))
    print(row("MiniBatchKMeans",   mb_ops_sec,  mb_ari,  mb_purity,  mb_sil))
    print(f"\nObserved clusters (inc): {n_clu} | singletons: {singleton_rate:.2%} | "
          f"avg size: {avg_sz:.1f} | p99 size: {p99_sz:.0f} | Peak RAM ~ {peak_mb:.1f} MB")

    # JSON artifact
    out_obj = {
        "params": {
            "n": n, "dim": dim, "thresh": thresh, "drift": drift, "p_new": p_new,
            "text_embeddings": bool(use_text_embeddings and HAVE_EMBED),
            "model_name": model_name if use_text_embeddings else None,
            "kmeans_batch": kmeans_batch, "kmeans_subsample": kmeans_subsample
        },
        "ops_sec": {
            "incremental": inc_ops_sec,
            "kmeans_global": km_ops_sec,
            "minibatch": mb_ops_sec
        },
        "quality": {
            "incremental": {"ARI": inc_ari, "purity": inc_purity, "silhouette": inc_sil},
            "kmeans_global": {"ARI": km_ari, "purity": km_purity, "silhouette": km_sil},
            "minibatch": {"ARI": mb_ari, "purity": mb_purity, "silhouette": mb_sil},
        },
        "diagnostics": {
            "n_clusters_inc": int(n_clu),
            "singleton_rate": float(singleton_rate),
            "avg_cluster_size": float(avg_sz),
            "p99_cluster_size": float(p99_sz),
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "peak_ram_mb": float(peak_mb),
        "timestamp": time.time(),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    return out_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--thresh", type=float, default=0.42)  # safer default
    ap.add_argument("--drift", type=float, default=0.002)
    ap.add_argument("--p-new", type=float, default=0.001)
    ap.add_argument("--text-emb", action="store_true", help="use sentence embeddings via steme_core")
    ap.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--kmeans-batch", type=int, default=4096)
    ap.add_argument("--kmeans-subsample", type=int, default=20000)
    ap.add_argument("--out", type=str, default="Benchmarks/bloc_run.json")
    args = ap.parse_args()

    run_bloc_bench(n=args.n, dim=args.dim, thresh=args.thresh, drift=args.drift,
                   p_new=args.p_new, use_text_embeddings=args.text_emb,
                   model_name=args.model_name, kmeans_batch=args.kmeans_batch,
                   kmeans_subsample=args.kmeans_subsample, out=args.out)

if __name__ == "__main__":
    main()
