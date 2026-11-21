# BLOC

**“Beliefs are not stored. They are clustered.”**

---

## 🧩 Belief & Label-Oriented Clustering (BLOC)

**BLOC** is a minimal clustering engine for managing belief states, reflections, and memory identity in AI cognition systems. It was designed to support:
- Conceptual clustering of agent memories
- Dynamic identity evolution based on semantic similarity
- Meaning formation through accumulation

It classifies memories against cluster identities via zero-shot NLI (through [STEME](https://github.com/tim-chen-yen-ting/steme)'s `STEME_nli`) rather than raw centroid-cosine distance — so absorption asks "does this memory imply the same concept?" instead of "is this embedding close to a drifted centroid?", which avoids the cosine-crowding collapse that plain centroid matching runs into as clusters accumulate.

---

## 🔧 Core API

- `add_to_cluster(memory, tag_pool=None, meta_tag_pool=None, threshold=0.15, floor=0.0)`
   Adds a new memory to the best-fitting cluster by NLI confidence against existing cluster names, or creates a new one (named via NLI against `meta_tag_pool`) if nothing clears `threshold`. `floor` is a separate, lower confidence bar on that new-cluster naming step — if the best-matching label in `meta_tag_pool` doesn't even clear `floor`, the memory isn't forced into any cluster and `None` is returned instead. Mutates `tag_pool` in place. Each memory is expected to contain:
   ```python
   {
       "content": "I believe in freedom",
       "importance": 0.9
   }
   ```
   Each cluster in `tag_pool` is a dict: `{"name", "centroid", "net_importance"}`. The
   centroid is still cosine-averaged and maintained on every absorption — it's kept for
   downstream uses like conflict detection, even though it no longer drives the
   absorption decision itself.

- `detect_belief_conflicts(belief_clusters, threshold=-0.6)`
   Scans all cluster pairs for centroid cosine similarity below `threshold`, returning
   `(cluster_a, cluster_b, similarity, importance_a, importance_b)` tuples — i.e. beliefs
   that have drifted into semantic opposition.

Clusters are dynamically updated with:
- Recalculated (and re-normalized) centroids
- Accumulated importance (EMA-weighted: `old_weight * 0.95 + memory_importance`)
- Memory membership

--- 

## 💡 Example Usage
```python
from bloc_core import add_to_cluster
from steme_core import embed
memories = [
    {"content": "I believe in freedom", "importance": 0.9},
    {"content": "Order must be preserved", "importance": 0.8},
    {"content": "Everyone deserves autonomy", "importance": 0.7}
]

clusters = [{
        'name': "Freedom",
        'centroid': embed("I support freedom"),
        'net_importance': 0
    }]
meta_clusters = [
    "Freedom", "Order", "Equality", "Justice", "Authority",
    "Discipline", "Autonomy", "Loyalty", "Obedience", "Responsibility",
]
for mem in memories:
    add_to_cluster(mem, clusters, meta_clusters, threshold=0.15)

print("\nClustered Memory:")
for c in clusters:
    print("Cluster Name: ",c['name'], " | Cluster Centroid: ", c['centroid'][:3], " | Cluster Importance: ",c['net_importance'])

```

---

## 🧠 Use Cases
- Social agents forming identity via accumulated belief clusters
- Reflection-based belief consolidation
- Semantic memory grouping for simulations or cognitive models

---

## 🔄 Embedding Compatibility
BLOC depends on [STEME](https://github.com/tim-chen-yen-ting/steme) directly — `embed`
for centroid vectors and `STEME_nli` for cluster absorption/naming, not just as a
loosely-compatible companion tool. Installing BLOC pulls STEME in as a dependency.
If a memory's `content` is already a vector (`np.ndarray`), BLOC uses it as-is instead
of re-embedding.

## ⛓️ Installation
```bash
pip install git+https://github.com/tim-chen-yen-ting/bloc.git
```

---

## 📜 License
MIT (open use encouraged)

---

## 🙌 Credits
Created by Tim Chen\
Built to power the Civilizism agent belief model\
Designed for use in semantic cognition pipelines.

---
> “Memory fades. Identity clusters.”

---

Built by [Yen-Ting (Tim) Chen](https://tim-chen-yen-ting.github.io/) · [Portfolio](https://tim-chen-yen-ting.github.io/)
