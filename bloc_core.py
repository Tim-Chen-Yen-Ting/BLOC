import numpy as np
from steme_core import embed, STEME_nli, cosine_similarity

def add_to_cluster(memory, tag_pool=None, meta_tag_pool=None, threshold=0.15, floor=0.0):
    """
    Adds memory to an existing or new cluster.

    Absorption decision uses NLI against existing cluster names (not centroid cosine),
    so it asks "does this memory imply the same concept?" rather than "is this embedding
    close to a drifted centroid?" — avoiding the cosine crowding collapse.

    CI threshold still applies: high CI (rigid) → low threshold → absorbs more.
    Centroid is still maintained for downstream uses (conflict detection, etc.).

    Args:
        memory (dict): Must have 'content' and 'importance'
        tag_pool (list): cluster dicts: {'name', 'centroid', 'net_importance'}
        meta_tag_pool (list): predefined label strings for naming new clusters
        threshold (float): min NLI confidence to absorb into existing cluster (= 1 - CI)
        floor (float): min NLI confidence (vs the full label pool) required to name a new
            cluster at all — below this, the memory doesn't imply any label well enough to
            force one, so it's left untagged instead of being crammed into the closest fit.

    Returns:
        str | None: name of the cluster the memory was added to, or None if no label
            (existing or new) cleared its confidence bar.
    """
    content = memory['content'] if not isinstance(memory['content'], np.ndarray) else ""
    new_vec = memory['content'] if isinstance(memory['content'], np.ndarray) else embed(content)
    memory_importance = memory.get('importance', 0.0)

    if tag_pool and content:
        # Drop clusters with invalid centroids
        valid_clusters = [c for c in tag_pool if not np.isnan(c['centroid']).any()]
        tag_pool[:] = valid_clusters

        if tag_pool:
            existing_names = [c['name'] for c in tag_pool]
            # NLI against existing cluster names — multi_label=True gives independent scores
            # so results don't collapse as more clusters are added
            nli_results = STEME_nli(content, existing_names, top_k=1, multi_label=True)
            best_score, best_label_dict = nli_results[0]
            best_label = best_label_dict['content']

            if best_score >= threshold:
                best_match = next((c for c in tag_pool if c['name'] == best_label), None)
                if best_match is not None:
                    old_weight = best_match.get('net_importance', 0.0)
                    new_weight = old_weight * 0.95 + memory_importance
                    if new_weight > 0:
                        numerator = np.array(best_match['centroid']) * old_weight + new_vec * memory_importance
                        updated = numerator / new_weight
                        updated /= np.linalg.norm(updated) + 1e-12
                        best_match['centroid'] = np.nan_to_num(updated)
                        best_match['net_importance'] = new_weight
                    return best_match['name']

    # No existing cluster matched — name the new one via NLI against the full label pool
    if meta_tag_pool and len(meta_tag_pool) != 0:
        labels = [m["content"] if isinstance(m, dict) else m for m in meta_tag_pool]
        if content:
            best_score, best_label_dict = STEME_nli(content, labels, top_k=1)[0]
            if best_score < floor:
                return None
            new_name = best_label_dict['content']
        else:
            new_name = str(memory['content'])
        # Guard: if a cluster with this name already exists, merge into it rather than
        # spawning a duplicate (happens when NLI score fell below threshold but the
        # best-matching label is the same as an existing cluster name).
        existing = next((c for c in tag_pool if c['name'] == new_name), None)
        if existing is not None:
            old_w = existing.get('net_importance', 0.0)
            new_w = old_w * 0.95 + memory_importance
            if new_w > 0:
                num = np.array(existing['centroid']) * old_w + new_vec * memory_importance
                updated = num / new_w
                existing['centroid'] = np.nan_to_num(updated / (np.linalg.norm(updated) + 1e-12))
                existing['net_importance'] = new_w
            return existing['name']
        tag_pool.append({'name': new_name, 'centroid': new_vec, 'net_importance': memory_importance})
        return new_name

    tag_pool.append({'name': str(content or memory['content']), 'centroid': new_vec, 'net_importance': memory_importance})
    return str(content or memory['content'])

def detect_belief_conflicts(belief_clusters, threshold=-0.6):
    """
    Detects belief clusters that contradict each other.

    Args:
        belief_clusters (list of dict): Each dict has 'name', 'centroid', 'net_importance'.
        threshold (float): Cosine similarity below this is considered conflict.

    Returns:
        list of tuples: (belief1_dict, belief2_dict, similarity, importance1, importance2)
    """
    conflicts = []
    for i in range(len(belief_clusters)):
        for j in range(i + 1, len(belief_clusters)):
            b1 = belief_clusters[i]
            b2 = belief_clusters[j]
            sim = cosine_similarity(
                np.array(b1['centroid']).reshape(1, -1), np.array(b2['centroid']).reshape(1, -1)
            )[0][0]
            if sim < threshold:
                conflicts.append((b1, b2, sim, b1['net_importance'], b2['net_importance']))
    return conflicts
