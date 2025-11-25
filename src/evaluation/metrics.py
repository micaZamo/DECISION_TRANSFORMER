import numpy as np
import torch

def hit_rate_at_k(recommended, target, k=10):
    return 1.0 if target in recommended[:k] else 0.0


def ndcg_at_k(recommended, target, k=10):
    topk = recommended[:k]
    if target not in topk:
        return 0.0
    rank = topk.index(target) + 1
    return 1.0 / np.log2(rank + 1)


def mrr_score(recommended, target):
    if target not in recommended:
        return 0.0
    rank = recommended.index(target) + 1
    return 1.0 / rank
