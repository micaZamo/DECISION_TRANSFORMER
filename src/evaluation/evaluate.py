import numpy as np
import torch
from tqdm import tqdm
from .metrics import hit_rate_at_k, ndcg_at_k, mrr_score


@torch.no_grad()
def evaluate_baseline(model, test_users, k_list=[5,10,20], context_len=20):
    metrics = {f"HR@{k}": [] for k in k_list}
    metrics.update({f"NDCG@{k}": [] for k in k_list})
    metrics["MRR"] = []

    for user in tqdm(test_users, desc="Evaluando baseline"):
        items = user["items"]
        ratings = user["ratings"]

        for t in range(context_len, len(items)):
            history = items[t-context_len:t]
            target_item = items[t]

            recs = model.recommend(history, k=max(k_list))

            for k in k_list:
                metrics[f"HR@{k}"].append(hit_rate_at_k(recs, target_item, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(recs, target_item, k))

            metrics["MRR"].append(mrr_score(recs, target_item))

    return {m: float(np.mean(v)) for m, v in metrics.items()}



@torch.no_grad()
def evaluate_decision_transformer(model, test_users, device, k_list=[5,10,20], context_len=20):
    metrics = {f"HR@{k}": [] for k in k_list}
    metrics.update({f"NDCG@{k}": [] for k in k_list})
    metrics["MRR"] = []

    for user in tqdm(test_users, desc="Evaluando Decision Transformer"):
        group = user["group"]
        items = user["items"]
        ratings = user["ratings"]

        for t in range(context_len, len(items)):
            hist_items = items[t-context_len:t]
            hist_ratings = ratings[t-context_len:t]
            target_item = items[t]

            rtg_val = float(sum(hist_ratings))

            states = torch.tensor(hist_items).long().unsqueeze(0).to(device)
            actions = torch.tensor(hist_items).long().unsqueeze(0).to(device)
            rtg_in = torch.full((1, context_len, 1), rtg_val).float().to(device)
            timesteps = torch.arange(context_len).long().unsqueeze(0).to(device)
            groups = torch.tensor([group]).long().to(device)

            logits = model(states, actions, rtg_in, timesteps, groups)
            scores = logits[0, -1, :].cpu().numpy()

            ranked_items = np.argsort(scores)[::-1].tolist()

            for k in k_list:
                metrics[f"HR@{k}"].append(hit_rate_at_k(ranked_items, target_item, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, target_item, k))

            metrics["MRR"].append(mrr_score(ranked_items, target_item))

    return {m: float(np.mean(v)) for m, v in metrics.items()}
