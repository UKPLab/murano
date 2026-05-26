"""Rank class-selective SST-2 SAE features."""

import json
from pathlib import Path

import torch

from murano.dataset import LabeledDataset
from murano.pipeline import Pipeline
from murano.steps import Load, SAEEncode, SAETopActivations, Save


DATASET, SPLIT = "stanfordnlp/sst2", "train"
MODEL_ID = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_8/width_16k/canonical"
N_EXAMPLES, TOP_FEATURES, K_CONTEXTS = 64, 10, 10
OUTPUT_DIR = "examples/artifacts/murano_sst2_sae_feature_enrichment"
STOPWORDS = set(
    "a an and all as by for from her his in is it its of on or some that the "
    "this to was where who with your".split()
)


def _rank_features(store, labels: list[int], tokenizer):
    """Return selective contrast rows and binary SST-2 label counts."""
    # Validate the binary label contract and activation batch shape.
    labels = [int(label) for label in labels]
    counts = {0: labels.count(0), 1: labels.count(1)}
    if set(labels) != {0, 1}:
        raise ValueError(f"SST-2 requires labels {{0, 1}}; got {counts}")

    acts = store.activations.float()
    if acts.dim() != 3 or acts.shape[0] != len(labels):
        raise ValueError(f"Expected [N, seq, n_features], got {acts.shape}")

    # Keep only non-padding, non-BOS, content-bearing token positions.
    mask = store.attention_mask.bool()
    if tokenizer.bos_token_id is not None:
        mask = mask & (store.tokens != tokenizer.bos_token_id)
    decoded = [
        tokenizer.decode([int(t)]).strip().lower()
        for t in store.tokens.reshape(-1).tolist()
    ]
    content = torch.tensor(
        [
            len(t) > 2 and t not in STOPWORDS and any(c.isalpha() for c in t)
            for t in decoded
        ]
    ).reshape(mask.shape)
    mask = mask & content

    # Score features by class contrast, selectivity, and top-token purity.
    token_counts = mask.sum(dim=1).clamp_min(1)
    pooled = (acts * mask.to(acts.dtype).unsqueeze(-1)).sum(dim=1)
    pooled = pooled / token_counts.to(acts.dtype).unsqueeze(-1)
    labels_t = torch.tensor(labels, dtype=torch.long)
    pos, neg = labels_t == 1, labels_t == 0
    pos_mean, neg_mean = pooled[pos].mean(0), pooled[neg].mean(0)
    delta = pos_mean - neg_mean
    effect = delta / pooled.std(0, unbiased=False).clamp_min(1e-6)
    pos_nz = (pooled[pos] > 0).sum(0)
    neg_nz = (pooled[neg] > 0).sum(0)
    pos_rate = pos_nz / pos.sum()
    neg_rate = neg_nz / neg.sum()
    selectivity = torch.where(delta >= 0, pos_rate - neg_rate, neg_rate - pos_rate)
    base_score = effect.abs() * selectivity.clamp_min(0)

    # Rank a candidate pool, then print associated-class token hits.
    rows, seq = [], acts.shape[1]
    labels_by_token = labels_t.repeat_interleave(seq)
    flat_mask = mask.reshape(-1)
    for feature_id in torch.argsort(base_score, descending=True)[: TOP_FEATURES * 30]:
        d = float(delta[feature_id].item())
        assoc_label = 1 if d >= 0 else 0
        flat = acts[..., feature_id].reshape(-1).masked_fill(~flat_mask, -torch.inf)
        _, top_idx = flat.topk(min(K_CONTEXTS, int(flat_mask.sum())))
        purity = float((labels_by_token[top_idx] == assoc_label).float().mean().item())
        side_mask = flat_mask & (labels_by_token == assoc_label)
        vals, idxs = flat.masked_fill(~side_mask, -torch.inf).topk(3)
        hits = []
        for val, flat_i in zip(vals.tolist(), idxs.tolist()):
            if val <= 0:
                continue
            n, j = divmod(int(flat_i), seq)
            token = tokenizer.decode([int(store.tokens[n, j])])
            hits.append(f"{token!r}: {store.texts[n].strip()}")
        if purity < 0.7 or not hits:
            continue
        rows.append(
            {
                "feature_id": int(feature_id),
                "assoc": "positive" if assoc_label else "negative",
                "score": float(base_score[feature_id].item() * purity),
                "delta": d,
                "effect": float(effect[feature_id].item()),
                "pos_nz": int(pos_nz[feature_id]),
                "neg_nz": int(neg_nz[feature_id]),
                "purity": purity,
                "hits": hits,
            }
        )
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:TOP_FEATURES]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, {"0": int(counts[0]), "1": int(counts[1])}


def main() -> None:
    """Run the two-phase SST-2 SAE feature enrichment example."""
    print("Preparing Murano model components...", flush=True)
    from murano.model import MuranoModel

    print("Loading SST-2 samples...", flush=True)
    dataset = LabeledDataset.from_hub(
        DATASET,
        text_column="sentence",
        label_column="label",
        split=SPLIT,
        n=N_EXAMPLES,
        label_names=["negative", "positive"],
    )
    print(f"Loading model: {MODEL_ID}", flush=True)
    model = MuranoModel(MODEL_ID)
    print(f"Encoding SAE activations: {SAE_RELEASE}/{SAE_ID}", flush=True)
    results = Pipeline(
        [Load(dataset), SAEEncode(model, release=SAE_RELEASE, sae_id=SAE_ID)]
    ).run()

    print("Ranking sentiment-contrastive features...", flush=True)
    rows, label_counts = _rank_features(
        results["sae_record"], list(dataset.labels), model.tokenizer
    )

    print("Collecting top activating contexts and saving artifacts...", flush=True)
    results = Pipeline(
        [
            SAETopActivations(
                model, k=K_CONTEXTS, feat_ids=[row["feature_id"] for row in rows]
            ),
            Save(output_dir=OUTPUT_DIR, model_id=MODEL_ID),
        ]
    ).run(results)

    print("Writing sentiment contrast JSON...", flush=True)
    contrast_path = Path(results["output_dir"]) / "sentiment_feature_contrast.json"
    payload = {
        "label_counts": label_counts,
        "score": "abs(effect) * class_rate_gap * top_token_purity",
        "features": rows,
    }
    contrast_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"SST-2 labels: negative={label_counts['0']} positive={label_counts['1']} | score: effect * selectivity * purity"
    )
    print("rank  feature  assoc      score  effect  pos>0  neg>0  purity  top hits")
    for row in rows:
        hits = " | ".join(row["hits"])
        print(
            f"{row['rank']:>4}  {row['feature_id']:>7}  {row['assoc']:<8}  {row['score']:>7.3f}  "
            f"{row['effect']:>6.2f}  {row['pos_nz']:>2}/{label_counts['1']}  "
            f"{row['neg_nz']:>2}/{label_counts['0']}  {row['purity']:>5.2f}  {hits}"
        )
    print(f"\nArtifacts: {results['output_dir']} | contrast: {contrast_path}")


if __name__ == "__main__":
    main()
