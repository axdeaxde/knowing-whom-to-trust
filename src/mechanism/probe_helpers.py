"""Grouped classifier selection and metadata controls."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
NUMERIC=["checkpoint_position","prompt_tokens","distance_to_end","last_trial","last_feedback_truthful","last_won"]

def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def aggregate(meta, predictions):
    d = meta[['run_id', 'advisor_id', 'label']].copy()
    d['score'] = predictions
    return d.groupby(['run_id', 'advisor_id', 'label'], sort=True, as_index=False).score.mean()


def fit(x, y, c):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver='liblinear', max_iter=3000, random_state=20260923))
    pipeline.fit(x, y)
    return pipeline


def cv(meta, x, c):
    y = meta.label.to_numpy()
    pred = np.zeros(len(y))
    for train, test in GroupKFold(5).split(x, y, meta.run_id):
        model = fit(x[train], y[train], c)
        pred[test] = model.predict_proba(x[test])[:, 1]
    score = aggregate(meta, pred)
    return auc(score.label, score.score)


def select(meta, activation, cfg, out=None):
    records = []

    def scan(layers, stage):
        for layer in layers:
            x = activation[:, layer].astype(np.float32)
            for c in cfg['probe']['c_grid']:
                records.append(dict(layer=int(layer), C=c, stage=stage, development_auc=cv(meta, x, c)))
    scan(cfg['probe']['coarse_layers'], 'coarse')
    coarse = sorted(records, key=lambda r: (-r['development_auc'], r['layer'], r['C']))[0]
    scan(sorted(set(range(max(0, coarse['layer'] - 3), min(63, coarse['layer'] + 3) + 1))), 'fine')
    best = sorted(records, key=lambda r: (-r['development_auc'], r['layer'], r['C']))[0]
    if out is not None:
        pd.DataFrame(records).to_csv(out, index=False)
    return best


def metadata_model(meta, include_recent=False):
    cols = NUMERIC + (['last_four_correct'] if include_recent else [])
    pre = ColumnTransformer([('id', OneHotEncoder(handle_unknown='ignore'), ['advisor_id']), ('numeric', StandardScaler(), cols)])
    model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000, solver='liblinear'))
    model.fit(meta, meta.label)
    return model


def evaluate_scores(meta, pred, rng, repeats=1000):
    score = aggregate(meta, pred)
    y = score.label.to_numpy()
    p = score.score.to_numpy()
    groups = score.run_id.to_numpy()
    runs = np.unique(groups)
    idx = [np.flatnonzero(groups == r) for r in runs]
    estimate = auc(y, p)
    boot = []
    exceed = 0
    for _ in range(2000):
        selected = np.concatenate([idx[k] for k in rng.integers(len(idx), size=len(idx))])
        boot.append(auc(y[selected], p[selected]))
    for _ in range(repeats):
        perm = y.copy()
        for ids in idx:
            perm[ids] = rng.permutation(perm[ids])
        exceed += auc(perm, p) >= estimate
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return dict(auc=estimate, ci_low=lo, ci_high=hi, permutation_p=(exceed + 1) / (repeats + 1), runs=len(runs), endpoints=len(score), within_run_auc=float(np.nanmean([auc(y[ix], p[ix]) for ix in idx])))
