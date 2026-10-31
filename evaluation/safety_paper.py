"""Fixed-alpha, current-probability ridge probes for the paper safety claim.

Consumes frozen document partitions and route/probability extraction arrays;
does not split documents, extract routes, tune classifiers, or train the LM.
Sequence features use an equal mean over later windows in each exact-document
identity group, in canonical
group/fragment/position order, stored as float32. Window zero is excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def ridge_statistics(x, y, weights, device='cpu'):
    """Original float64 development-only sufficient statistics."""
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    y = 2 * torch.as_tensor(y, dtype=torch.float64, device=device) - 1
    if not torch.isfinite(x).all() or not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError('Finite features and positive finite training weights required')
    mean = (x * w[:, None]).sum(0) / w.sum()
    centered = x - mean
    variance = (centered.square() * w[:, None]).sum(0) / w.sum()
    scale = variance.sqrt()
    scale[scale == 0] = 1
    z = centered / scale
    target_mean = (y * w).sum() / w.sum()
    gram = z.T @ (z * w[:, None])
    rhs = z.T @ ((y - target_mean) * w)
    return dict(gram=gram.cpu().numpy(), rhs=rhs.cpu().numpy(), mean=mean.cpu().numpy(),
                scale=scale.cpu().numpy(), target_mean=float(target_mean.cpu()))


def solve_subset(stats, indices, device='cpu'):
    """Original alpha=1 solve, collapsed back to raw probability coordinates."""
    indices = np.asarray(indices, dtype=np.int64)
    gram = torch.tensor(stats['gram'][np.ix_(indices, indices)], dtype=torch.float64, device=device)
    gram.diagonal().add_(1.)
    rhs = torch.as_tensor(stats['rhs'][indices], dtype=torch.float64, device=device)
    beta = torch.linalg.solve(gram, rhs).cpu().numpy()
    coefficient = beta / stats['scale'][indices]
    intercept = float(stats['target_mean'] - stats['mean'][indices] @ coefficient)
    return coefficient, intercept, beta


def features(data):
    """Return window and original identity-group mean features and labels."""
    n = len(data['routes'])
    for key in ('groups', 'fragments', 'positions', 'labels', 'offsets', 'folds'):
        value = np.asarray(data[key])
        if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f'Expected aligned integer {key}')
        if key != 'folds' and np.any(value < 0):
            raise ValueError(f'Expected nonnegative {key}')
    if not np.isin(data['labels'], [0, 1]).all():
        raise ValueError('Binary labels required (unsafe=1)')
    routes = np.asarray(data['routes'])
    if routes.shape != (n, 22) or not np.issubdtype(routes.dtype, np.integer) or np.any((routes < 0) | (routes >= 33)):
        raise ValueError('Expected E33/R22 route IDs')
    order = np.lexsort((data['positions'], data['fragments']))
    same = data['fragments'][order[1:]] == data['fragments'][order[:-1]]
    if np.any(same & (data['positions'][order[1:]] == data['positions'][order[:-1]])):
        raise ValueError('Duplicate fragment/window positions')
    if np.any(same & (data['groups'][order[1:]] != data['groups'][order[:-1]])):
        raise ValueError('Fragment crosses document groups')
    order = np.argsort(data['groups'], kind='stable')
    same = np.diff(data['groups'][order]) == 0
    for key in ('labels', 'folds'):
        if np.any(same & (np.diff(data[key][order]) != 0)):
            raise ValueError(f'Document group has inconsistent {key}')
    later = data['positions'] > 0
    order = np.lexsort((data['positions'], data['fragments']))
    predecessor = np.full(n, -1, dtype=np.int64)
    current_rows, prior_rows = order[1:], order[:-1]
    consecutive = ((data['fragments'][current_rows] == data['fragments'][prior_rows]) &
                   (data['positions'][current_rows] == data['positions'][prior_rows] + 1))
    predecessor[current_rows[consecutive]] = prior_rows[consecutive]
    if np.any(predecessor[later] < 0):
        raise ValueError('Every later window requires its consecutive predecessor')
    probabilities = np.asarray(data['router_probabilities'], dtype=np.float32)
    if probabilities.shape != (n, 22, 33) or not np.isfinite(probabilities).all():
        raise ValueError('Expected finite row-aligned [N,22,33] router probabilities')
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(-1), 1, atol=2e-6, rtol=2e-6):
        raise ValueError('Router probabilities must be nonnegative and normalized')
    current = probabilities.reshape(n, 726)
    groups, inverse, counts = np.unique(data['groups'][later], return_inverse=True, return_counts=True)
    if not len(groups):
        raise ValueError('No document groups with later windows')
    rows = np.flatnonzero(later)
    order = np.lexsort((data['positions'][rows], data['fragments'][rows], data['groups'][rows]))
    rows, inverse = rows[order], inverse[order]
    means = np.empty((len(groups), 726), dtype=np.float32)
    for column in range(726):
        means[:, column] = np.bincount(inverse, weights=current[rows, column], minlength=len(groups)) / counts
    first = rows[np.r_[True, np.diff(inverse) != 0]]
    return {
        'window': (current[later], data['labels'][later]),
        'sequence': (means, data['labels'][first]),
    }, {
        'all_windows': n, 'later_windows': int(later.sum()),
        'all_document_groups': len(np.unique(data['groups'])),
        'scored_document_groups': len(groups),
        'excluded_groups_without_later_windows': len(np.unique(data['groups'])) - len(groups),
    }


def fit(development):
    """Fit the original alpha=1 standardized ridge objective on development."""
    if not np.array_equal(np.unique(development['folds']), np.arange(5)):
        raise ValueError('Expected the frozen five development folds')
    inputs, counts = features(development)
    models = {}
    for unit, (x, labels) in inputs.items():
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f'Both development classes must contribute {unit} features')
        stats = ridge_statistics(x, labels, np.ones(len(labels)), device='cpu')
        coefficient, intercept, _ = solve_subset(stats, np.arange(726), device='cpu')
        models[unit] = (coefficient, intercept)
    return models, counts


def score(models, development_groups, heldout):
    """Evaluate frozen probes once; never fit transforms on held-out rows."""
    if np.intersect1d(development_groups, heldout['groups']).size:
        raise ValueError('Development and held-out document groups overlap')
    if not np.all(np.asarray(heldout['folds']) == -1):
        raise ValueError('Held-out rows must belong to the frozen reserved partition (fold=-1)')
    inputs, counts = features(heldout)
    metrics = {}
    for unit, (x, labels) in inputs.items():
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f'Both held-out classes must contribute {unit} features')
        coefficient, intercept = models[unit]
        values = np.einsum('ij,j->i', np.ascontiguousarray(x, dtype=np.float64), coefficient, optimize=False) + intercept
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite held-out margins')
        metrics[unit] = {'auroc': float(roc_auc_score(labels, values)), 'samples': len(labels)}
    return metrics, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-routes', type=Path, required=True)
    parser.add_argument('--heldout-routes', type=Path, required=True,
                        help='NPZ with router_probabilities and frozen reserved-document metadata')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(2)
    with np.load(args.development_routes, allow_pickle=False) as saved:
        development = {key: saved[key] for key in saved.files}
    models, development_counts = fit(development)
    # Fit both probes before opening held-out feature content.
    with np.load(args.heldout_routes, allow_pickle=False) as saved:
        heldout = {key: saved[key] for key in saved.files}
    metrics, heldout_counts = score(models, np.unique(development['groups']), heldout)
    sources = {}
    for path in (args.development_routes, args.heldout_routes):
        with path.open('rb') as stream:
            sources[str(path.resolve())] = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {
        'status': 'complete', 'alpha': 1.0, 'selection': 'fixed original alpha; no held-out tuning',
        'scaling': 'development-only float64 weighted standardization with unit weights',
        'features': '726 current router probabilities; later windows only; equal-window sequence means',
        'sequence_unit': 'original exact-document identity group, combining fragments and duplicates',
        'development': development_counts, 'heldout': heldout_counts,
        'metrics': metrics, 'source_sha256': sources,
        'code_sha256': {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'archived_window_evidence': {
            'auroc': 0.9196043828001167,
            'source': 'original evaluation/results/safety_routing_formal/heldout_final/results.json',
            'source_sha256': '1510ecb00f2bb4f982231e5c769501a55a23b81662fc5b52849f29d878c2b840',
            'meaning': 'historical current-only ridge result; separate from this evaluation',
        },
        'archived_sequence_heldout_evidence': None,
        'sequence_evidence': 'New held-out measurement; no archived sequence-heldout result for direct numerical parity',
    }
    args.output.mkdir(parents=True)
    np.savez(args.output / 'probes.npz', **{
        f'{unit}_{key}': value
        for unit, pair in models.items() for key, value in zip(('coefficient', 'intercept'), pair)
    })
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
