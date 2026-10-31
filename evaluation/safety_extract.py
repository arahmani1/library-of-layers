"""Portable paper safety router-probability extraction on the frozen split.

Preserves the original eight-logical-rank, batch-16 packed-row geometry while
running on one selected GPU. Resume reuses only shards with identical source,
checkpoint, partition, code, and geometry identities. No probe fitting occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


DOCUMENTS_SHA256 = '53ad42f92da369eee02f3775f68512ff4ca8b23120a3dd6afa906855da695eb4'
CORPUS_SHA256 = {
    'safe': '7723a0c5c1958baf49ebca892dd8ec54d06ece5238f4e721b9afc64c5fa9729f',
    'unsafe': '0f1a16dc88b1fecbc63edf6566c7e97becc178256311dfcf952cb030e30ab51c',
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RouterConfidence:
    """Original observation-only hook at each depth-zero window's first token."""
    def __init__(self, model):
        self.model, self.handles, self.values = model, [], []
        self.first = None

    def clear(self):
        self.values.clear()
        self.first = None

    def _capture(self, module, args, kwargs, output):
        del module, args
        depth = int(kwargs['depth'])
        if depth != len(self.values):
            raise ValueError('Expected one ordered router invocation per depth')
        ids = kwargs['global_window_id'].reshape(-1)
        if self.first is None:
            self.first = torch.full((int(ids.max()) + 1,), ids.numel(), device=ids.device, dtype=torch.long)
            self.first.scatter_reduce_(0, ids, torch.arange(ids.numel(), device=ids.device), reduce='amin')
            self.first.clamp_(max=ids.numel() - 1)
        self.values.append(output.detach().float()[ids[self.first]])
        # Returning None preserves the model's original forward result.

    def result(self):
        if len(self.values) != self.model.recurrent_depth:
            raise ValueError('Incomplete confidence depth coverage')
        return self.first, torch.stack(self.values, dim=1)

    def __enter__(self):
        modules = {id(self.model._router_for_depth(d)): self.model._router_for_depth(d)
                   for d in range(self.model.recurrent_depth)}
        self.handles = [module.register_forward_hook(self._capture, with_kwargs=True) for module in modules.values()]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def partition_batch(data, starts, ends, partition, row_ids, block_size, eot, selected_partition):
    """Original packed-document geometry; read only the selected token spans."""
    shape = (len(row_ids), block_size)
    inputs = np.full(shape, eot, dtype=np.int64)
    doc_ids = np.zeros(shape, dtype=np.int64)
    positions = np.zeros(shape, dtype=np.int64)
    owners = np.full(shape, -1, dtype=np.int64)
    for row, row_id in enumerate(row_ids):
        begin, finish = int(row_id) * block_size, (int(row_id) + 1) * block_size
        first = max(0, int(np.searchsorted(starts, begin, side='right')) - 1)
        last = int(np.searchsorted(starts, finish, side='left'))
        local_doc = 1 if starts[first] == begin else 0
        written = 0
        for document in range(first, last):
            a, b = max(begin, int(starts[document])), min(finish, int(ends[document]))
            if a >= b:
                continue
            lo, hi = a - begin, b - begin
            doc_ids[row, lo:hi] = local_doc
            positions[row, lo:hi] = np.arange(hi - lo)
            if partition[document] == selected_partition:
                inputs[row, lo:hi] = data[a:b]
                owners[row, lo:hi] = document
            local_doc += 1
            written = hi
        if written < block_size:
            doc_ids[row, written:] = local_doc
            positions[row, written:] = np.arange(block_size - written)
    return torch.from_numpy(inputs), torch.from_numpy(doc_ids), torch.from_numpy(positions), owners


def validate_windows(arrays, docs, cfg, covered_by_split, selected_partition):
    """Original exact packed-window and full-token coverage checks."""
    width, window = cfg['block_size'], cfg['window']
    n = len(arrays['labels'])
    for key in ('labels', 'groups', 'fragments', 'positions', 'offsets', 'folds', 'n_tokens'):
        if np.asarray(arrays[key]).shape != (n,):
            raise ValueError('Misaligned window metadata')
    routes, p = arrays['routes'], arrays['router_probabilities']
    if routes.shape != (n, cfg['recurrent_depth']) or not np.issubdtype(routes.dtype, np.integer) or np.any((routes < 0) | (routes >= cfg['n_experts'])):
        raise ValueError('Invalid extracted route IDs')
    if p.shape != (n, cfg['recurrent_depth'], cfg['n_experts']) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=2e-6):
        raise ValueError('Invalid router probabilities')
    if not np.isin(arrays['labels'], [0, 1]).all() or (np.any(arrays['folds'] < 0) if selected_partition == 0 else np.any(arrays['folds'] != -1)):
        raise ValueError('Extracted records cross frozen partitions')
    if len(np.unique(np.column_stack((arrays['labels'], arrays['offsets'])), axis=0)) != n:
        raise ValueError('Duplicate route anchors')
    for label, split in enumerate(('safe', 'unsafe')):
        select = np.flatnonzero(docs['labels'] == label)
        total = (int(docs['ends'][select][-1]) + width - 1) // width
        if not np.array_equal(np.sort(covered_by_split[split]), np.arange(total)):
            raise ValueError(f'Incomplete or duplicate row coverage for {split}')
        mask = arrays['labels'] == label
        offsets = arrays['offsets'][mask]
        local = np.searchsorted(docs['starts'][select], offsets, side='right') - 1
        if np.any(local < 0):
            raise ValueError('Anchor precedes document inventory')
        owners = select[local]
        if np.any(docs['partition'][owners] != selected_partition):
            raise ValueError('Wrong document partition')
        for key in ('groups', 'folds'):
            if not np.array_equal(arrays[key][mask], docs[key][owners]):
                raise ValueError(f'Document {key} ownership mismatch')
        row_start = offsets // width * width
        fragment_start = np.maximum(row_start, docs['starts'][owners])
        relative = offsets - fragment_start
        if np.any(offsets >= docs['ends'][owners]) or np.any(relative % window) or not np.array_equal(arrays['positions'][mask], relative // window):
            raise ValueError('Invalid causal window position')
        if not np.array_equal(arrays['fragments'][mask], label * 10**12 + fragment_start):
            raise ValueError('Fragment boundary changed')
        expected_tokens = np.minimum(window, np.minimum(docs['ends'][owners] - offsets, row_start + width - offsets))
        if np.any(expected_tokens <= 0) or not np.array_equal(arrays['n_tokens'][mask], expected_tokens):
            raise ValueError('Window token coverage mismatch')
        observed = np.bincount(owners, weights=arrays['n_tokens'][mask], minlength=len(docs['labels']))
        eligible = (docs['labels'] == label) & (docs['partition'] == selected_partition)
        if not np.array_equal(observed, np.where(eligible, docs['ends'] - docs['starts'], 0)):
            raise ValueError('Document token coverage mismatch')
    order = np.lexsort((arrays['positions'], arrays['fragments']))
    same = arrays['fragments'][order[1:]] == arrays['fragments'][order[:-1]]
    if np.any(arrays['positions'][order][np.r_[True, ~same]] != 0) or np.any(same & (np.diff(arrays['positions'][order]) != 1)):
        raise ValueError('Missing first window or causal predecessor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--documents', type=Path, required=True, help='Original checksum-pinned documents.npz partition inventory')
    parser.add_argument('--partition', choices=('development', 'heldout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    import router
    from evaluation.model_loading import build_evaluation_model
    partition = 0 if args.partition == 'development' else -1
    if sha256(args.documents) != DOCUMENTS_SHA256:
        raise ValueError('Use the original frozen document partitions')
    manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    if manifest['dataset'] != {'id': 'locuslab/fineweb_annotated', 'revision': 'd96861caa3a5c926f9f228871fb6e33871e9ea30'}:
        raise ValueError('Unexpected safety dataset revision')
    if manifest['token_dtype'] != 'uint32' or manifest['tokenizer'] != {
        'id': 'allenai/Olmo-3-1025-7B', 'revision': 'a81bae42db3975be1671e27b9c9a56da1a9f980f',
        'vocab_size': 100278, 'model_vocab_size': 100352, 'eot_token_id': 100257,
    }:
        raise ValueError('Unexpected safety tokenizer or token format')
    corpus = {}
    for split in ('safe', 'unsafe'):
        path = args.data_dir / manifest['splits'][split]['tokens']
        digest = sha256(path)
        if digest != CORPUS_SHA256[split] or manifest['splits'][split]['tokens_count'] != 25_000_000:
            raise ValueError(f'Use the pinned {split} validation corpus')
        corpus[split] = {'path': str(path.resolve()), 'sha256': digest}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = checkpoint['config']
    expected = dict(n_head=16, n_embd=1024,
                    attention_strategy='shifted_window',
                    normalization='affine_rmsnorm', qk_norm=True,
                    n_experts=33, recurrent_depth=22, n_prelude_layers=1, n_coda_layers=1,
                    weight_tie_router=True, block_size=1024, window=128, window_stride=64,
                    first_window_routing='learned_token_depth',
                    eot_token_id=100257, vocab_size=100352)
    if (cfg['model'].get('arch') != 'lol_dynamic_window'
            or any(cfg['model'].get(key) != value for key, value in expected.items())):
        raise ValueError('Checkpoint does not match the paper safety architecture')
    protocol = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                    documents_sha256=DOCUMENTS_SHA256, partition=args.partition, config=cfg,
                    manifest=manifest, corpus=corpus, logical_world_size=8, batch_size=16,
                    source_checkout=str(source),
                    code_sha256={str(path.resolve()): sha256(path) for path in
                                 [Path(__file__), source / 'router.py', source / 'evaluation/model_loading.py', *source.joinpath('model').glob('*.py')]})
    protocol_text = json.dumps(protocol, sort_keys=True, indent=2) + '\n'
    protocol_hash = hashlib.sha256(protocol_text.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and protocol_path.read_text() != protocol_text:
        raise ValueError('Extraction identity changed; use a fresh output directory')
    protocol_path.write_text(protocol_text)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Production extraction requires a selected CUDA GPU')
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    model, _ = build_evaluation_model(checkpoint['model_state_dict'], cfg)
    model.to(device).eval()
    del checkpoint
    with np.load(args.documents, allow_pickle=False) as saved:
        docs = {key: saved[key] for key in saved.files}
    shards = args.output / 'shards'
    shards.mkdir(exist_ok=True)
    pieces, covered = [], {'safe': [], 'unsafe': []}
    with RouterConfidence(model) as capture:
        for label, split in enumerate(('safe', 'unsafe')):
            selected = np.flatnonzero(docs['labels'] == label)
            starts, ends = docs['starts'][selected], docs['ends'][selected]
            data = np.memmap(corpus[split]['path'], dtype='uint32', mode='r')
            width = cfg['model']['block_size']
            for rank in range(8):
                rank_rows = np.arange((len(data) + width - 1) // width)[rank::8]
                for batch, begin in enumerate(range(0, len(rank_rows), 16)):
                    rows = rank_rows[begin:begin + 16]
                    path = shards / f'{split}-rank{rank:02d}-batch{batch:06d}.npz'
                    if not path.exists():
                        x, doc, pos, owners = partition_batch(data, starts, ends, docs['partition'][selected], rows,
                                                             width, cfg['model']['eot_token_id'], partition)
                        useful = (owners >= 0).any(1)
                        active_rows, owners = rows[useful], owners[useful]
                        x, doc, pos = x[useful].to(device), doc[useful].to(device), pos[useful].to(device)
                        fields = {key: np.empty(0, dtype=np.int64) for key in
                                  ('labels', 'groups', 'fragments', 'positions', 'offsets', 'folds', 'n_tokens')}
                        fields['routes'] = np.empty((0, 22), dtype=np.int16)
                        fields['router_probabilities'] = np.empty((0, 22, 33), dtype=np.float32)
                        if len(active_rows):
                            capture.clear()
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                                out = router.validation_forward(model, x, doc, pos, None, return_router_trace=True)
                            first, probabilities = capture.result()
                            stats = out[3]
                            routes, tokens, _, valid = router.window_routing_rows(stats['routing_token_paths'], stats['routing_depth0_window_ids'], doc)
                            first_np = first.cpu().numpy()
                            owner = owners.reshape(-1)[first_np]
                            keep = (owner >= 0) & valid.cpu().numpy()
                            global_doc = selected[owner[keep]]
                            local = pos.reshape(-1)[first].cpu().numpy()[keep]
                            absolute = active_rows[first_np[keep] // width] * width + first_np[keep] % width
                            if np.any(docs['partition'][global_doc] != partition):
                                raise ValueError('Wrong partition reached extraction output')
                            fields = dict(routes=routes.cpu().numpy()[keep].astype(np.int16),
                                          router_probabilities=probabilities.cpu().numpy()[keep],
                                          labels=np.full(keep.sum(), label, dtype=np.int8), groups=docs['groups'][global_doc],
                                          fragments=label * 10**12 + absolute - local, positions=local // cfg['model']['window'],
                                          offsets=absolute, folds=docs['folds'][global_doc], n_tokens=tokens.cpu().numpy()[keep])
                            del out, stats, probabilities
                        temporary = path.with_suffix('.tmp.npz')
                        np.savez_compressed(temporary, **fields, covered_rows=rows, protocol_sha256=np.array(protocol_hash))
                        temporary.replace(path)
                    with np.load(path, allow_pickle=False) as saved:
                        if str(saved['protocol_sha256']) != protocol_hash or not np.array_equal(saved['covered_rows'], rows):
                            raise ValueError('Resumed shard has a different identity or geometry')
                        pieces.append({key: saved[key] for key in saved.files if key not in ('covered_rows', 'protocol_sha256')})
                        covered[split].extend(rows.tolist())
                    if batch % 50 == 0:
                        print(args.partition, split, 'logical rank', rank, 'batch', batch, flush=True)
    arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    order = np.lexsort((arrays['offsets'], arrays['labels']))
    arrays = {key: value[order] for key, value in arrays.items()}
    validate_windows(arrays, docs, cfg['model'], covered, partition)
    temporary = args.output / 'windows.tmp.npz'
    np.savez_compressed(temporary, **arrays)
    temporary.replace(args.output / 'windows.npz')
    (args.output / 'coverage.json').write_text(json.dumps(dict(status='complete', protocol_sha256=protocol_hash,
        windows_sha256=sha256(args.output / 'windows.npz'), all_source_rows_covered=True,
        all_partition_tokens_covered=True, first_windows_retained=True, n_tokens=int(arrays['n_tokens'].sum())), indent=2) + '\n')


if __name__ == '__main__':
    main()
