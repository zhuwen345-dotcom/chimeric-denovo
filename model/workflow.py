"""Split checks, initialization provenance, and validation-only metrics."""
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from training_common import LossMeter, digest

FORMAT = 'casanovo_dda_classifier_v4'


def peptide_keys(record):
    return set(record.get('split_peptide_keys') or
               [re.sub('[^A-Z]', '', p).replace('I', 'L') for p in record['model_labels']])


def exposure(records, indices):
    peptides, spectra, raw_files = set(), set(), set()
    for i in indices:
        peptides.update(peptide_keys(records[i]))
        spectra.add(json.dumps([records[i]['raw_file'], str(records[i]['scan'])]))
        raw_files.add(records[i]['raw_file'])
    return {'peptides': sorted(peptides), 'spectra': sorted(spectra), 'raw_files': sorted(raw_files)}


def validate_splits(data):
    """Read labels only to check overlap; never compute test performance here."""
    n = len(data.records)
    if len(data.peaks) != n or len(data.lengths) != n or len(data.precursors) != n:
        raise ValueError('Prepared arrays and records have different lengths.')
    if data.peaks.ndim != 3 or data.peaks.shape[2] != 2 or data.precursors.shape != (n, 3):
        raise ValueError('Expected peaks (N, P, 2) and instrument precursors (N, 3).')
    policy = data.manifest.get('split_policy', 'peptide_disjoint')
    if policy not in ('peptide_disjoint', 'raw_disjoint'):
        raise ValueError('Unknown explicit split policy: ' + policy)
    reports, sets = {}, {}
    for split in ('train', 'validation', 'test'):
        indices = data.splits[split]
        if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= n:
            raise ValueError('Empty/duplicate/out-of-range split: ' + split)
        for i in indices:
            labels = data.records[i]['model_labels']
            if len(labels) not in (1, 2) or len(set(labels)) != len(labels):
                raise ValueError('Expected one or two distinct sequence labels.')
            if not 0 < data.lengths[i] <= data.peaks.shape[1]:
                raise ValueError('Invalid peak length.')
        state = exposure(data.records, indices)
        if len(state['spectra']) != len(indices):
            raise ValueError('Duplicate RAW/scan records within split: ' + split)
        sets[split] = {k: set(v) for k, v in state.items()}
        singles = sum(len(data.records[i]['model_labels']) == 1 for i in indices)
        doubles = len(indices) - singles
        if split != 'test' and not (singles and doubles):
            raise ValueError('Train and validation each need both label counts.')
        reports[split] = {'spectra': len(indices), 'single': singles, 'double': doubles,
                          'raw_files': sorted({data.records[i]['raw_file'] for i in indices})}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(data.splits[a]) & set(data.splits[b]) or sets[a]['spectra'] & sets[b]['spectra']:
            raise ValueError(f'Overlapping spectra: {a}/{b}.')
        key = 'raw_files' if policy == 'raw_disjoint' else 'peptides'
        if sets[a][key] & sets[b][key]:
            raise ValueError(f'Overlapping {key} for explicit {policy} policy: {a}/{b}.')
    print('Split policy: ' + policy + '; peptide overlaps: ' + json.dumps({a+'_vs_'+b:len(sets[a]['peptides'] & sets[b]['peptides'])
          for a,b in (('train','validation'),('train','test'),('validation','test'))}), flush=True)
    return reports


def initialize(model, path, base_hash, data):
    """A new task inherits weights; optimizer/RNG history starts anew.

    For historical runs, read the preserved parent data/config to audit training
    exposure. v4 checkpoints carry this exposure forward across initialization.
    """
    inherited = {'peptides': [], 'spectra': [], 'raw_files': []}
    if path:
        path = Path(path).resolve()
        parent = torch.load(path, map_location='cpu', weights_only=False)
        accepted = ('casanovo_dda_resumable_v1', 'casanovo_dda_continuation_v2',
                    'casanovo_dda_encoder_stage_v3', FORMAT)
        if parent.get('format') not in accepted:
            raise ValueError('Initialize with a trusted full project checkpoint, not a TL head-only checkpoint.')
        if parent['immutable_config']['base_sha256'] != base_hash:
            raise ValueError('Initializer and official base weights differ.')
        missing, unexpected = model.load_state_dict(parent['state_dict'], strict=False)
        expected = set() if parent['format'] == FORMAT else {
            'classification_head.0.weight', 'classification_head.0.bias',
            'classification_head.2.weight', 'classification_head.2.bias'}
        if set(missing) != expected or unexpected:
            raise ValueError(f'Incompatible initializer: missing={missing}, unexpected={unexpected}')
        inherited = parent.get('training_exposure')
        if inherited is None:
            config_path = path.parent / 'config.json'
            config = json.loads(config_path.read_text())
            previous_data = Path(config['arguments']['data'])
            if digest(previous_data / 'manifest.json') != parent['immutable_config']['data_manifest_sha256']:
                raise ValueError('Cannot audit the initializer: historical prepared data changed.')
            records = [json.loads(s) for s in (previous_data / 'records.jsonl').read_text().splitlines()]
            splits = json.loads((previous_data / 'splits.json').read_text())
            inherited = exposure(records, splits['train'])
            if parent['format'] != 'casanovo_dda_resumable_v1':
                # v2/v3 start from run01 on this same historical split.
                if parent['immutable_config'].get('initializer_epoch') != 10:
                    raise ValueError('Historical initialization chain needs a manual exposure audit.')
        del parent
    if 'raw_files' not in inherited:
        inherited['raw_files'] = sorted({json.loads(value)[0] for value in inherited['spectra']})
    policy = data.manifest.get('split_policy', 'peptide_disjoint')
    for split in ('validation', 'test'):
        heldout = exposure(data.records, data.splits[split])
        checked = ('spectra', 'raw_files') if policy == 'raw_disjoint' else ('spectra', 'peptides')
        if any(set(inherited[k]) & set(heldout[k]) for k in checked):
            raise ValueError('New ' + split + ' overlaps the initializer training exposure. Rebuild the split.')
    current = exposure(data.records, data.splits['train'])
    return {k: sorted(set(inherited[k]) | set(current[k])) for k in current}


def classification_metrics(truth, probabilities, threshold):
    """AUROC using average ranks for ties; no sklearn dependency."""
    y, p = np.asarray(truth, dtype=np.int64), np.asarray(probabilities, dtype=float)
    if not len(y) or not np.isfinite(p).all():
        raise ValueError('Empty or non-finite classifier predictions.')
    predicted = p >= threshold
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    positives, negatives = int(y.sum()), int((y == 0).sum())
    auc = None
    if positives and negatives:
        order = np.argsort(p, kind='stable')
        ordered = p[order]
        ranks = np.empty(len(p), dtype=float)
        start = 0
        while start < len(p):
            end = start + 1
            while end < len(p) and ordered[end] == ordered[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    return {'spectra': len(y), 'auroc': auc, 'threshold': threshold,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'double_recall': recall, 'single_specificity': specificity,
            'single_predicted_double_rate': fp / negatives if negatives else None,
            'precision_double': tp / (tp + fp) if tp + fp else None,
            'balanced_accuracy': (recall + specificity) / 2 if positives and negatives else None,
            'brier_score': float(np.mean((p - y) ** 2))}


@torch.no_grad()
def evaluate_components(model, data, indices, device, batch_size, task,
                        stop_requested=None, progress_every=512):
    model.eval()
    seq_meter, cls_meter = LossMeter(), LossMeter()
    truth, probabilities, rows = [], [], []
    started, reported = time.perf_counter(), 0
    print(f'Validation/evaluation: 0/{len(indices)} (0.0%)', flush=True)
    for start in range(0, len(indices), batch_size):
        if stop_requested and stop_requested():
            return None
        group = indices[start:start + batch_size]
        spectra, precursors, labels = data.batch(group, device)
        seq, cls, logits = model.loss_components(spectra, precursors, labels, task)
        if not torch.isfinite(cls).all() or (seq is not None and not torch.isfinite(seq).all()):
            raise RuntimeError('Non-finite evaluation loss.')
        cls_meter.add(cls, labels)
        if seq is not None:
            seq_meter.add(seq, labels)
        probs = logits.sigmoid().cpu().tolist()
        targets = [int(len(p) == 2) for p in labels]
        truth.extend(targets)
        probabilities.extend(probs)
        for i, y, probability in zip(group, targets, probs):
            r = data.records[i]
            rows.append({'raw_file': r['raw_file'], 'scan': r['scan'], 'reference_count': y + 1,
                         'chimeric_probability': probability,
                         'predicted_count': 1 + int(probability >= model.threshold)})
        done = start + len(group)
        if done - reported >= progress_every or done == len(indices):
            print(f'Validation/evaluation: {done}/{len(indices)} ({100*done/len(indices):.1f}%) | '
                  f'{(time.perf_counter()-started)/60:.1f} min', flush=True)
            reported = done
    return {'sequence_ce': seq_meter.means(), 'classification_bce': cls_meter.means(),
            'classification': classification_metrics(truth, probabilities, model.threshold)}, rows
