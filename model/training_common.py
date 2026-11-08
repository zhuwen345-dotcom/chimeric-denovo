"""Shared data, provenance and evaluation helpers for the resumable trainer."""
import hashlib
import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import torch

from model import ChimeraSequenceModel as SeparatorModel
from separator_base import sequence_with_modifications


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


class PreparedData:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        self.splits = json.loads((self.path / 'splits.json').read_text())
        self.records = [json.loads(line) for line in (self.path / 'records.jsonl').open()]
        self.peaks = np.load(self.path / 'peaks.npy', mmap_mode='r')
        self.lengths = np.load(self.path / 'lengths.npy', mmap_mode='r')
        self.precursors = np.load(self.path / 'precursors.npy', mmap_mode='r')
        # Check all prepared inputs on startup; training never edits this directory.
        for name, expected in self.manifest['files_sha256'].items():
            if digest(self.path / name) != expected:
                raise ValueError('Prepared data changed: ' + name)

    def batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        length = int(self.lengths[indices].max())
        spectra = torch.from_numpy(np.array(self.peaks[indices, :length], copy=True)).to(device)
        precursors = torch.from_numpy(np.array(self.precursors[indices], copy=True)).to(device)
        labels = [self.records[int(i)]['model_labels'] for i in indices]
        return spectra, precursors, labels


def per_spectrum_loss(model, spectra, precursors, labels):
    memory, mask = model.memories(spectra)
    original = model.loss_for_order(labels, precursors, memory, mask, False)
    if any(len(peptides) == 2 for peptides in labels):
        swapped = model.loss_for_order(labels, precursors, memory, mask, True)
        is_double = torch.tensor([len(peptides) == 2 for peptides in labels], device=original.device)
        return torch.where(is_double, torch.minimum(original, swapped), original)
    return original


class LossMeter:
    def __init__(self, state=None):
        self.sums = {'all': 0.0, 'single': 0.0, 'double': 0.0}
        self.counts = {'all': 0, 'single': 0, 'double': 0}
        if state:
            self.sums.update(state['sums'])
            self.counts.update(state['counts'])

    def add(self, losses, labels):
        values = losses.detach().cpu().tolist()
        for value, peptides in zip(values, labels):
            kind = 'single' if len(peptides) == 1 else 'double'
            for key in ('all', kind):
                self.sums[key] += value
                self.counts[key] += 1

    def means(self):
        return {key: self.sums[key] / self.counts[key] if self.counts[key] else None
                for key in self.sums}

    def state(self):
        return {'sums': self.sums.copy(), 'counts': self.counts.copy()}


@torch.no_grad()
def evaluate_loss(model, data, indices, device, batch_size, stop_requested=None):
    model.eval()
    meter = LossMeter()
    for start in range(0, len(indices), batch_size):
        if stop_requested and stop_requested():
            return None
        spectra, precursors, labels = data.batch(indices[start:start + batch_size], device)
        losses = per_spectrum_loss(model, spectra, precursors, labels)
        if not torch.isfinite(losses).all():
            raise RuntimeError('Non-finite validation/test loss.')
        meter.add(losses, labels)
    return meter.means()


@torch.no_grad()
def evaluate_sequences(model, data, indices, device, batch_size=8, stop_requested=None):
    """Free generation: references are compared only AFTER model.generate()."""
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        if stop_requested and stop_requested():
            return None
        group = indices[start:start + batch_size]
        spectra, precursors, labels = data.batch(group, device)
        predictions = model.generate(spectra, precursors)
        for index, reference, prediction in zip(group, labels, predictions):
            record = data.records[index]
            expected, actual = set(reference), set(prediction['peptides'])
            rows.append({'raw_file': record['raw_file'], 'scan': record['scan'],
                         'reference_pseudolabels': reference,
                         'prediction': prediction['peptides'],
                         'matched_peptides': len(expected & actual),
                         'complete_set_match': expected == actual,
                         'ended_with_eos': prediction['ended_with_eos'],
                         'truncated': prediction['truncated'],
                         'chimeric_probability': prediction.get('chimeric_probability'),
                         'predicted_count': prediction.get('predicted_count'),
                         'routing': prediction.get('routing'),
                         'count_constraint_satisfied': prediction.get('count_constraint_satisfied')})
    summary = {'spectra': len(rows)}
    for count, name in ((1, 'single'), (2, 'double')):
        selected = [row for row in rows if len(row['reference_pseudolabels']) == count]
        n = len(selected)
        summary[name] = {
            'spectra': n,
            'complete_set_match_rate': sum(row['complete_set_match'] for row in selected) / n if n else None,
            'at_least_one_match_rate': sum(row['matched_peptides'] >= 1 for row in selected) / n if n else None,
            'two_output_rate': sum(len(row['prediction']) == 2 for row in selected) / n if n else None,
            'truncation_rate': sum(row['truncated'] for row in selected) / n if n else None,
        }
    return summary, rows
