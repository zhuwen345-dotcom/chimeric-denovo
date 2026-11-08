"""Engineering pilot: one mixed spectrum -> one/two peptide sequences.

Reuses Casanovo 4.0.0 and adds one separator token, with order-invariant
teacher-forcing loss for two labels. Greedy decoding deliberately does not use
the original single-peptide precursor-mass filter. No held-out accuracy claim.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Each architecture carries its own Casanovo and encoder/decoder sources.
import sys
for _bundle_parent in Path(__file__).resolve().parents:
    _bundle = _bundle_parent / 'backbone'
    if (_bundle / 'casanovo').is_dir() and (_bundle / 'depthcharge').is_dir():
        sys.path.insert(0, str(_bundle))
        break

from casanovo.denovo.model import Spec2Pep
from spectrum_utils.spectrum import MsmsSpectrum


def sequence_with_modifications(label):
    sequence = label['sequence'].replace('I', 'L')
    tokens = ['C+57.021' if aa == 'C' else aa for aa in sequence]
    prefix = ''
    for item in filter(None, (s.strip() for s in label['assigned_modifications'].split(','))):
        match = re.fullmatch(r'(N-term|\d+[A-Z])\(([-\d.]+)\)', item)
        if not match:
            raise ValueError('Unsupported modification format: '+item)
        position, mass = match[1], float(match[2])
        if position == 'N-term' and abs(mass-42.0106) < 0.01:
            prefix = '+42.011'
        elif position != 'N-term':
            index = int(position[:-1])-1
            assert sequence[index] == position[-1].replace('I', 'L')
            if position[-1] == 'C' and abs(mass-57.02146) < 0.01:
                continue
            if position[-1] == 'M' and abs(mass-15.9949) < 0.01:
                tokens[index] = 'M+15.995'
            else:
                raise ValueError('Unsupported modification: '+item)
        else:
            raise ValueError('Unsupported terminal modification: '+item)
    return prefix + ''.join(tokens)


class MixedDataset:
    def __init__(self, path, limit_per_class=32, seed=454):
        self.path = Path(path)
        self.mz = np.load(self.path/'mz.npy', mmap_mode='r')
        self.intensity = np.load(self.path/'intensity.npy', mmap_mode='r')
        self.offsets = np.load(self.path/'offsets.npy', mmap_mode='r')
        records = [json.loads(line) for line in (self.path/'records.jsonl').open()]
        # Start the tiny pilot with a stricter subset; this still is not ground truth.
        records = [r for r in records if all(label['report_qvalue'] <= 0.001
                   and label['percolator_qvalue'] <= 0.001 and label['percolator_pep'] <= 0.01
                   for label in r['labels'])]
        compatible = []
        for record in records:
            try:
                for label in record['labels']:
                    if not set(label['sequence']).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
                        raise ValueError('Noncanonical residue')
                    sequence_with_modifications(label)
                compatible.append(record)
            except (ValueError, AssertionError):
                continue
        records = compatible
        rng = random.Random(seed)
        self.records = []
        for count in [1, 2]:
            pool = [r for r in records if len(r['labels']) == count]
            self.records.extend(rng.sample(pool, min(limit_per_class, len(pool))))
        rng.shuffle(self.records)

    def batch(self, indices, device):
        peaks, precursors, labels, selected = [], [], [], []
        for index in indices:
            record = self.records[index]
            a,b = self.offsets[record['peak_row']:record['peak_row']+2]
            instrument = record['instrument_input']
            mz = instrument['selected_precursor_mz']
            charge = instrument['selected_precursor_charge']
            assert charge > 0
            spectrum = MsmsSpectrum(str(record['scan']), mz, charge,
                                    self.mz[a:b].copy(), self.intensity[a:b].astype(np.float32))
            spectrum.set_mz_range(50, 2500)
            spectrum.remove_precursor_peak(2.0, 'Da')
            spectrum.filter_intensity(0.01, 150)
            spectrum.scale_intensity('root', 1)
            norm = np.linalg.norm(spectrum.intensity)
            if not len(spectrum.mz) or not np.isfinite(norm) or norm == 0:
                raise ValueError('Empty/invalid processed spectrum: '+str(record['scan']))
            peaks.append(torch.from_numpy(np.column_stack([spectrum.mz, spectrum.intensity/norm])).float())
            # Global instrument condition only; no candidate precursor is fed to the model.
            precursors.append([(mz-1.007276466621)*charge, charge, mz])
            labels.append([sequence_with_modifications(label) for label in record['labels']])
            selected.append({'raw_file': record['raw_file'], 'scan': record['scan']})
        return (nn.utils.rnn.pad_sequence(peaks, batch_first=True).to(device),
                torch.tensor(precursors, dtype=torch.float32, device=device), labels, selected)


class SeparatorModel(nn.Module):
    def __init__(self, checkpoint, freeze_encoder=True, encoder_last_layers=0):
        super().__init__()
        # Official trusted v4 checkpoint contains legacy NumPy training metadata.
        # Build the same model directly, rather than relying on old Lightning's
        # implicit torch.load default (changed to weights_only=True in torch 2.6).
        saved_base = torch.load(checkpoint, map_location='cpu', weights_only=False)
        base = Spec2Pep(**saved_base['hyper_parameters'])
        if saved_base.get('format') != 'architecture_only_v1':
            base.load_state_dict(saved_base['state_dict'], strict=True)
        del saved_base
        self.encoder = base.encoder
        self.decoder = base.decoder
        assert self.decoder.reverse, 'Pilot terminal-token handling assumes the v4 reverse decoder.'
        self.freeze_encoder = freeze_encoder
        self.stop = self.decoder._aa2idx['$']
        old_embedding = self.decoder.aa_encoder
        old_final = self.decoder.final
        self.separator = old_embedding.num_embeddings
        self.decoder._amino_acids.append('|')
        self.decoder._aa2idx['|'] = self.separator
        self.decoder._idx2aa[self.separator] = '|'
        embedding = nn.Embedding(old_embedding.num_embeddings+1, old_embedding.embedding_dim, padding_idx=0)
        final = nn.Linear(old_final.in_features, old_final.out_features+1)
        with torch.no_grad():
            embedding.weight[:-1].copy_(old_embedding.weight)
            embedding.weight[-1].normal_(std=0.02)
            final.weight[:-1].copy_(old_final.weight)
            final.bias[:-1].copy_(old_final.bias)
            final.weight[-1].copy_(old_final.weight[1:].mean(dim=0))
            final.bias[-1] = -3.0
        self.decoder.aa_encoder = embedding
        self.decoder.final = final
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.encoder_last_layers = 0
        if encoder_last_layers:
            blocks = self.encoder.transformer_encoder.layers
            if not 0 < encoder_last_layers <= len(blocks):
                raise ValueError('Invalid number of encoder layers to unfreeze.')
            self.encoder.requires_grad_(False)
            for block in blocks[-encoder_last_layers:]:
                block.requires_grad_(True)
            self.encoder_last_layers = encoder_last_layers
            # memories() must NOT enter no_grad when any encoder layer is trained.
            self.freeze_encoder = False
        self.nterminal_ids = [idx for idx, token in self.decoder._idx2aa.items() if token.startswith(('+', '-'))]

    def memories(self, spectra):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(spectra)
        return self.encoder(spectra)

    def targets(self, labels, swapped=False):
        result = []
        for peptides in labels:
            peptides = list(reversed(peptides)) if swapped else peptides
            parts = [self.decoder.tokenize(p, partial=True) for p in peptides]
            tokens = parts[0]
            if len(parts) == 2:
                tokens = torch.cat([tokens, tokens.new_tensor([self.separator]), parts[1]])
            tokens = torch.cat([tokens, tokens.new_tensor([self.stop])])
            result.append(tokens)
        return result

    def loss_for_order(self, labels, precursors, memory, mask, swapped):
        prediction, truth = self.decoder(self.targets(labels, swapped), precursors, memory, mask)
        scores = prediction[:, :-1]
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), truth.flatten(),
                               ignore_index=0, reduction='none').reshape_as(truth)
        return loss.sum(dim=1)/(truth != 0).sum(dim=1).clamp_min(1)

    def loss(self, spectra, precursors, labels):
        memory, mask = self.memories(spectra)
        original = self.loss_for_order(labels, precursors, memory, mask, False)
        swapped = self.loss_for_order(labels, precursors, memory, mask, True)
        return torch.minimum(original, swapped).mean()

    @torch.no_grad()
    def generate(self, spectra, precursors, max_tokens=120, min_aa=6):
        self.eval()
        memory, mask = self.memories(spectra)
        batch = len(spectra)
        prefixes = [[] for _ in range(batch)]
        done = [False]*batch
        for _ in range(max_tokens):
            sequences = [torch.tensor(p, dtype=torch.long, device=spectra.device) for p in prefixes] if prefixes[0] else None
            logits, _ = self.decoder(sequences, precursors, memory, mask)
            for i in range(batch):
                if done[i]:
                    continue
                scores = logits[i, len(prefixes[i])].clone()
                scores[0] = -torch.inf
                segment = prefixes[i][prefixes[i].index(self.separator)+1:] if self.separator in prefixes[i] else prefixes[i]
                aa_count = sum(t not in self.nterminal_ids for t in segment)
                if self.separator in prefixes[i]:
                    scores[self.separator] = -torch.inf
                if aa_count < min_aa:
                    scores[self.stop] = -torch.inf
                    scores[self.separator] = -torch.inf
                    scores[self.nterminal_ids] = -torch.inf
                if segment and segment[-1] in self.nterminal_ids and self.decoder.reverse:
                    allowed = [self.stop] + ([] if self.separator in prefixes[i] else [self.separator])
                    keep = torch.full_like(scores, -torch.inf)
                    keep[allowed] = scores[allowed]
                    scores = keep
                token = int(scores.argmax())
                prefixes[i].append(token)
                done[i] = token == self.stop
            if all(done):
                break
        output = []
        for tokens, ended in zip(prefixes, done):
            parts, current = [], []
            for token in tokens:
                if token in [self.stop, self.separator]:
                    if current:
                        pieces = [self.decoder._idx2aa[t] for t in current]
                        parts.append(''.join(reversed(pieces) if self.decoder.reverse else pieces))
                    current = []
                else:
                    current.append(token)
            # An unfinished last peptide is not presented as a completed prediction.
            output.append({'peptides': list(dict.fromkeys(parts)), 'ended_with_eos': ended,
                           'truncated': not ended, 'raw_tokens': tokens})
        return output


def optimizer_step_with_new_token_rate(optimizer, model, factor=1.0):
    """Apply a distinct AdamW learning-rate multiplier to appended token rows.

    Retains the existing state_dict layout. Scaling gradients alone would not
    yield the intended Adam update because of its normalization.
    """
    if factor == 1.0:
        optimizer.step()
        return
    if factor <= 0:
        raise ValueError('New-token learning-rate factor must be positive.')
    parameters = [model.decoder.final.weight, model.decoder.final.bias,
                  model.decoder.aa_encoder.weight]
    previous = [parameter[-1].detach().clone() for parameter in parameters]
    optimizer.step()
    with torch.no_grad():
        for parameter, old in zip(parameters, previous):
            parameter[-1].copy_(old + factor * (parameter[-1] - old))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--new-token-lr-factor', type=float, default=1.0,
                        help='New-token row update multiplier; 1 preserves the original pilot recipe.')
    args = parser.parse_args()
    allowed = sorted(__import__('os').sched_getaffinity(0))
    __import__('os').sched_setaffinity(0, allowed[:4])
    torch.set_num_threads(4)
    random.seed(454)
    np.random.seed(454)
    torch.manual_seed(454)
    assert torch.cuda.is_available(), 'CUDA not available'
    device = torch.device('cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset = MixedDataset(args.data)
    model = SeparatorModel(args.checkpoint).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    fixed = dataset.batch(list(range(args.batch_size)), device)
    model.eval()
    with torch.no_grad():
        initial_loss = float(model.loss(*fixed[:3]))
    logs = []
    rng = random.Random(454)
    for step in range(args.steps):
        model.train()
        spectra, precursors, labels, _ = dataset.batch(rng.sample(range(len(dataset.records)), args.batch_size), device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(spectra, precursors, labels)
        assert torch.isfinite(loss), loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_step_with_new_token_rate(optimizer, model, args.new_token_lr_factor)
        item = {'step': step+1, 'training_loss': float(loss.detach())}
        logs.append(item)
        if (step+1) % 4 == 0:
            print(json.dumps(item), flush=True)
    model.eval()
    with torch.no_grad():
        final_loss = float(model.loss(*fixed[:3]))
    predictions = model.generate(fixed[0], fixed[1])
    summary = {'steps': args.steps, 'batch_size': args.batch_size, 'training_spectra': len(dataset.records),
               'new_token_learning_rate_factor': args.new_token_lr_factor,
               'encoder_frozen': True, 'separator_id': model.separator,
               'pilot_label_filter': 'both report/percolator q<=0.001 and percolator PEP<=0.01 per label',
               'fixed_training_batch_loss_before': initial_loss, 'fixed_training_batch_loss_after': final_loss,
               'peak_preprocessing': 'Casanovo v4-style: top150, min relative intensity0.01, sqrt/L2',
               'precursor_input': 'instrument-selected global condition; no candidate precursor input',
               'decoding': 'greedy, no single-peptide mass filter; maximum two sequences',
               'evaluation_status': 'training-flow pilot only; no held-out performance claim',
               'examples': [{'spectrum': s, 'reference_pseudolabels': y, 'prediction': p}
                            for s,y,p in zip(fixed[3], fixed[2], predictions)]}
    torch.save({'state_dict': model.state_dict(), 'source_checkpoint': args.checkpoint,
                'separator_id': model.separator, 'summary': summary,
                'note': 'Research wrapper state; not a drop-in standard Casanovo checkpoint; optimizer state omitted.'},
               out/'separator_pilot.pt')
    (out/'training_log.json').write_text(json.dumps(logs, indent=2))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'used_training_scans.json').write_text(json.dumps([{'raw_file': r['raw_file'], 'scan': r['scan']} for r in dataset.records], indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
