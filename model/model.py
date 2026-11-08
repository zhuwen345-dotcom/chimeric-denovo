"""Casanovo v4 sequence decoder plus a CasanovoTL-style binary prediction head.

Reference: Noble-Lab/casanovo-tl, chimera/chimera_task.ipynb,
"Casanovo foundation": 512 -> 512 -> 1 with ReLU. We use logits and
BCEWithLogitsLoss instead of explicitly applying Sigmoid before BCE.
The classifier predicts one versus exactly two search labels in this prototype.
It does not identify the sequences or partition peaks between peptides.
"""
import torch
from torch import nn
from torch.nn import functional as F

from separator_base import SeparatorModel


class ChimeraSequenceModel(SeparatorModel):
    def __init__(self, checkpoint, freeze_encoder=True, encoder_last_layers=0,
                 routing='unconstrained', threshold=0.5, routing_strength=1.0):
        super().__init__(checkpoint, freeze_encoder, encoder_last_layers)
        self.classification_head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.routing, self.threshold, self.routing_strength = routing, threshold, routing_strength

    @staticmethod
    def pool_peaks(memory, mask):
        """Mean real peak embeddings; exclude padding AND Casanovo's global token."""
        valid = ~mask[:, 1:]
        counts = valid.sum(dim=1, keepdim=True)
        if (counts == 0).any():
            raise ValueError('Cannot classify a spectrum with no retained peaks.')
        return (memory[:, 1:] * valid.unsqueeze(-1)).sum(dim=1) / counts

    def classify_memory(self, memory, mask):
        pooled = self.pool_peaks(memory, mask)
        if self.training and getattr(self, 'classifier_features_detached', False):
            pooled = pooled.detach()
        return self.classification_head(pooled).squeeze(-1)

    def configure_task(self, task, encoder_last_layers=0):
        if task not in ('classifier', 'joint', 'sequence') or encoder_last_layers not in (0, 1):
            raise ValueError('Unknown task or encoder layer count.')
        if task == 'classifier' and encoder_last_layers:
            raise ValueError('The classifier-only stage uses a frozen encoder, as in TL embeddings training.')
        self.encoder.requires_grad_(False)
        self.freeze_encoder = not bool(encoder_last_layers)
        if encoder_last_layers:
            self.encoder.transformer_encoder.layers[-1].requires_grad_(True)
        self.decoder.requires_grad_(task != 'classifier')
        self.classification_head.requires_grad_(task != 'sequence')

    def loss_components(self, spectra, precursors, labels, task):
        if any(len(p) not in (1, 2) for p in labels):
            raise ValueError('Only one or two distinct peptide labels are supported.')
        memory, mask = self.memories(spectra)
        logits = self.classify_memory(memory, mask)
        truth = logits.new_tensor([len(p) == 2 for p in labels])
        classification = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
        sequence = None
        if task != 'classifier':
            original = self.loss_for_order(labels, precursors, memory, mask, False)
            if truth.bool().any():
                swapped = self.loss_for_order(labels, precursors, memory, mask, True)
                sequence = torch.where(truth.bool(), torch.minimum(original, swapped), original)
            else:
                sequence = original
        return sequence, classification, logits

    @torch.no_grad()
    def generate(self, spectra, precursors, max_tokens=120, min_aa=6,
                 routing=None, threshold=None):
        """No reference labels or searched secondary precursor enter generation.

        hard: constrain the number of segments using predicted count, not truth.
        soft: add a binary log prior to SEP/EOS after the first segment's minimum
        length. This is an experimental decoding heuristic, not TL's algorithm.
        unconstrained: ignore the head when choosing tokens (sequence control).
        """
        self.eval()
        mode = self.routing if routing is None else routing
        cutoff = self.threshold if threshold is None else threshold
        if mode not in ('unconstrained', 'soft', 'hard') or not 0 < cutoff < 1:
            raise ValueError('Invalid routing mode or threshold.')
        memory, mask = self.memories(spectra)
        probabilities = self.classify_memory(memory, mask).sigmoid()
        predicted_counts = 1 + (probabilities >= cutoff).long()
        prefixes = [[] for _ in range(len(spectra))]
        done = [False] * len(spectra)
        for _ in range(max_tokens):
            sequences = [torch.tensor(p, dtype=torch.long, device=spectra.device) for p in prefixes] if prefixes[0] else None
            logits, _ = self.decoder(sequences, precursors, memory, mask)
            for i in range(len(spectra)):
                if done[i]:
                    continue
                scores = logits[i, len(prefixes[i])].clone()
                scores[0] = -torch.inf
                has_separator = self.separator in prefixes[i]
                segment = prefixes[i][prefixes[i].index(self.separator) + 1:] if has_separator else prefixes[i]
                aa_count = sum(t not in self.nterminal_ids for t in segment)
                if has_separator:
                    scores[self.separator] = -torch.inf
                if aa_count < min_aa:
                    scores[self.stop] = scores[self.separator] = -torch.inf
                    scores[self.nterminal_ids] = -torch.inf
                if segment and segment[-1] in self.nterminal_ids and self.decoder.reverse:
                    allowed = [self.stop] + ([] if has_separator else [self.separator])
                    keep = torch.full_like(scores, -torch.inf)
                    keep[allowed] = scores[allowed]
                    scores = keep
                if mode == 'hard':
                    if predicted_counts[i] == 1:
                        scores[self.separator] = -torch.inf
                    elif not has_separator:
                        scores[self.stop] = -torch.inf
                elif mode == 'soft' and not has_separator and aa_count >= min_aa:
                    p = probabilities[i].clamp(1e-6, 1 - 1e-6)
                    scores[self.separator] += self.routing_strength * p.log()
                    scores[self.stop] += self.routing_strength * torch.log1p(-p)
                if not torch.isfinite(scores).any():
                    raise RuntimeError('All decoding choices were masked; inspect the count constraints.')
                token = int(scores.argmax())
                prefixes[i].append(token)
                done[i] = token == self.stop
            if all(done):
                break
        output = []
        for i, (tokens, ended) in enumerate(zip(prefixes, done)):
            parts, current = [], []
            for token in tokens:
                if token in (self.stop, self.separator):
                    if current:
                        pieces = [self.decoder._idx2aa[t] for t in current]
                        parts.append(''.join(reversed(pieces) if self.decoder.reverse else pieces))
                    current = []
                else:
                    current.append(token)
            unique = list(dict.fromkeys(parts))
            count = int(predicted_counts[i])
            output.append({'peptides': unique, 'ended_with_eos': ended, 'truncated': not ended,
                           'chimeric_probability': float(probabilities[i]), 'predicted_count': count,
                           'routing': mode, 'count_constraint_satisfied': ended and len(unique) == count,
                           'duplicate_segment': len(parts) != len(unique), 'raw_tokens': tokens})
        return output
