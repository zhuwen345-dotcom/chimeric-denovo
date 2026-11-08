"""Two independently decoded peptides with permutation-invariant supervision.

The second slot never consumes the first slot's reference/prediction. Both use
the same spectrum and shared pretrained decoder. Instrument precursor features
are context, not asserted peptide masses: trainable slot-specific scales start
at (1,1) for slot 0 and (0,0) for the secondary slot. No searched secondary mass
or charge is used. This is an experimental architecture, not a validated gain.
"""
from functools import lru_cache
import re

import torch
from torch import nn
from torch.nn import functional as F

from model import ChimeraSequenceModel


def match_peptide_losses(zero_a, one_b, zero_b, one_a, hard_weight=0.):
    """Two-label matching with optional extra loss for the harder peptide.

    Difficulty means length-normalized teacher-forced CE, not peak intensity.
    Labels may select an assignment during training, never during generation.
    Return ordinary matched CE separately so validation retains its old meaning.
    """
    direct, swapped = (zero_a + one_b) / 2, (zero_b + one_a) / 2
    if hard_weight == 0:
        return torch.minimum(direct, swapped), torch.zeros_like(direct), direct <= swapped
    hard_direct, hard_swapped = torch.maximum(zero_a, one_b), torch.maximum(zero_b, one_a)
    choose_direct = direct + hard_weight * hard_direct <= swapped + hard_weight * hard_swapped
    ordinary = torch.where(choose_direct, direct, swapped)
    extra = hard_weight * torch.where(choose_direct, hard_direct, hard_swapped)
    return ordinary, extra, choose_direct


class DualSlotModel(ChimeraSequenceModel):
    architecture = 'dual_slot_v1'
    joint_candidate_selection = False

    def __init__(self, *args, evaluation_beam_size=4, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.decoder.aa_encoder.embedding_dim
        self.slot_embeddings = nn.Parameter(torch.zeros(2, width))
        with torch.no_grad():
            self.slot_embeddings[1].normal_(std=.02)
        self.precursor_scales = nn.Parameter(torch.tensor([[1., 1.], [0., 0.]]))
        self.evaluation_beam_size = evaluation_beam_size

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Only explicit non-strict initialization may introduce the new slots.
        # A full resume remains strict and requires both learned tensors.
        new = ('slot_embeddings', 'precursor_scales')
        if not strict and not any(k in state_dict for k in new):
            state_dict = dict(state_dict)
            current = self.state_dict()
            state_dict.update({k: current[k] for k in new})
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def configure_task(self, task, encoder_last_layers=0):
        if task != 'joint':
            raise ValueError('Dual-slot experiment currently supports joint only.')
        super().configure_task(task, encoder_last_layers)
        self.slot_embeddings.requires_grad_(True)
        self.precursor_scales.requires_grad_(True)

    def slot_forward(self, sequences, precursors, memory, mask, slot):
        """Pretrained decoder with a slot-conditioned start token; no extra AA."""
        decoder = self.decoder
        if sequences is None:
            tokens = None
        else:
            tokens = nn.utils.rnn.pad_sequence(
                [decoder.tokenize(s) for s in sequences], batch_first=True)
        mass = decoder.mass_encoder(precursors[:, None, 0])
        charge = decoder.charge_encoder(precursors[:, 1].int() - 1)[:, None, :]
        start = (mass * self.precursor_scales[slot, 0] +
                 charge * self.precursor_scales[slot, 1] +
                 self.slot_embeddings[slot][None, None, :])
        target = start if tokens is None else torch.cat([start, decoder.aa_encoder(tokens)], dim=1)
        padding = torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
        if tokens is not None:
            padding[:, 1:] = tokens == 0
        target = decoder.pos_encoder(target)
        causal = torch.triu(torch.ones(target.shape[1], target.shape[1],
                                      device=target.device, dtype=torch.bool), diagonal=1)
        hidden = decoder.transformer_decoder(
            tgt=target, memory=memory, tgt_mask=causal,
            tgt_key_padding_mask=padding, memory_key_padding_mask=mask)
        return decoder.final(hidden), tokens

    def slot_loss(self, peptides, precursors, memory, mask, slot):
        targets = [torch.cat([self.decoder.tokenize(p, partial=True),
                              torch.tensor([self.stop], device=precursors.device)])
                   for p in peptides]
        logits, truth = self.slot_forward(targets, precursors, memory, mask, slot)
        scores = logits[:, :-1]
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), truth.flatten(),
                               ignore_index=0, reduction='none').reshape_as(truth)
        return loss.sum(1) / (truth != 0).sum(1).clamp_min(1)

    def loss_components(self, spectra, precursors, labels, task):
        if task != 'joint' or any(len(x) not in (1, 2) for x in labels):
            raise ValueError('Expected joint task with one/two distinct labels.')
        memory, mask = self.memories(spectra)
        logits = self.classify_memory(memory, mask)
        truth = logits.new_tensor([len(x) == 2 for x in labels])
        classification = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
        # Single spectra train only slot 0; the classifier controls cardinality.
        first = [x[0] for x in labels]
        sequence = self.slot_loss(first, precursors, memory, mask, 0)
        self.sequence_regularizer = sequence.new_zeros(sequence.shape)
        selected = truth.bool().nonzero(as_tuple=True)[0]
        if len(selected):
            double = [labels[i] for i in selected.tolist()]
            p, m, k = precursors[selected], memory[selected], mask[selected]
            a, b = [x[0] for x in double], [x[1] for x in double]
            zero_a = sequence[selected]
            one_b = self.slot_loss(b, p, m, k, 1)
            zero_b = self.slot_loss(b, p, m, k, 0)
            one_a = self.slot_loss(a, p, m, k, 1)
            sequence = sequence.clone()
            correct = torch.minimum(zero_a + one_b, zero_b + one_a) / 2
            sequence[selected] = correct
            if getattr(self, 'diversity_weight', 0):
                duplicate_a = (zero_a + one_a) / 2
                duplicate_b = (zero_b + one_b) / 2
                penalty = (F.relu(self.diversity_margin + correct - duplicate_a) +
                           F.relu(self.diversity_margin + correct - duplicate_b)) / 2
                self.sequence_regularizer[selected] = self.diversity_weight * penalty
        return sequence, classification, logits

    @torch.no_grad()
    def generate(self, spectra, precursors, max_tokens=120, min_aa=6,
                 routing=None, threshold=None, beam_size=None, length_penalty=1.):
        from evaluate_holdout import generate_beam
        self.eval()
        mode = self.routing if routing is None else routing
        if mode != 'hard':
            raise ValueError('Dual-slot cardinality currently uses hard classification; use routing=hard.')
        cutoff = self.threshold if threshold is None else threshold
        if not 0 < cutoff < 1:
            raise ValueError('Invalid threshold.')
        memory, mask = self.memories(spectra)
        probabilities = self.classify_memory(memory, mask).sigmoid()
        counts = 1 + (probabilities >= cutoff).long()
        width = self.evaluation_beam_size if beam_size is None else beam_size
        collect = self.joint_candidate_selection and width > 1
        first_view = _SlotView(self, memory, mask, 0)
        first = generate_beam(first_view, spectra, precursors, beam_size=width,
                              max_tokens=max_tokens, min_aa=min_aa, length_penalty=length_penalty,
                              return_candidates=collect)
        second = {}
        chosen = (counts == 2).nonzero(as_tuple=True)[0]
        if len(chosen):
            view = _SlotView(self, memory[chosen], mask[chosen], 1)
            decoded = generate_beam(view, spectra[chosen], precursors[chosen], beam_size=width,
                                    max_tokens=max_tokens, min_aa=min_aa, length_penalty=length_penalty,
                                    return_candidates=collect)
            second = dict(zip(chosen.tolist(), decoded))
        out = []
        for i, primary in enumerate(first):
            secondary = second.get(i)
            if collect and secondary is not None:
                pairs = [(a, b) for a in primary.get('candidates', [])
                         for b in secondary.get('candidates', [])
                         if len(a['peptides']) == len(b['peptides']) == 1 and
                         a['peptides'][0] != b['peptides'][0]]
                if pairs:
                    primary, secondary = max(pairs, key=lambda pair:
                                             pair[0]['beam_score'] + pair[1]['beam_score'])
            parts = primary['peptides'][:]
            ended = primary['ended_with_eos']
            if secondary is not None:
                parts += secondary['peptides']
                ended = ended and secondary['ended_with_eos']
            unique = list(dict.fromkeys(parts))
            count = int(counts[i])
            out.append({'peptides': unique, 'ended_with_eos': ended, 'truncated': not ended,
                        'chimeric_probability': float(probabilities[i]), 'predicted_count': count,
                        'routing': 'hard', 'count_constraint_satisfied': ended and len(unique) == count,
                        'duplicate_segment': len(parts) != len(unique), 'raw_tokens': primary['raw_tokens'],
                        'beam_size': width, 'beam_length_penalty': length_penalty})
        return out


class DistinctDualSlotModel(DualSlotModel):
    """v2: score distinct candidate PAIRS and train against duplicate sets.

    The extra hinge compares correct-set mean NLL with AA and BB duplicate-set
    NLL. Labels enter the TRAINING loss only, never candidate selection.
    """
    architecture = 'dual_slot_v2'
    joint_candidate_selection = True

    def __init__(self, *args, diversity_weight=.5, diversity_margin=.2, **kwargs):
        super().__init__(*args, **kwargs)
        if diversity_weight < 0 or diversity_margin < 0:
            raise ValueError('Diversity weight and margin must be nonnegative.')
        self.diversity_weight = diversity_weight
        self.diversity_margin = diversity_margin


@lru_cache(maxsize=100000)
def fragment_mz(peptide):
    """Weak training annotations: b/y ions at charges 1 and 2, with exact PTMs.

    This is used only by loss_components. Reference peptides never enter
    generation or its peak gates. Shared peaks may be positive for both slots.
    """
    masses = dict(zip('ACDEFGHIKLMNPQRSTVWY', (
        71.037114, 103.009185, 115.026943, 129.042593, 147.068414,
        57.021464, 137.058912, 113.084064, 128.094963, 113.084064,
        131.040485, 114.042927, 97.052764, 128.058578, 156.101111,
        87.032028, 101.047679, 99.068414, 186.079313, 163.063329)))
    tokens = re.findall(r'[A-Z](?:[+-]\d+(?:\.\d+)?)?|^[+-]\d+(?:\.\d+)?', peptide)
    if ''.join(tokens) != peptide:
        raise ValueError('Unsupported fragment annotation sequence: ' + peptide)
    residues, terminal = [], 0.
    for token in tokens:
        if token[0] in '+-':
            terminal += float(token)
        else:
            residues.append(masses[token[0]] + (float(token[1:]) if len(token) > 1 else 0.))
    ions, prefix, total = [], 0., sum(residues)
    for residue in residues[:-1]:
        prefix += residue
        for neutral in (prefix + terminal, total - prefix + 18.010565):
            for charge in (1, 2):
                ions.append(neutral / charge + 1.007276466621)
    return tuple(ions)


class EvidenceDualSlotModel(DistinctDualSlotModel):
    """Learn spectrum-dependent peak emphasis for each independently decoded slot.

    Zero output initialization makes peak multipliers exactly one, preserving
    the v2 initializer. Weak b/y annotations guide the gates during training;
    inference gates depend only on observed spectrum embeddings. The 0.5..1.5
    multiplier preserves unannotated evidence instead of removing peaks.
    """
    architecture = 'dual_slot_evidence'

    def __init__(self, *args, evidence_weight=.1, hard_peptide_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        if evidence_weight < 0:
            raise ValueError('Evidence weight must be nonnegative.')
        if not 0 <= hard_peptide_weight <= 1:
            raise ValueError('Hard-peptide auxiliary weight must be finite and in [0,1].')
        width = self.decoder.aa_encoder.embedding_dim
        self.peak_evidence = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.peak_evidence[-1].weight)
        nn.init.zeros_(self.peak_evidence[-1].bias)
        self.evidence_weight = evidence_weight
        self.hard_peptide_weight = hard_peptide_weight

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict and not any(k.startswith('peak_evidence.') for k in state_dict):
            state_dict = dict(state_dict)
            state_dict.update({k: v for k, v in self.state_dict().items()
                               if k.startswith('peak_evidence.')})
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def configure_task(self, task, encoder_last_layers=0):
        super().configure_task(task, encoder_last_layers)
        self.peak_evidence.requires_grad_(True)

    def slot_forward(self, sequences, precursors, memory, mask, slot):
        evidence = self.peak_evidence(memory[:, 1:])[..., slot]
        multiplier = .5 + evidence.sigmoid()
        conditioned = torch.cat((memory[:, :1], memory[:, 1:] * multiplier.unsqueeze(-1)), dim=1)
        return super().slot_forward(sequences, precursors, conditioned, mask, slot)

    @staticmethod
    def peak_targets(spectra, peptides):
        ions = [fragment_mz(p) for p in peptides]
        maximum = max(map(len, ions))
        table = spectra.new_full((len(ions), maximum), -10000.)
        for i, values in enumerate(ions):
            table[i, :len(values)] = spectra.new_tensor(values)
        tolerance = torch.maximum(table.abs() * 20e-6, table.new_tensor(.02))
        matches = (spectra[:, :, 0, None] - table[:, None, :]).abs() <= tolerance[:, None, :]
        return matches.any(-1) & (spectra[:, :, 1] > 0)

    @staticmethod
    def peak_loss(logits, targets, valid):
        positive, negative = targets & valid, ~targets & valid
        values = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
        # Keep absent annotations neutral: do not train an all-negative gate on
        # peptides with no matched b/y evidence in the retained input peaks.
        count = positive.sum(1)
        pos = (values * positive).sum(1) / count.clamp_min(1)
        neg = (values * negative).sum(1) / negative.sum(1).clamp_min(1)
        return (pos + neg) * .5 * (count > 0)

    def loss_components(self, spectra, precursors, labels, task):
        if task != 'joint' or any(len(x) not in (1, 2) for x in labels):
            raise ValueError('Expected joint task with one/two distinct labels.')
        memory, mask = self.memories(spectra)
        logits = self.classify_memory(memory, mask)
        is_double = logits.new_tensor([len(x) == 2 for x in labels]).bool()
        classification = F.binary_cross_entropy_with_logits(logits, is_double.to(logits.dtype), reduction='none')
        first = [x[0] for x in labels]
        sequence = self.slot_loss(first, precursors, memory, mask, 0)
        peak_logits = self.peak_evidence(memory[:, 1:, :self.decoder.aa_encoder.embedding_dim])
        valid = ~mask[:, 1:]
        target_a = self.peak_targets(spectra, first)
        aux = self.peak_loss(peak_logits[..., 0], target_a, valid)
        penalty = sequence.new_zeros(sequence.shape)
        hard_penalty = sequence.new_zeros(sequence.shape)
        selected = is_double.nonzero(as_tuple=True)[0]
        if len(selected):
            double = [labels[i] for i in selected.tolist()]
            p, m, k = precursors[selected], memory[selected], mask[selected]
            a, b = [x[0] for x in double], [x[1] for x in double]
            zero_a = sequence[selected]
            one_b = self.slot_loss(b, p, m, k, 1)
            zero_b = self.slot_loss(b, p, m, k, 0)
            one_a = self.slot_loss(a, p, m, k, 1)
            correct, hard, choose_direct = match_peptide_losses(
                zero_a, one_b, zero_b, one_a,
                self.hard_peptide_weight if self.training else 0.)
            self.matched_direct_assignment = choose_direct
            hard_penalty[selected] = hard
            sequence = sequence.clone(); sequence[selected] = correct
            target_b = self.peak_targets(spectra[selected], b)
            evidence, v, ta = peak_logits[selected], valid[selected], target_a[selected]
            aux_direct = (self.peak_loss(evidence[..., 0], ta, v) + self.peak_loss(evidence[..., 1], target_b, v)) / 2
            aux_swapped = (self.peak_loss(evidence[..., 0], target_b, v) + self.peak_loss(evidence[..., 1], ta, v)) / 2
            aux = aux.clone(); aux[selected] = torch.where(choose_direct, aux_direct, aux_swapped)
            penalty[selected] = self.diversity_weight * (
                F.relu(self.diversity_margin + correct - (zero_a + one_a) / 2) +
                F.relu(self.diversity_margin + correct - (zero_b + one_b) / 2)) / 2
        self.evidence_loss = aux
        self.hard_peptide_loss = hard_penalty
        self.sequence_regularizer = penalty + self.evidence_weight * aux + hard_penalty
        return sequence, classification, logits


class SpectralSeparation(nn.Module):
    """Learn exclusive/shared peak allocation and contextualize each slot.

    Allocation states are unassigned, A-only, B-only, shared. They describe
    weak fragment annotations, not physical ion-intensity fractions.
    Neutral allocation and a zero residual strength preserve the parent model.
    """
    def __init__(self, width):
        super().__init__()
        self.allocation = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 4))
        nn.init.zeros_(self.allocation[-1].weight)
        nn.init.zeros_(self.allocation[-1].bias)
        self.context = nn.TransformerEncoderLayer(width, 8, 1024, dropout=.1,
                                                  batch_first=True, norm_first=True)
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, memory, mask, slot):
        probabilities = self.allocation(memory[:, 1:]).softmax(-1)
        own = probabilities[..., slot+1] + probabilities[..., 3]
        multiplier = .25 + 1.5 * own
        conditioned = torch.cat((memory[:, :1], memory[:, 1:] * multiplier[..., None]), dim=1)
        contextual = self.context(conditioned, src_key_padding_mask=mask)
        return conditioned + self.strength.tanh() * (contextual-conditioned)


class SeparationDualSlotModel(EvidenceDualSlotModel):
    """Slot-specific spectral reconstruction before the shared sequence decoder."""
    architecture = 'dual_slot_separation'
    allocation_weight = .05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_separation = SpectralSeparation(self.decoder.aa_encoder.embedding_dim)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict and not any(k.startswith('spectral_separation.') for k in state_dict):
            state_dict = dict(state_dict)
            state_dict.update({k:v for k,v in self.state_dict().items() if k.startswith('spectral_separation.')})
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def configure_task(self, task, encoder_last_layers=0):
        super().configure_task(task, encoder_last_layers)
        self.spectral_separation.requires_grad_(True)

    def prepare_slot_memory(self, memory, mask, slot):
        # Allocate from original frozen embeddings; retain the parent's gate.
        separated = self.spectral_separation(memory, mask, slot)
        multiplier = .5 + self.peak_evidence(memory[:, 1:])[..., slot].sigmoid()
        return torch.cat((separated[:, :1], separated[:, 1:] * multiplier[..., None]), dim=1)

    def slot_forward(self, sequences, precursors, memory, mask, slot):
        conditioned = self.prepare_slot_memory(memory, mask, slot)
        return DualSlotModel.slot_forward(self, sequences, precursors, conditioned, mask, slot)

    def loss_components(self, spectra, precursors, labels, task):
        result = super().loss_components(spectra, precursors, labels, task)
        memory, mask = self.memories(spectra)
        a = self.peak_targets(spectra, [x[0] for x in labels])
        b = torch.zeros_like(a)
        selected = torch.tensor([len(x)==2 for x in labels], device=spectra.device).nonzero(as_tuple=True)[0]
        if len(selected):
            second = self.peak_targets(spectra[selected], [labels[i][1] for i in selected.tolist()])
            first = a[selected].clone()
            direct = self.matched_direct_assignment[:, None]
            a = a.clone()
            a[selected] = torch.where(direct, first, second)
            b[selected] = torch.where(direct, second, first)
        targets = a.long() + 2*b.long()
        logits = self.spectral_separation.allocation(memory[:, 1:])
        values = F.cross_entropy(logits.transpose(1,2), targets, reduction='none')
        valid = ~mask[:, 1:]
        category_loss, category_count = values.new_zeros(len(labels)), values.new_zeros(len(labels))
        for category in range(4):
            chosen = valid & (targets==category)
            n = chosen.sum(1)
            category_loss += (values*chosen).sum(1)/n.clamp_min(1)
            category_count += (n>0)
        # No matching fragments means no reliable allocation supervision.
        annotated = ((a|b)&valid).any(1)
        allocation = category_loss/category_count.clamp_min(1)*annotated
        self.allocation_loss = allocation
        self.sequence_regularizer = self.sequence_regularizer + self.allocation_weight*allocation
        return result


class _DecoderView:
    def __init__(self, parent, slot, prepared=False):
        self.parent, self.slot = parent, slot
        self.prepared = prepared
        self.reverse = parent.decoder.reverse
        self._idx2aa = parent.decoder._idx2aa

    def __call__(self, sequences, precursors, memory, mask):
        if self.prepared:
            return DualSlotModel.slot_forward(self.parent, sequences, precursors, memory, mask, self.slot)
        return self.parent.slot_forward(sequences, precursors, memory, mask, self.slot)


class _SlotView:
    """Present one slot to the existing, reference-free beam implementation."""
    def __init__(self, parent, memory, mask, slot):
        self.parent, self.memory, self.mask = parent, memory, mask
        prepared = hasattr(parent, 'prepare_slot_memory')
        if prepared:
            self.memory = parent.prepare_slot_memory(memory, mask, slot)
        self.decoder = _DecoderView(parent, slot, prepared=prepared)
        self.separator, self.stop = parent.separator, parent.stop
        self.nterminal_ids = parent.nterminal_ids
        self.routing, self.threshold, self.routing_strength = 'hard', .5, 1.

    def eval(self):
        self.parent.eval()
        return self

    def memories(self, spectra):
        return self.memory, self.mask

    def classify_memory(self, memory, mask):
        # Each slot emits a single peptide: always forbid the old SEP token.
        return memory.new_full((len(memory),), -20.)

    def generate(self, spectra, precursors, **kwargs):
        return ChimeraSequenceModel.generate(self, spectra, precursors, **kwargs)
