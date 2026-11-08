"""Trainable second-peptide recovery from a predicted first peptide.

Reference anchors are used only in the training objective. Generation takes
only spectra and measured precursors and anchors every branch to a generated
candidate. Original independent-slot candidates remain available.
"""
import math
import random
import re
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fragment_attention import FragmentAttentionModel
from dual_slot import _SlotView
from evaluate_holdout import generate_beam
from evidence_decode import FEATURES, rank_spectrum
from route import apply_route


class ConditionalResidualModel(FragmentAttentionModel):
    architecture = 'conditional_residual_v1'

    def __init__(self, *args, conditional_weight=.75, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.decoder.aa_encoder.embedding_dim
        self.conditional_context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        nn.init.zeros_(self.conditional_context[-1].weight)
        nn.init.zeros_(self.conditional_context[-1].bias)
        self.conditional_start = nn.Parameter(torch.zeros(width))
        self.rt_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.conditional_weight = float(conditional_weight)
        self.residual_power = .30
        self.max_peptide_tokens = 52
        self.conditional_enabled = True
        self.precursor_dropout = .30

    def memories(self, spectra):
        memory, mask = super().memories(spectra)
        # The otherwise-unused global-token raw intensity carries one context
        # dropout decision per spectrum. All PIT permutations see the same
        # decision, including after selecting/reordering rows.
        if self.training:
            keep = torch.rand(len(memory), device=memory.device) >= self.precursor_dropout
            memory = memory.clone()
            memory[:, 0, -1] = keep.to(memory.dtype)
        return memory, mask

    def noisy_anchor(self, peptide):
        """A denoising training aid, not a substitute for free-generation tests."""
        draw = random.random()
        if draw < .10:
            return ''
        if draw >= .40:
            return peptide
        tokens = self.decoder.tokenize(peptide, partial=True).tolist()
        if len(tokens) < 3:
            return peptide
        residues = self.is_residue.nonzero(as_tuple=True)[0].tolist()
        if random.random() < .5:
            for _ in range(random.randint(1, min(3, len(tokens)))):
                i = random.randrange(len(tokens))
                if tokens[i] in residues:
                    tokens[i] = random.choice(residues)
        else:
            tokens = tokens[:random.randint(max(2, len(tokens)//2), len(tokens)-1)]
        pieces = [self.decoder._idx2aa[t] for t in tokens]
        return ''.join(reversed(pieces)) if self.decoder.reverse else ''.join(pieces)

    def conditional_memory(self, spectra, anchors, power=None):
        """Attenuate anchor-matched peaks in LINEAR intensity; keep every peak.

        Input intensities are sqrt/L2. The multiplier is therefore sqrt(power)
        and the spectrum is L2 normalized again. No target-specific exceptions
        protect shared peaks: exactly the same operation is used at inference.
        """
        if len(anchors) != len(spectra):
            raise ValueError('Anchor/spectrum row mismatch')
        hit = self.anchor_matches(spectra, anchors)
        if power is None:
            power = self.residual_power
        power = torch.as_tensor(power, device=spectra.device, dtype=spectra.dtype)
        if power.ndim == 1:
            power = power[:, None]
        intensity = spectra[..., 1] * torch.where(hit, power.sqrt(), 1.)
        intensity = intensity / intensity.norm(dim=1, keepdim=True).clamp_min(1e-8)
        residual = torch.stack((spectra[..., 0], intensity), dim=-1)
        memory, mask = self.memories(residual)
        # Score fragment evidence on the original observed intensities.
        memory = torch.cat((memory[..., :-2], F.pad(spectra.float(), (0, 0, 1, 0))), dim=-1)
        summaries = []
        for anchor in anchors:
            tokens = (self.decoder.tokenize(anchor, partial=True) if anchor else
                      torch.empty(0, dtype=torch.long, device=spectra.device))
            summaries.append(self.decoder.aa_encoder(tokens).mean(0) if len(tokens)
                             else memory.new_zeros(self.decoder.aa_encoder.embedding_dim))
        context = self.conditional_context(torch.stack(summaries))
        # A masked metadata row carries context through beam row reordering.
        # It is removed before the actual decoder attention operation.
        metadata = F.pad(context, (0, 3))[:, None, :]
        memory = torch.cat((F.pad(memory, (0, 1)), metadata), dim=1)
        mask = F.pad(mask, (0, 1), value=True)
        return memory, mask

    @lru_cache(maxsize=20000)
    def anchor_ions(self, anchor):
        if not anchor:
            return ()
        # Tokenize with the actual decoder vocabulary, including composite
        # terminal modifications such as +43.006-17.027.
        pieces = re.split(r'(?<=.)(?=[A-Z])', anchor.replace('I', 'L'))
        masses = self.decoder._peptide_mass.masses
        residue = [masses[p] for p in pieces if p[0].isalpha()]
        terminal = sum(masses[p] for p in pieces if p[0] in '+-')
        prefix, total, ions = 0., sum(residue), []
        for mass in residue[:-1]:
            prefix += mass
            for neutral in (prefix + terminal, total - prefix + 18.010565):
                ions.extend(neutral/z + 1.007276466621 for z in (1, 2))
        return tuple(ions)

    def anchor_matches(self, spectra, anchors):
        ions = [self.anchor_ions(p) for p in anchors]
        table = spectra.new_full((len(ions), max(map(len, ions), default=0)), -10000.)
        for i, values in enumerate(ions):
            table[i, :len(values)] = spectra.new_tensor(values)
        tolerance = (table.abs() * 20e-6).clamp_min(.02)
        return (((spectra[:, :, 0, None] - table[:, None, :]).abs() <= tolerance[:, None, :])
                .any(-1) & (spectra[:, :, 1] > 0))

    def slot_forward(self, sequences, precursors, memory, mask, slot, return_rt=False):
        width = self.decoder.aa_encoder.embedding_dim
        if memory.shape[-1] == width + 2:
            if not self.training:
                return super().slot_forward(sequences, precursors, memory, mask, slot)
            original = memory
            keep = memory[:, 0, -1][:, None, None]
            mass = self.decoder.mass_encoder(precursors[:, None, 0])
            charge = self.decoder.charge_encoder(precursors[:, 1].int()-1)[:, None, :]
            start = (mass*self.precursor_scales[slot, 0] + charge*self.precursor_scales[slot, 1])*keep
            start = start + self.slot_embeddings[slot][None, None, :]
        elif memory.shape[-1] == width + 3:
            context = memory[:, -1, :width]
            original = memory[:, :-1, :width+2]
            mask = mask[:, :-1]
            slot = 1
            # No asserted second precursor: it may differ from the selected
            # instrument precursor in mass, charge, or both.
            start = (self.slot_embeddings[1] + self.conditional_start + context)[:, None, :]
        else:
            raise ValueError('Unrecognized conditional memory')
        embedded, peaks = original[..., :width], original[:, 1:, -2:]
        decoder = self.decoder
        tokens = None if sequences is None else nn.utils.rnn.pad_sequence(
            [decoder.tokenize(s) for s in sequences], batch_first=True)
        target = start if tokens is None else torch.cat((start, decoder.aa_encoder(tokens)), dim=1)
        padding = torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
        if tokens is not None:
            padding[:, 1:] = tokens == 0
        target = decoder.pos_encoder(target)
        causal = torch.triu(torch.ones(target.shape[1], target.shape[1], dtype=torch.bool,
                                       device=target.device), diagonal=1)
        bias = self.peak_attention_bias(embedded, mask, slot, target.shape[1])
        hidden = decoder.transformer_decoder(tgt=target, memory=embedded, tgt_mask=causal,
            tgt_key_padding_mask=padding, memory_mask=bias, memory_key_padding_mask=None)
        logits = decoder.final(hidden).float() + self.fragment_bias(tokens, peaks, slot)
        if return_rt:
            valid = ~padding
            pooled = (hidden * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            rt = self.rt_head(pooled).squeeze(-1)
            return logits, tokens, rt
        return logits, tokens

    def loss_components(self, spectra, precursors, labels, task, rt_targets=None):
        result = super().loss_components(spectra, precursors, labels, task)
        self.conditional_sequence_ce = result[0].new_zeros(result[0].shape)
        self.rt_aux_loss = result[0].new_zeros(()).detach()
        if self.training and rt_targets is not None and getattr(self, 'rt_weight', 0.) > 0.:
            memory, mask = self.memories(spectra)
            first = [x[0] for x in labels]
            _, _, rt = self.slot_forward(first, precursors, memory, mask, 0, return_rt=True)
            valid = torch.isfinite(rt_targets)
            if valid.any():
                self.rt_aux_loss = ((rt[valid] - rt_targets[valid]) ** 2).mean()
        if not self.training or not self.conditional_weight:
            return result
        selected = [i for i, label in enumerate(labels) if len(label) == 2]
        if not selected:
            return result
        anchors, targets = [], []
        for i in selected:
            direction = random.randrange(2)
            anchors.append(self.noisy_anchor(labels[i][direction]))
            targets.append(labels[i][1-direction])
        ids = torch.tensor(selected, device=spectra.device)
        # Broad attenuation during training reduces reliance on one subtraction
        # strength and retains signal from overlapping b/y peaks.
        power = torch.empty(len(ids), device=spectra.device).uniform_(.15, .65)
        memory, mask = self.conditional_memory(spectra[ids], anchors, power)
        loss = self.slot_loss(targets, precursors[ids], memory, mask, 1)
        conditional = result[0].new_zeros(result[0].shape).index_copy(0, ids, loss)
        self.conditional_sequence_ce = conditional
        self.sequence_regularizer = self.sequence_regularizer + self.conditional_weight * conditional
        return result

    @torch.no_grad()
    def generate(self, spectra, precursors, max_tokens=120, min_aa=6, routing=None,
                 threshold=None, beam_size=None, length_penalty=1.):
        if routing not in (None, 'hard'):
            raise ValueError('Conditional output policy requires hard count routing')
        return decode_diagnostics(self, spectra, precursors,
            beam_size=self.evaluation_beam_size if beam_size is None else beam_size,
            policy=self.output_policy, threshold=threshold,
            max_tokens=min(max_tokens, self.max_peptide_tokens), min_aa=min_aa,
            length_penalty=length_penalty)


def select_expanded_candidates(row, policy):
    """Apply the original full policy once to the deduplicated expanded pool.

    Assignment choice follows rank_spectrum: keep the largest independent
    average log probability for each unordered peptide set, retaining the
    earlier assignment on ties. Both the raw beam and ranker see this pool.
    """
    unique = {}
    for candidate, features in zip(row['feature_sets'], row['features']):
        key = tuple(sorted(candidate))
        if key not in unique or features[2] > unique[key][2]:
            unique[key] = list(features)
    candidates = list(unique)
    features = list(unique.values())
    result = dict(row, feature_sets=[list(key) for key in candidates], features=features,
                  candidate_sets=[list(key) for key in candidates],
                  candidate_peptide_union=sorted({peptide for key in candidates for peptide in key}))
    hypotheses = {}
    for count in (1, 2):
        choices = [key for key in candidates if len(key) == count]
        hypotheses[str(count)] = list(max(choices, key=lambda key: unique[key][2])) if choices else []
    result['hypotheses_by_count'] = hypotheses
    beam = hypotheses[str(row['classifier_count'])] or row['independent_beam_peptides'][:]
    # Retain the established output ordering when the winning set is unchanged.
    if sorted(beam) == sorted(row['independent_beam_peptides']):
        beam = row['independent_beam_peptides'][:]
    result['expanded_beam_peptides'] = beam[:]
    result['beam_peptides'] = beam[:]
    ranked = beam[:]
    if policy is not None and candidates:
        if tuple(policy['features']) != FEATURES:
            raise ValueError('Ranker feature schema mismatch')
        scores = np.asarray(features, dtype=float) @ np.asarray(policy['weights'])
        ranked = list(candidates[int(np.argmax(scores))])
    # Reset the earlier ranking before apply_route so its idempotence support
    # cannot accidentally reuse a decision made on the independent pool.
    result['expanded_ranked_peptides'] = ranked[:]
    result['ranked_peptides'] = ranked[:]
    result['peptides'] = ranked[:]
    if policy is not None:
        result = apply_route(result, policy)
    result['predicted_count'] = len(result['peptides'])
    ended = bool(result['peptides']) and tuple(sorted(result['peptides'])) in unique
    result['count_constraint_satisfied'] = ended and len(result['peptides']) in (1, 2)
    result['ended_with_eos'] = ended
    result['truncated'] = not ended
    return result


@torch.no_grad()
def decode_diagnostics(model, spectra, precursors, beam_size=4, policy=None,
                       threshold=None, max_tokens=52, min_aa=6, length_penalty=1.):
    """Independent and anchored candidate sets, with no reference-label input."""
    if length_penalty != 1. or beam_size < 2:
        raise ValueError('Conditional candidate scoring requires beam >= 2 and alpha=1')
    if policy is not None and beam_size != 4:
        raise ValueError('The saved ranker was fitted for beam 4')
    model.eval()
    max_tokens = min(max_tokens, model.max_peptide_tokens)
    cutoff = model.threshold if threshold is None else threshold
    if not 0 < cutoff < 1:
        raise ValueError('Invalid classifier threshold')
    memory, mask = model.memories(spectra)
    probabilities = model.classify_memory(memory, mask).sigmoid().cpu().tolist()
    decode_args = dict(beam_size=beam_size, max_tokens=max_tokens, min_aa=min_aa,
                       length_penalty=length_penalty, return_candidates=True)
    slots = [generate_beam(_SlotView(model, memory, mask, slot), spectra, precursors,
                           **decode_args) for slot in (0, 1)]
    xs, ps = spectra.float().cpu().numpy(), precursors.float().cpu().numpy()
    rows = []
    for i, probability in enumerate(probabilities):
        pools = [slot[i] for slot in slots]
        row = rank_spectrum(model, xs[i], float(ps[i, 0]), probability, pools, cutoff,
                            policy=policy, return_features=True)
        if policy is None:
            row['peptides'] = row['beam_peptides'][:]
        row['independent_ranked_peptides'] = row['peptides'][:]
        if policy is not None:
            row = apply_route(row, policy)
        row['independent_peptides'] = row['peptides'][:]
        row['independent_beam_peptides'] = row['beam_peptides'][:]
        row['classifier_count'] = 1 + int(probability >= cutoff)
        row['hypotheses_by_count'] = {}
        for count in (1, 2):
            choices = [(s, f) for s, f in zip(row['feature_sets'], row['features']) if len(s) == count]
            row['hypotheses_by_count'][str(count)] = list(max(choices, key=lambda x: x[1][2])[0]) if choices else []
        row['slot_candidates'] = [[dict(peptide=c['peptides'][0], beam_score=c['beam_score'])
                                   for c in pool.get('candidates', [])] for pool in pools]
        row['conditional_branches'] = []
        row['candidate_peptide_union'] = sorted({p for candidate in row['candidate_sets'] for p in candidate})
        rows.append(row)
    if model.conditional_enabled:
        # One anchor from each independent slot. Single-output hypotheses are
        # preserved; conditional recovery is a two-peptide hypothesis only.
        owners, anchors, anchor_candidates, anchor_slots = [], [], [], []
        for i, row in enumerate(rows):
            if len(row['independent_peptides']) != 2:
                continue
            seen = set()
            for slot_index, slot in enumerate(slots):
                pool = slot[i].get('candidates', [])
                if pool and pool[0]['peptides'][0] not in seen:
                    anchor = pool[0]['peptides'][0]
                    owners.append(i); anchors.append(anchor); anchor_candidates.append(pool[0])
                    anchor_slots.append(slot_index)
                    seen.add(anchor)
        if owners:
            ids = torch.tensor(owners, device=spectra.device)
            # Limit residual beam batches to the caller's spectral batch size.
            for begin in range(0, len(ids), max(1, len(spectra))):
                end = min(len(ids), begin + max(1, len(spectra)))
                selected = ids[begin:end]
                cmemory, cmask = model.conditional_memory(spectra[selected], anchors[begin:end])
                decoded = generate_beam(_SlotView(model, cmemory, cmask, 1),
                                        spectra[selected], precursors[selected], **decode_args)
                # Batch independent re-scoring instead of issuing one tiny GPU
                # forward for every anchor. Row ownership remains explicit.
                rescored = {}
                for target_slot in (0, 1):
                    entries = [(j, c) for j, rem in enumerate(decoded, begin)
                               if 1-anchor_slots[j] == target_slot
                               for c in rem.get('candidates', [])]
                    for offset in range(0, len(entries), 32):
                        part = entries[offset:offset+32]
                        repeated = torch.tensor([owners[j] for j, _ in part],
                                                device=spectra.device, dtype=torch.long)
                        scores = -model.slot_loss([c['peptides'][0] for _, c in part],
                            precursors[repeated], memory[repeated], mask[repeated], target_slot)
                        for (j, candidate), score in zip(part, scores):
                            rescored.setdefault(j, []).append(dict(candidate,
                                conditional_beam_score=candidate['beam_score'], beam_score=float(score)))
                for j, remaining in enumerate(decoded, begin):
                    i, anchor = owners[j], anchors[j]
                    candidates = remaining.get('candidates', [])
                    if candidates:
                        # Conditional likelihoods are not calibrated against
                        # independent likelihoods. Re-score each generated
                        # completion under the opposite original slot, on the
                        # original spectrum, before using the frozen ranker.
                        remaining = dict(remaining, candidates=rescored[j])
                    pools = [dict(candidates=[anchor_candidates[j]], peptides=[anchor]), remaining]
                    extra = rank_spectrum(model, xs[i], float(ps[i, 0]), probabilities[i], pools,
                                          cutoff, policy=None, return_features=True)
                    for candidate, features in zip(extra['feature_sets'], extra['features']):
                        if len(candidate) == 2:
                            rows[i]['candidate_sets'].append(candidate)
                            rows[i]['feature_sets'].append(candidate)
                            rows[i]['features'].append(features)
                    rows[i]['conditional_branches'].append(dict(anchor=anchor,
                        candidates=remaining.get('candidates', [])))
    return [select_expanded_candidates(row, policy) for row in rows]
