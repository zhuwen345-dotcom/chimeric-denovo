"""Finite slot-specific priors in decoder cross-attention.

Keep the original encoder features as keys/values. A learned peak preference
adds to attention logits instead of rescaling the feature vector. This is a
static spectrum-conditioned prior, not peptide-conditioned physical separation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from dual_slot import EvidenceDualSlotModel


class AttentionEvidenceDualSlotModel(EvidenceDualSlotModel):
    architecture = 'dual_slot_attention'

    def __init__(self, *args, attention_initial_strength=.1, attention_max_bias=4., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(attention_initial_strength) or attention_initial_strength <= 0:
            raise ValueError('Attention strength must be positive and finite.')
        if not math.isfinite(attention_max_bias) or attention_max_bias <= 0:
            raise ValueError('Attention bias bound must be positive and finite.')
        self.peak_attention_log_scale = nn.Parameter(torch.tensor(
            attention_initial_strength + math.log(-math.expm1(-attention_initial_strength))))
        # Persistent in best/last: independent inference uses the exact route
        # reached in training. New stages start identically to the old model.
        self.register_buffer('peak_attention_progress', torch.tensor(0.))
        self._attention_progress = 0.
        self.attention_max_bias = float(attention_max_bias)

    def set_attention_progress(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid attention transition progress.')
        self.peak_attention_progress.fill_(value)
        self._attention_progress = value

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self._attention_progress = float(self.peak_attention_progress)
        if not 0 <= self._attention_progress <= 1:
            raise ValueError('Invalid saved attention transition progress.')
        return result

    def peak_attention_bias(self, memory, mask, slot, target_length):
        logits = self.peak_evidence(memory[:, 1:])[..., slot].float()
        # Zero evidence logits give exactly zero prior, including relative to
        # the unmodified global token. Finite priors never hard-delete a peak.
        peak_bias = (self.peak_attention_progress * F.softplus(self.peak_attention_log_scale)
                     * (F.logsigmoid(logits) + math.log(2.)))
        peak_bias = peak_bias.clamp(-self.attention_max_bias, self.attention_max_bias)
        bias = torch.cat((peak_bias.new_zeros((len(memory), 1)), peak_bias), dim=1)
        # Fold padding into this float mask rather than mixing float/boolean
        # memory masks. Padding is the only source of negative infinity.
        bias = bias.masked_fill(mask, float('-inf'))
        heads = self.decoder.transformer_decoder.layers[0].multihead_attn.num_heads
        # Materialize B*heads*S only; query positions use a stride-zero view.
        return bias[:, None, :].expand(-1, heads, -1).reshape(
            len(memory)*heads, 1, memory.shape[1]).expand(-1, target_length, -1)

    def slot_forward(self, sequences, precursors, memory, mask, slot):
        if self._attention_progress == 0:
            # Exact historical path, not just similar outputs at initialization.
            return EvidenceDualSlotModel.slot_forward(self, sequences, precursors, memory, mask, slot)
        progress = self.peak_attention_progress
        legacy_weight = .5 + self.peak_evidence(memory[:, 1:])[..., slot].sigmoid()
        weights = 1 + (1-progress)*(legacy_weight-1)
        conditioned = torch.cat((memory[:, :1], memory[:, 1:]*weights[..., None]), dim=1)
        decoder = self.decoder
        tokens = (None if sequences is None else nn.utils.rnn.pad_sequence(
            [decoder.tokenize(s) for s in sequences], batch_first=True))
        mass = decoder.mass_encoder(precursors[:, None, 0])
        charge = decoder.charge_encoder(precursors[:, 1].int()-1)[:, None, :]
        start = (mass*self.precursor_scales[slot, 0] + charge*self.precursor_scales[slot, 1]
                 + self.slot_embeddings[slot][None, None, :])
        target = start if tokens is None else torch.cat((start, decoder.aa_encoder(tokens)), dim=1)
        padding = torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
        if tokens is not None:
            padding[:, 1:] = tokens == 0
        target = decoder.pos_encoder(target)
        causal = torch.triu(torch.ones(target.shape[1], target.shape[1],
            dtype=torch.bool, device=target.device), diagonal=1)
        bias = self.peak_attention_bias(memory, mask, slot, target.shape[1])
        hidden = decoder.transformer_decoder(tgt=target, memory=conditioned, tgt_mask=causal,
            tgt_key_padding_mask=padding, memory_mask=bias, memory_key_padding_mask=None)
        return decoder.final(hidden), tokens
