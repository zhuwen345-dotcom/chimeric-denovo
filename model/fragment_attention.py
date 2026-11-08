"""Score next residues against observed suffix fragments during training and beam search."""
import torch
from torch import nn
from torch.nn.functional import pad

from peak_attention import AttentionEvidenceDualSlotModel


class FragmentAttentionModel(AttentionEvidenceDualSlotModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.decoder.reverse:
            raise ValueError("Suffix ions require C-to-N decoding")
        masses = self.decoder._peptide_mass.masses
        vocab = self.decoder.final.out_features
        values = torch.zeros(vocab)
        residues = torch.zeros(vocab, dtype=torch.bool)
        for idx, token in self.decoder._idx2aa.items():
            if token in masses:
                values[idx] = masses[token]
                residues[idx] = token[0].isalpha()
        self.register_buffer("token_mass", values, persistent=False)
        self.register_buffer("is_residue", residues, persistent=False)
        # Per slot: y1 hit, y1 intensity, y2 hit, y2 intensity.
        self.fragment_weights = nn.Parameter(torch.zeros(2, 4))
        self.output_policy = None

    @torch.no_grad()
    def generate(self, spectra, precursors, max_tokens=120, min_aa=6,
                 routing=None, threshold=None, beam_size=None, length_penalty=1.):
        width = self.evaluation_beam_size if beam_size is None else beam_size
        if self.output_policy is None or width == 1:
            return super().generate(spectra, precursors, max_tokens, min_aa,
                                    routing, threshold, width, length_penalty)
        if width != 4 or length_penalty != 1. or routing not in (None, "hard"):
            raise ValueError("The saved candidate policy was fitted for beam 4 with length penalty 1")
        from evidence_decode import evidence_generate
        return evidence_generate(self, spectra, precursors, beam_size=width,
                                 threshold=threshold, max_tokens=max_tokens, min_aa=min_aa,
                                 policy=self.output_policy)

    def memories(self, spectra):
        memory, mask = super().memories(spectra)
        # Store observed peaks alongside embeddings so sub-batching and beam
        # expansion always preserve row identities, including both PIT orders.
        peaks = pad(spectra.float(), (0, 0, 1, 0))
        return torch.cat((memory.float(), peaks), dim=-1), mask

    def classify_memory(self, memory, mask):
        return super().classify_memory(memory[..., :-2], mask)

    def fragment_bias(self, tokens, peaks, slot):
        # Position t sees exactly t emitted tokens. A candidate supplies the
        # last mass in the hypothetical suffix; future reference tokens never
        # contribute to that position.
        mz, intensity = peaks.float().unbind(dim=-1)
        mz, order = mz.masked_fill(intensity <= 0, torch.inf).sort(dim=1)
        intensity = intensity.gather(1, order).clamp_min(0)
        intensity = intensity / intensity.amax(1, keepdim=True).clamp_min(1e-8)
        if tokens is None:
            prefix = mz.new_zeros((len(mz), 1))
        else:
            prefix = pad(self.token_mass[tokens].cumsum(1), (1, 0))
        suffix = prefix[:, :, None] + self.token_mass + 18.010565
        weights = self.fragment_weights[slot]
        score = 0
        for z in (1, 2):
            query = (suffix / z + 1.0072764666).flatten(1).contiguous()
            insert = torch.searchsorted(mz, query)
            tol = (query * 20e-6).clamp_min(.02)
            hit = torch.zeros_like(query)
            amplitude = torch.zeros_like(query)
            for positions in ((insert - 1).clamp_min(0), insert.clamp_max(mz.shape[1] - 1)):
                distance = (mz.gather(1, positions) - query).abs()
                match = (1 - distance / tol).clamp_min(0)
                hit = torch.maximum(hit, match)
                amplitude = torch.maximum(amplitude, match * intensity.gather(1, positions))
            score = score + (weights[2*z-2] * hit + weights[2*z-1] * amplitude).reshape_as(suffix)
        return score * self.is_residue

    def slot_forward(self, sequences, precursors, memory, mask, slot):
        logits, tokens = super().slot_forward(sequences, precursors, memory[..., :-2], mask, slot)
        return logits.float() + self.fragment_bias(tokens, memory[:, 1:, -2:], slot), tokens
