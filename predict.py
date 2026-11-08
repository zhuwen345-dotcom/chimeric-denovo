#!/usr/bin/env python3
"""Chimeric (single/double) peptide de novo sequencing from MS2 spectra.

Usage examples:
  python predict.py --input spectra.mgf --output predictions.csv
  python predict.py --input spectra.mgf --output pred.csv --beam-size 4
  python predict.py --input spectra.mgf --output pred.csv --use-rt \
      --rt-checkpoint checkpoints/rt_model.pt --rt-weight 1.0
  python predict.py --input spectra.mgf --output pred.csv --use-intensity \
      --intensity-checkpoint checkpoints/intensity.pt --intensity-weight 2.46

Only the MS2 spectrum and its measured precursor are required.  The optional
retention-time and fragment-intensity modules are re-ranking add-ons; they are
provided for completeness and ablation and were not required for the reported
final performance.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "model"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "backbone"))

PROTON = 1.007276466621
MAX_PEAKS = 800
MIN_MZ = 1.0
MAX_MZ = 6500.0
PRECURSOR_REMOVAL = 1.0


def parse_mgf(path):
    spectra = []
    current = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "BEGIN IONS":
            current = dict(peaks=[], id=None, pepmass=None, charge=None, rt=None)
            continue
        if current is None:
            continue
        if upper == "END IONS":
            if current["peaks"]:
                spectra.append(current)
            current = None
            continue
        if "=" in line and not line[0].isdigit():
            key, value = line.split("=", 1)
            key, value = key.strip().upper(), value.strip()
            if key == "TITLE":
                current["id"] = value
            elif key == "PEPMASS":
                current["pepmass"] = float(value.split()[0])
            elif key == "CHARGE":
                digits = re.search(r"(\d+)", value)
                current["charge"] = int(digits.group(1)) if digits else None
            elif key == "RTINSECONDS":
                current["rt"] = float(value.split()[0])
            elif key == "RTINMINUTES":
                current["rt"] = float(value.split()[0]) * 60.0
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                current["peaks"].append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return spectra


def load_jsonl(path):
    spectra = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            spectra.append(json.loads(line))
    return spectra


def preprocess_peaks(peaks, precursor_mz):
    peaks = np.asarray(peaks, dtype=float).reshape(-1, 2)
    valid = (peaks[:, 0] >= MIN_MZ) & (peaks[:, 0] <= MAX_MZ) & (peaks[:, 1] > 0)
    peaks = peaks[valid]
    peaks = peaks[np.abs(peaks[:, 0] - precursor_mz) > PRECURSOR_REMOVAL]
    if len(peaks) > MAX_PEAKS:
        keep = np.argsort(-peaks[:, 1])[:MAX_PEAKS]
        peaks = peaks[np.sort(keep)]
    peaks = peaks[np.argsort(peaks[:, 0])]
    intensity = np.sqrt(peaks[:, 1])
    norm = np.linalg.norm(intensity)
    if norm > 0:
        intensity = intensity / norm
    return np.stack((peaks[:, 0], intensity), axis=1)


def prepare_spectra(records):
    prepared = []
    for record in records:
        if "precursor" in record and record.get("preprocessed"):
            precursor = [float(v) for v in record["precursor"]]
            peaks = np.asarray(record["peaks"], dtype=float).reshape(-1, 2)
        else:
            charge = float(record["charge"])
            mono_mz = float(record["pepmass"])
            neutral = mono_mz * charge - charge * PROTON
            precursor = [neutral, charge, mono_mz]
            peaks = preprocess_peaks(record["peaks"], mono_mz)
        prepared.append(dict(
            id=str(record.get("id") or record.get("title") or len(prepared)),
            precursor=precursor,
            peaks=peaks,
            rt=record.get("rt"),
            rt_normalized=record.get("rt_normalized"),
        ))
    return prepared


class RTModel(nn.Module):
    """BiGRU peptide -> per-RAW percentile-normalized retention time."""

    def __init__(self, vocab_size, embedding=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.gru = nn.GRU(embedding, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden * 2 + 3, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, tokens, lengths, mod_counts, terminals):
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        mask = torch.arange(output.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        mean = (output * mask[..., None].float()).sum(1) / lengths[:, None].float()
        max_pooled = output.masked_fill(~mask[..., None], -1e9).max(1).values
        side = torch.stack((lengths.float(), mod_counts, terminals), dim=1)
        return self.head(torch.cat((mean, max_pooled, side), dim=1)).squeeze(-1)


def load_rt(path, device):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    model = RTModel(len(state["vocab"]) + 2)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["vocab"]


@torch.no_grad()
def predict_rt(model, vocab, peptides, device):
    rows = []
    for peptide in peptides:
        tokens, terminal, mods = [], 0.0, 0
        ok = True
        for token in re.findall(r"^[+-]\d+(?:\.\d+)?|[A-Z](?:[+-]\d+(?:\.\d+)?)?", peptide):
            if token[0] in "+-":
                terminal = float(token)
                continue
            if token not in vocab:
                ok = False
                break
            tokens.append(vocab[token])
            if len(token) > 1:
                mods += 1
        rows.append((tokens, terminal, mods) if ok and len(tokens) >= 4 else None)
    valid = [(i, r) for i, r in enumerate(rows) if r is not None]
    out = [None] * len(peptides)
    if not valid:
        return out
    length = max(len(r[0]) for _, r in valid)
    tokens = torch.zeros(len(valid), length, dtype=torch.long, device=device)
    lengths = torch.zeros(len(valid), dtype=torch.long, device=device)
    terminals = torch.zeros(len(valid), device=device)
    mods = torch.zeros(len(valid), device=device)
    for row, (_, (seq, terminal, count)) in enumerate(valid):
        tokens[row, :len(seq)] = torch.tensor(seq, device=device)
        lengths[row] = len(seq)
        terminals[row] = terminal
        mods[row] = count
    values = model(tokens, lengths, mods, terminals).cpu().tolist()
    for row, (index, _) in enumerate(valid):
        out[index] = float(values[row])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help=".mgf or preprocessed .jsonl spectra")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--checkpoint", default=str(REPO / "checkpoints/model.pt"))
    parser.add_argument("--beam-size", type=int, default=4,
                        help="beam width (the frozen ranker applies at beam 4)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="chimeric-probability cutoff for single vs double")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-rt", action="store_true",
                        help="re-rank candidates with the retention-time model")
    parser.add_argument("--rt-checkpoint", default=str(REPO / "checkpoints/rt_model.pt"))
    parser.add_argument("--rt-weight", type=float, default=1.0)
    parser.add_argument("--use-intensity", action="store_true",
                        help="re-rank candidates with the fragment-intensity mixture model")
    parser.add_argument("--intensity-checkpoint",
                        default=str(REPO / "checkpoints/intensity.pt"))
    parser.add_argument("--intensity-weight", type=float, default=2.46)
    args = parser.parse_args()
    if args.beam_size < 2:
        parser.error("--beam-size must be >= 2 (dual-slot decoding)")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must lie between 0 and 1")

    from train_spectral_mixture import load_trained_mixture
    from conditional_residual import decode_diagnostics
    from spectral_mixture import Hypothesis

    input_path = Path(args.input)
    records = parse_mgf(input_path) if input_path.suffix.lower() == ".mgf" else load_jsonl(input_path)
    spectra = prepare_spectra(records)
    if not spectra:
        raise SystemExit("No spectra found in " + str(input_path))

    device = torch.device(args.device)
    model, objective = load_trained_mixture(REPO, Path(args.checkpoint), MODEL_DIR, str(device))
    model.eval()

    rt_model = rt_vocab = None
    if args.use_rt:
        rt_model, rt_vocab = load_rt(args.rt_checkpoint, device)

    if args.use_intensity:
        state = torch.load(str(args.intensity_checkpoint), map_location="cpu", weights_only=False)
        objective.intensity.load_state_dict(state["intensity_state"])
        objective.intensity.eval()

    # Observation-normalized retention time for --use-rt: percentile scaling
    # estimated on the submitted batch (training used per-RAW 5/95 scaling).
    rt_grid = {}
    if args.use_rt:
        raw = [s["rt"] for s in spectra if s.get("rt") is not None]
        bounds = None
        if len(raw) >= 10:
            bounds = tuple(float(v) for v in np.percentile(np.asarray(raw), [5, 95]))
        for spectrum in spectra:
            if spectrum.get("rt_normalized") is not None:
                rt_grid[spectrum["id"]] = float(spectrum["rt_normalized"])
            elif spectrum.get("rt") is not None and bounds is not None:
                lo, high = bounds
                rt_grid[spectrum["id"]] = min(1.0, max(0.0, (spectrum["rt"] - lo) / max(1e-6, high - lo)))

    policy = model.output_policy if (model.output_policy is not None and args.beam_size == 4) else None
    weights = None if policy is None else np.asarray(policy["weights"], dtype=float)
    rows = []
    for begin in range(0, len(spectra), args.batch_size):
        batch = spectra[begin:begin + args.batch_size]
        max_len = max(len(s["peaks"]) for s in batch)
        peaks = np.zeros((len(batch), max_len, 2), dtype=np.float32)
        for row, spectrum in enumerate(batch):
            peaks[row, :len(spectrum["peaks"])] = spectrum["peaks"]
        tensor = torch.from_numpy(peaks).to(device)
        precursors = torch.tensor([s["precursor"] for s in batch], dtype=torch.float32, device=device)
        with torch.no_grad():
            decoded = decode_diagnostics(model, tensor, precursors, beam_size=args.beam_size,
                                         policy=policy, threshold=args.threshold, max_tokens=52)
        for spectrum, source_tensor, precursor_tensor, item in zip(batch, tensor, precursors, decoded):
            peptides = item["peptides"]
            rt_error = intensity_cost = None
            if args.use_rt and spectrum["id"] in rt_grid:
                unique = sorted({p.replace("I", "L") for s in item["feature_sets"] for p in s})
                predicted = predict_rt(rt_model, rt_vocab, unique, device)
                lookup = {p: v for p, v in zip(unique, predicted) if v is not None}
                errors = []
                for candidate in item["feature_sets"]:
                    values = [abs(lookup[p.replace("I", "L")] - rt_grid[spectrum["id"]])
                              for p in candidate if p.replace("I", "L") in lookup]
                    errors.append(float(np.mean(values)) if values else 0.5)
                rt_error = errors
            if args.use_intensity:
                hypotheses = [Hypothesis(tuple(candidate)) for candidate in item["feature_sets"]]
                if hypotheses:
                    fits = objective.reconstruction_costs(
                        source_tensor.unsqueeze(0), hypotheses)
                    intensity_cost = [float(fit["cost"]) for fit in fits]
            if weights is not None and (rt_error is not None or intensity_cost is not None):
                features = np.asarray(item["features"], dtype=float)
                scores = features @ weights
                if rt_error is not None:
                    scores = scores - args.rt_weight * np.asarray(rt_error)
                if intensity_cost is not None:
                    scores = scores - args.intensity_weight * np.asarray(intensity_cost)
                count = int(item["classifier_count"])
                ids = [i for i, s in enumerate(item["feature_sets"]) if len(s) == count]
                if ids:
                    peptides = item["feature_sets"][ids[int(np.argmax(scores[ids]))]]
            rows.append(dict(
                id=spectrum["id"],
                peptides=";".join(sorted(peptides)),
                n_peptides=len(peptides),
                chimeric_probability=round(float(item["chimeric_probability"]), 6),
                rt_error=(round(float(np.mean(rt_error)), 6) if rt_error is not None else ""),
                reconstruction_cost=(round(float(np.min(intensity_cost)), 6)
                                     if intensity_cost is not None else ""),
            ))
        print(json.dumps(dict(done=min(begin + args.batch_size, len(spectra)),
                              total=len(spectra))), flush=True)

    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.output, len(rows), "spectra")


if __name__ == "__main__":
    main()
