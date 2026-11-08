"""Sequence-only fragment intensities and additive single/two-peptide fitting.

Independent implementation inspired by spectral mixture search, not CHIMERYS
or Sage source. Inputs are retained sqrt/L2 MS2 peaks. Squaring recovers their
relative *linear* intensities before adding peptide contributions. No observed
intensity enters the peptide-to-fragment predictor itself.
"""
from dataclasses import dataclass
from itertools import combinations, product
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

PROTON, WATER = 1.007276466621, 18.010565
AA = dict(zip('ACDEFGHIKLMNPQRSTVWY', (
    71.037114,103.009185,115.026943,129.042593,147.068414,
    57.021464,137.058912,113.084064,128.094963,113.084064,
    131.040485,114.042927,97.052764,128.058578,156.101111,
    87.032028,101.047679,99.068414,186.079313,163.063329)))
IDS = {a:i+1 for i,a in enumerate(AA)}


def parse_peptide(sequence):
    sequence = sequence.replace('I','L')
    tokens = re.findall(r'^[+-]\d+(?:\.\d+)?|[A-Z](?:[+-]\d+(?:\.\d+)?)?', sequence)
    if ''.join(tokens) != sequence:
        raise ValueError('Unsupported peptide spelling: '+sequence)
    terminal, residues, modifications = 0., [], []
    for token in tokens:
        if token[0] in '+-':
            terminal += float(token)
        else:
            if token[0] not in AA:
                raise ValueError('Unsupported residue: '+token)
            residues.append(token[0]);modifications.append(float(token[1:]) if len(token)>1 else 0.)
    if not 2 <= len(residues) <= 128 or not all(math.isfinite(x) and abs(x)<=500 for x in modifications+[terminal]):
        raise ValueError('Unsupported peptide length/modification.')
    if any(AA[a]+m<=0 for a,m in zip(residues,modifications)):
        raise ValueError('Nonphysical modified residue mass.')
    return residues, modifications, terminal


def peptide_mass(sequence):
    residues, mods, terminal = parse_peptide(sequence)
    return sum(AA[a]+m for a,m in zip(residues,mods))+terminal+WATER


def fragment_batch(sequences, charges, device):
    parsed=[parse_peptide(s) for s in sequences]
    if len(parsed)!=len(charges) or not parsed:
        raise ValueError('Nonempty sequence/charge lists must have equal length.')
    width=max(len(x[0]) for x in parsed)
    ids=torch.zeros(len(parsed),width,dtype=torch.long,device=device)
    mods=torch.zeros(len(parsed),width,device=device)
    masses=torch.zeros_like(mods);terminal=torch.zeros(len(parsed),device=device)
    lengths=torch.tensor([len(x[0]) for x in parsed],device=device)
    for i,(residues,delta,nterm) in enumerate(parsed):
        ids[i,:len(residues)]=torch.tensor([IDS[a] for a in residues],device=device)
        mods[i,:len(residues)]=torch.tensor(delta,device=device)
        masses[i,:len(residues)]=torch.tensor([AA[a]+m for a,m in zip(residues,delta)],device=device)
        terminal[i]=nterm
    z=torch.as_tensor(charges,dtype=torch.float32,device=device)
    if not bool(((z>=1)&(z<=10)&(z==z.round())).all()):
        raise ValueError('Charges must be integers in [1,10].')
    prefix=masses.cumsum(1)[:,:-1]
    b=prefix+terminal[:,None];y=masses.sum(1)[:,None]-prefix+WATER
    mz=torch.stack((b+PROTON,b/2+PROTON,y+PROTON,y/2+PROTON),-1)
    valid=torch.arange(width-1,device=device)[None,:] < lengths[:,None]-1
    ion_valid=valid[:,:,None].expand_as(mz).clone()
    ion_valid[:,:,1] &= z[:,None]>=2;ion_valid[:,:,3] &= z[:,None]>=2
    return ids, mods, terminal, z, mz, ion_valid


class FragmentIntensityModel(nn.Module):
    """Small trainable b1/b2/y1/y2 predictor; not pretrained AlphaPeptDeep.

    Only sequence, modifications and candidate charge are inputs. Missing
    instrument/energy metadata is an explicit limitation, not inferred truth.
    """
    def __init__(self, width=96, layers=2):
        super().__init__()
        self.config=dict(width=width,layers=layers)
        self.embedding=nn.Embedding(len(IDS)+1,width,padding_idx=0)
        self.position=nn.Embedding(128,width)
        self.condition=nn.Linear(3,width)
        layer=nn.TransformerEncoderLayer(width,4,width*3,.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),nn.Linear(width,4))

    def forward(self,sequences,charges):
        ids,mods,terminal,z,mz,valid=fragment_batch(sequences,charges,self.embedding.weight.device)
        features=torch.stack((mods/100,terminal[:,None].expand_as(mods)/100,z[:,None].expand_as(mods)/5),-1)
        h=self.embedding(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))[None]+self.condition(features)
        h=self.encoder(h,src_key_padding_mask=ids==0)
        pooled=(h*(ids!=0)[:,:,None]).sum(1)/(ids!=0).sum(1)[:,None]
        hidden=torch.cat((h[:,:-1],h[:,1:],pooled[:,None].expand(-1,h.shape[1]-1,-1)),-1)
        intensity=F.softplus(self.head(hidden).float())*valid
        return mz.flatten(1),intensity.flatten(1),valid.flatten(1)


def linear_observation(spectra):
    if not bool(torch.isfinite(spectra).all()) or bool((spectra[...,1]<0).any()):
        raise ValueError('Spectrum contains invalid intensities/masses.')
    valid=(spectra[...,0]>0)&(spectra[...,1]>0)&torch.isfinite(spectra).all(-1)
    # sqrt(I)/||sqrt(I)|| -> I/sum(I); do not add sqrt-space intensities.
    observed=spectra[...,1].float().square()*valid
    if not bool((observed.sum(1)>0).all()):raise ValueError('Empty/nonfinite spectrum.')
    return observed/observed.sum(1,keepdim=True),valid


def project_fragments(spectra,ion_mz,ion_intensity,ion_valid,ppm=20.,absolute=.02):
    """Project theoretical ions to retained peaks, sharing a single ion's mass
    across nearby peaks instead of duplicating its intensity. No intensity
    values are consulted to choose the mapping.
    """
    _,valid=linear_observation(spectra)
    delta=(spectra[:,:,0,None].float()-ion_mz[:,None,:].float()).abs()
    tolerance=(ion_mz.abs()*ppm*1e-6).clamp_min(absolute)
    weights=(1-delta/tolerance[:,None,:]).clamp_min(0)*ion_valid[:,None,:]*valid[:,:,None]
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    projected=(weights*ion_intensity[:,None,:].float()).sum(-1)
    support=(weights>0).any(-1)
    return projected, support


def fit_nonnegative(templates,observed,valid,ridge=1e-5,noise_penalty=.02):
    """Exact active-set nonnegative ridge fit for <=2 peptides + flat noise.

    templates [B,P,K], K=1 or 2; shared peaks are sums, never forced exclusive.
    The only background freedom is one uniform profile with a penalized scalar
    amplitude. An arbitrary per-peak background would make fitting tautological.
    """
    if templates.ndim!=3 or templates.shape[:2]!=observed.shape or templates.shape[2] not in (1,2):
        raise ValueError('Expected one or two templates per spectrum.')
    if ridge<=0 or noise_penalty<0 or not bool(valid.any(1).all()):raise ValueError('Invalid fit settings.')
    with torch.autocast(device_type=templates.device.type,enabled=False):
        profiles=templates.float()*valid[:,:,None]
        profiles=profiles/profiles.sum(1,keepdim=True).clamp_min(1e-12)
        noise=valid.float()/valid.sum(1,keepdim=True)
        matrix=torch.cat((profiles,noise[:,:,None]),-1)
        y=observed.float()*valid
        batch,_,columns=matrix.shape
        gram=matrix.transpose(1,2)@matrix
        diagonal=matrix.new_full((columns,),ridge);diagonal[-1]+=noise_penalty
        gram=gram+torch.diag(diagonal)[None]
        rhs=(matrix.transpose(1,2)@y[:,:,None]).squeeze(-1)
        best=y.square().sum(1);coeff=matrix.new_zeros(batch,columns)
        for size in range(1,columns+1):
            for subset in combinations(range(columns),size):
                ids=torch.tensor(subset,device=matrix.device)
                a=torch.linalg.solve(gram[:,ids][:,:,ids],rhs[:,ids,None]).squeeze(-1)
                feasible=(a>=0).all(1)
                candidate=torch.zeros_like(coeff);candidate[:,ids]=a.clamp_min(0)
                prediction=(matrix*candidate[:,None]).sum(-1)
                cost=(prediction-y).square().sum(1)+(candidate.square()*diagonal).sum(1)
                better=feasible&(cost<best)
                best=torch.where(better,cost,best);coeff=torch.where(better[:,None],candidate,coeff)
        prediction=(matrix*coeff[:,None]).sum(-1)
        error=(prediction-y).square().sum(1)/y.square().sum(1).clamp_min(1e-12)
        penalized=error+(coeff.square()*diagonal).sum(1)/y.square().sum(1).clamp_min(1e-12)
        return dict(error=error,penalized_error=penalized,coefficients=coeff,reconstructed=prediction,
                    peptide_contributions=profiles*coeff[:,:-1,None].transpose(1,2),
                    noise_fraction=coeff[:,-1])


def intensity_training_loss(model,spectra,sequences,charges):
    mz,intensity,valid=model(sequences,charges)
    predicted,support=project_fragments(spectra,mz,intensity,valid)
    observed,peaks=linear_observation(spectra)
    target=observed*support
    eligible=support.sum(1)>=6
    cosine=F.cosine_similarity(predicted.clamp_min(1e-12).sqrt(),target.clamp_min(1e-12).sqrt(),dim=1)
    # Uniform-ion spectra provide a concrete baseline on this same filtered set.
    uniform,_=project_fragments(spectra,mz,valid.float(),valid)
    baseline=F.cosine_similarity(uniform.clamp_min(1e-12).sqrt(),target.clamp_min(1e-12).sqrt(),dim=1)
    return 1-cosine,dict(cosine=cosine,uniform_cosine=baseline,eligible=eligible)


@dataclass(frozen=True)
class Hypothesis:
    peptides: tuple

    def __post_init__(self):
        canonical=tuple(sorted({p.replace('I','L') for p in self.peptides}))
        if len(canonical)!=len(self.peptides) or len(canonical) not in (1,2):
            raise ValueError('Hypotheses contain one or two distinct peptides.')
        for p in canonical:parse_peptide(p)
        object.__setattr__(self,'peptides',canonical)


class MixtureObjective(nn.Module):
    """Frozen intensity teacher + trainable physical-score weight.

    Costs use only a candidate sequence and observed peaks. Reference labels
    enter ranking supervision exclusively, never fitting or inference scoring.
    """
    def __init__(self,intensity_model,charges=(2,3,4,5),initial_weight=1.,complexity_penalty=.02):
        super().__init__()
        if not charges or any(int(z)!=z or not 1<=z<=10 for z in charges):raise ValueError('Invalid candidate charges.')
        if initial_weight<=0 or complexity_penalty<0:raise ValueError('Invalid score settings.')
        self.intensity=intensity_model.eval().requires_grad_(False)
        self.charges=tuple(charges);self.complexity_penalty=complexity_penalty
        self.raw_weight=nn.Parameter(torch.tensor(math.log(math.expm1(initial_weight))))

    def train(self,mode=True):
        super().train(mode);self.intensity.eval();return self

    @torch.no_grad()
    def reconstruction_costs(self,spectrum,hypotheses):
        if spectrum.ndim!=3 or len(spectrum)!=1:raise ValueError('One observed spectrum is required.')
        if not hypotheses:return []
        peptides=sorted({p for h in hypotheses for p in h.peptides})
        seq=[p for p in peptides for _ in self.charges];z=list(self.charges)*len(peptides)
        with torch.autocast(device_type=spectrum.device.type,enabled=False):
            mz,intensity,valid=self.intensity(seq,z)
        templates,_=project_fragments(spectrum.expand(len(seq),-1,-1),mz,intensity,valid)
        lookup={p:templates[i*len(self.charges):(i+1)*len(self.charges)] for i,p in enumerate(peptides)}
        observation,peaks=linear_observation(spectrum)
        results=[]
        for h in hypotheses:
            combinations_z=list(product(range(len(self.charges)),repeat=len(h.peptides)))
            profiles=torch.stack([torch.stack([lookup[p][j] for p,j in zip(h.peptides,choice)],-1) for choice in combinations_z])
            fit=fit_nonnegative(profiles,observation.expand(len(profiles),-1),peaks.expand(len(profiles),-1))
            error=fit['penalized_error']+self.complexity_penalty*(len(h.peptides)-1)
            choice=int(error.argmin())
            results.append(dict(cost=error[choice],charges=[self.charges[j] for j in combinations_z[choice]],
                coefficients=fit['coefficients'][choice],noise_fraction=fit['noise_fraction'][choice],
                reconstructed=fit['reconstructed'][choice]))
        return results

    def scores(self,sequence_scores,count_logit,costs,hypotheses):
        is_double=sequence_scores.new_tensor([len(h.peptides)==2 for h in hypotheses])
        log_count=-F.binary_cross_entropy_with_logits(count_logit.expand_as(is_double),is_double,reduction='none')
        return sequence_scores+log_count-F.softplus(self.raw_weight)*costs.detach()

    def supervised_loss(self,sequence_scores,count_logit,costs,hypotheses,reference,risk_weight=.1):
        expected=Hypothesis(tuple(reference))
        positive=[i for i,h in enumerate(hypotheses) if h==expected]
        if len(positive)!=1:raise ValueError('Training pool must contain exactly one positive reference set.')
        scores=self.scores(sequence_scores,count_logit,costs,hypotheses)
        rank=F.cross_entropy(scores[None],torch.tensor(positive,device=scores.device))
        risk=(scores.softmax(0)*costs.detach()).sum()
        # Risk is a candidate-level surrogate: gradients go through candidate
        # log-likelihoods/count prior, NOT through discrete generation or NNLS.
        return rank+risk_weight*risk,dict(rank=rank.detach(),risk=risk.detach())
