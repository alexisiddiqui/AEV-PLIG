"""Siamese difference head (arm S), pair head and Zernike-moments heads (arm P) on AEV-PLIG's shared encoder.

The clone's GATv2Net is split, without copying any module, into
  encoder   GNN_layers + BN_layers, then cat(global max pool, global mean pool) = z [B, 2 * hidden * head]
  affinity  fc1 / bn_connect1 / fc2 / bn_connect2 / fc3 / bn_connect3 / out (the pointwise MLP)
and AffinityWithSiamese(gnn, siamese, pair, moments) adds optional heads on z: a SiameseHead (arm S), a PairHead
and MomentsHeads (arm P). forward(data) is the pointwise prediction (exactly GATv2Net.forward), so every code path
that calls model(data) is unchanged.

SiameseHead: m(z_i, z_j) = MLP([z_i - z_j, z_i * z_j]) -> scalar, two hidden layers (Linear, activation,
BatchNorm1d, the clone's MLP order) and delta_hat(z_i, z_j) = (m(z_i, z_j) - m(z_j, z_i)) / 2, so
delta_hat is antisymmetric. Both orders go through the MLP as ONE batch: BatchNorm then sees the same rows
for (i, j) and (j, i) (antisymmetry holds in train mode too), a single pair is a batch of two (train mode
never raises), and eval mode uses the running statistics (any batch size, including one pair).

PairHead (arm P, the GQT aux_redge head): l(z_a, z_c) = MLP([z_a * z_c, |z_a - z_c|, z_a - z_c]) -> scalar, two
hidden layers (Linear, activation, LayerNorm). The signed difference block makes l asymmetric (l(a, c) != l(c, a)),
which "c is more potent than a" needs. LayerNorm, not BatchNorm: rows are independent, so any number of pairs
(one included) behaves the same in train and eval mode.

MomentsHead (arm P): one hidden layer (Linear, activation, LayerNorm), then Linear -> [n, out]; regresses the
standardised pocket or ligand 3D Zernike invariants F (7 channels x 49 = 343) from z. The wrapper holds them in a
ModuleDict `moments` keyed "pocket" / "ligand".

Moments as INPUT (arm Z): MomentsInput(d_in, proj) maps the standardised descriptor vector data.moments [B, d_in]
through Linear(d_in, proj) -> activation -> LayerNorm (proj 0: the raw vector), and the wrapper (moments_in=...)
rebuilds the affinity MLP's fc1 with input width D + proj (D = the pooled embedding width) so the pointwise head reads
cat(z, MomentsInput(moments)); embed(data) is still z alone and every head on z (Siamese, pair, moments) reads z
only. The descriptor vector of an --input-moments mode (moments_input_matrix): pocket = the pocket F (343), ligand =
the ligand F (343), both = pocket then ligand, bothx = pocket, ligand, then the cross block (the elementwise product
of the pocket complement, channels pos<->neg and donor<->acceptor swapped, with the ligand vector, 343), all in the
side file's standardised units, then ONE missing flag (1 when a block the mode uses has no descriptor; that block is
zeros, the cross block is zeros when either side is missing): d_in = 343 * k + 1. MomentsOnlyNet(d_in) is the
no-graph baseline: an MLP d_in -> 512 -> 256 -> 1 (Linear, activation, BatchNorm1d as the clone's MLP) whose
forward(data) reads data.moments only.

State dicts: the wrapper's keys are encoder.<GATv2Net key>, affinity.<GATv2Net key>, siamese.<head key>,
pair.<head key>, moments.<pocket|ligand>.<head key> and moments_in.<key>; an absent head has no keys (a Siamese-only
wrapper saves exactly as in arm S, a wrapper without moments_in exactly as in arm P). With moments_in, affinity.fc1
is the enlarged layer and moments_in.fc / moments_in.ln hold the projection (none when proj is 0). plain_to_wrapped
/ wrapped_to_plain rename a plain GATv2Net state dict to and from the wrapper's keys (deterministic, order kept,
unknown keys are an error; wrapped_to_plain drops every head and refuses moments_in keys: an enlarged fc1 is not a
GATv2Net layer); AffinityWithSiamese.from_plain_state_dict / to_plain_state_dict apply them to a model.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

from collections import OrderedDict

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from model_defs import activation_function_dict

ENCODER_KEYS = ("GNN_layers", "BN_layers")
AFFINITY_KEYS = ("fc1", "bn_connect1", "fc2", "bn_connect2", "fc3", "bn_connect3", "out")
ENCODER, AFFINITY, SIAMESE = "encoder", "affinity", "siamese"
PAIR, MOMENTS = "pair", "moments"
MOMENTS_IN = "moments_in"
HEAD_PREFIXES = (SIAMESE, PAIR, MOMENTS)   # optional heads: dropped by wrapped_to_plain
MOMENTS_TARGETS = ("pocket", "ligand")
MOMENTS_DIM = 343       # 7 channels x 49 invariants (order 12) of pocket_F / ligand_F
MOMENTS_HIDDEN = 512
N_ZERNIKE_CHANNELS = 7  # pocket_fields.CHANNELS: occupancy, hydrophobicity, pos, neg, aromatic, donor, acceptor
# the pocket channel that faces each channel of the ligand (twin_select's complementarity): pos <-> neg (2, 3) and
# donor <-> acceptor (5, 6) swapped, occupancy / hydrophobicity / aromatic kept
COMPLEMENT_PERM = (0, 1, 3, 2, 4, 6, 5)
INPUT_MOMENTS_BLOCKS = {"none": (), "pocket": ("pocket",), "ligand": ("ligand",), "both": ("pocket", "ligand"),
                        "bothx": ("pocket", "ligand", "cross")}
INPUT_MOMENTS_CHOICES = tuple(INPUT_MOMENTS_BLOCKS)
INPUT_MOMENTS_PROJ = 256
MOMENTS_ONLY_HIDDEN = (512, 256)


def embed_dim(gnn):
    """Width of the pooled embedding z of a GATv2Net (2 * hidden_dim * head): from the last GATv2Conv (heads x
    out_channels, concatenated), else fc1's input width (which a moments-input wrapper enlarges)."""
    last = gnn.GNN_layers[-1] if len(getattr(gnn, "GNN_layers", ())) else None
    heads_, out = getattr(last, "heads", None), getattr(last, "out_channels", None)
    if isinstance(heads_, int) and isinstance(out, int) and getattr(last, "concat", True):
        return 2 * heads_ * out
    return int(gnn.fc1.in_features)


class Encoder(nn.Module):
    """GATv2Net's message-passing half: GATv2Conv -> activation -> BatchNorm per layer, then
    cat(max pool, mean pool). Shares the GATv2Net's modules (no copy)."""

    def __init__(self, gnn):
        super().__init__()
        self.GNN_layers = gnn.GNN_layers
        self.BN_layers = gnn.BN_layers
        self.activation = gnn.activation
        self.dim = embed_dim(gnn)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)
        return torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


class AffinityMLP(nn.Module):
    """GATv2Net's pointwise MLP on z: fc -> activation -> BatchNorm1d (x3), then out -> [B, 1].
    Shares the GATv2Net's modules (no copy)."""

    def __init__(self, gnn):
        super().__init__()
        for name in AFFINITY_KEYS:
            setattr(self, name, getattr(gnn, name))
        self.activation = gnn.activation

    def forward(self, z):
        x = self.bn_connect1(self.activation(self.fc1(z)))
        x = self.bn_connect2(self.activation(self.fc2(x)))
        x = self.bn_connect3(self.activation(self.fc3(x)))
        return self.out(x)


class SiameseHead(nn.Module):
    """delta_hat(z_i, z_j) = (m(z_i, z_j) - m(z_j, z_i)) / 2 with m = MLP([z_i - z_j, z_i * z_j])."""

    def __init__(self, dim, hidden=512, act="leaky_relu"):
        super().__init__()
        self.dim, self.hidden, self.act = int(dim), int(hidden), act
        self.activation = activation_function_dict[act]
        self.fc1 = nn.Linear(2 * self.dim, self.hidden)
        self.bn1 = nn.BatchNorm1d(self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.bn2 = nn.BatchNorm1d(self.hidden)
        self.out = nn.Linear(self.hidden, 1)

    def m(self, zi, zj):
        """The un-symmetrised score m(z_i, z_j) [n]."""
        x = torch.cat([zi - zj, zi * zj], dim=1)
        x = self.bn1(self.activation(self.fc1(x)))
        x = self.bn2(self.activation(self.fc2(x)))
        return self.out(x).view(-1)

    def forward(self, zi, zj):
        """delta_hat [n] for the pairs (zi[k], zj[k]); an empty input gives an empty output."""
        if zi.shape != zj.shape or zi.dim() != 2 or zi.shape[1] != self.dim:
            raise ValueError(f"SiameseHead(dim={self.dim}) needs two [n, {self.dim}] inputs, got "
                             f"{tuple(zi.shape)} and {tuple(zj.shape)}")
        n = zi.shape[0]
        if n == 0:
            return zi.new_zeros(0)
        both = self.m(torch.cat([zi, zj]), torch.cat([zj, zi]))
        return (both[:n] - both[n:]) / 2


class PairHead(nn.Module):
    """l(z_a, z_c) = MLP([z_a * z_c, |z_a - z_c|, z_a - z_c]) -> scalar (asymmetric: the signed block)."""

    def __init__(self, dim, hidden=512, act="leaky_relu"):
        super().__init__()
        self.dim, self.hidden, self.act = int(dim), int(hidden), act
        self.activation = activation_function_dict[act]
        self.fc1 = nn.Linear(3 * self.dim, self.hidden)
        self.ln1 = nn.LayerNorm(self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.ln2 = nn.LayerNorm(self.hidden)
        self.out = nn.Linear(self.hidden, 1)

    def forward(self, za, zc):
        """l [n] for the pairs (za[k], zc[k]) (anchor, candidate); an empty input gives an empty output."""
        if za.shape != zc.shape or za.dim() != 2 or za.shape[1] != self.dim:
            raise ValueError(f"PairHead(dim={self.dim}) needs two [n, {self.dim}] inputs, got "
                             f"{tuple(za.shape)} and {tuple(zc.shape)}")
        if za.shape[0] == 0:
            return za.new_zeros(0)
        d = za - zc
        x = torch.cat([za * zc, d.abs(), d], dim=1)
        x = self.ln1(self.activation(self.fc1(x)))
        x = self.ln2(self.activation(self.fc2(x)))
        return self.out(x).view(-1)


class MomentsHead(nn.Module):
    """[n, dim] -> [n, out]: Linear, activation, LayerNorm, Linear (standardised Zernike invariants)."""

    def __init__(self, dim, out=MOMENTS_DIM, hidden=MOMENTS_HIDDEN, act="leaky_relu"):
        super().__init__()
        self.dim, self.out_dim, self.hidden, self.act = int(dim), int(out), int(hidden), act
        self.activation = activation_function_dict[act]
        self.fc1 = nn.Linear(self.dim, self.hidden)
        self.ln1 = nn.LayerNorm(self.hidden)
        self.out = nn.Linear(self.hidden, self.out_dim)

    def forward(self, z):
        if z.dim() != 2 or z.shape[1] != self.dim:
            raise ValueError(f"MomentsHead(dim={self.dim}) needs an [n, {self.dim}] input, got {tuple(z.shape)}")
        if z.shape[0] == 0:
            return z.new_zeros((0, self.out_dim))
        return self.out(self.ln1(self.activation(self.fc1(z))))


def input_moments_targets(mode):
    """The side-file descriptors (pocket / ligand) an --input-moments mode reads."""
    if mode not in INPUT_MOMENTS_BLOCKS:
        raise ValueError(f"unknown input-moments mode {mode!r} (known: {', '.join(INPUT_MOMENTS_CHOICES)})")
    return tuple(t for t in MOMENTS_TARGETS if t in INPUT_MOMENTS_BLOCKS[mode])


def input_moments_dim(mode):
    """d_in of an --input-moments mode: 343 per block (pocket, ligand, cross) plus the missing flag; 0 for none."""
    if mode not in INPUT_MOMENTS_BLOCKS:
        raise ValueError(f"unknown input-moments mode {mode!r} (known: {', '.join(INPUT_MOMENTS_CHOICES)})")
    k = len(INPUT_MOMENTS_BLOCKS[mode])
    return 0 if k == 0 else MOMENTS_DIM * k + 1


def pocket_complement(pocket):
    """[n, 343] (channel-major, 7 x 49) -> the same with channels pos <-> neg and donor <-> acceptor swapped."""
    n = pocket.shape[0]
    return pocket.reshape(n, N_ZERNIKE_CHANNELS, -1)[:, list(COMPLEMENT_PERM)].reshape(n, -1)


def moments_input_matrix(pocket, ligand, mask_pocket, mask_ligand, mode):
    """The [n, d_in] float32 input of an --input-moments mode from standardised descriptors (torch tensors; pocket /
    ligand [n, 343] or None when the mode does not read them, masks [n] bool, True = descriptor present). A block
    whose descriptor is missing is zeros (never its stored values, which may be NaN), the cross block is zeros when
    either side is missing, and the last column is 1 when any block the mode uses is missing."""
    blocks = INPUT_MOMENTS_BLOCKS[mode]
    if not blocks:
        raise ValueError("--input-moments none has no input vector")
    parts, missing, used = [], None, {}
    for target, values, mask in (("pocket", pocket, mask_pocket), ("ligand", ligand, mask_ligand)):
        if target not in blocks:
            continue
        if values is None or mask is None:
            raise ValueError(f"--input-moments {mode} needs the {target} descriptor")
        mask = mask.view(-1).to(torch.bool)
        if values.dim() != 2 or values.shape[1] != MOMENTS_DIM or values.shape[0] != mask.numel():
            raise ValueError(f"{target} descriptors must be [n, {MOMENTS_DIM}] with n masks, got "
                             f"{tuple(values.shape)} and {mask.numel()}")
        v = torch.where(mask[:, None], values.to(torch.float32), torch.zeros((), dtype=torch.float32,
                                                                              device=values.device))
        used[target] = v
        parts.append(v)
        missing = ~mask if missing is None else missing | ~mask
    if "cross" in blocks:
        parts.append(pocket_complement(used["pocket"]) * used["ligand"])
    parts.append(missing.to(torch.float32)[:, None])
    return torch.cat(parts, dim=1)


class MomentsInput(nn.Module):
    """data.moments [n, d_in] -> [n, out_dim]: Linear(d_in, proj) -> activation -> LayerNorm; proj 0: the raw
    vector (out_dim = d_in, no parameters). `mode` (the --input-moments mode) is kept for the evaluator."""

    def __init__(self, d_in, proj=INPUT_MOMENTS_PROJ, act="leaky_relu", mode=None):
        super().__init__()
        self.d_in, self.proj, self.act, self.mode = int(d_in), int(proj), act, mode
        if mode is not None and input_moments_dim(mode) != self.d_in:
            raise ValueError(f"--input-moments {mode} has d_in {input_moments_dim(mode)}, not {self.d_in}")
        if self.d_in < 1 or self.proj < 0:
            raise ValueError(f"MomentsInput needs d_in >= 1 and proj >= 0, got {self.d_in}, {self.proj}")
        self.activation = activation_function_dict[act]
        if self.proj > 0:
            self.fc = nn.Linear(self.d_in, self.proj)
            self.ln = nn.LayerNorm(self.proj)
        self.out_dim = self.proj if self.proj > 0 else self.d_in

    def forward(self, m):
        if m.dim() != 2 or m.shape[1] != self.d_in:
            raise ValueError(f"MomentsInput(d_in={self.d_in}) needs an [n, {self.d_in}] input, got {tuple(m.shape)}")
        if self.proj == 0:
            return m
        return self.ln(self.activation(self.fc(m)))


def batch_moments(data, n, d_in):
    """data.moments as an [n, d_in] float32 tensor; a clear error when the batch carries none."""
    m = getattr(data, "moments", None)
    if m is None:
        raise RuntimeError("this model takes moments inputs but the batch has no data.moments (the trainer / "
                           "evaluator attaches them from the Zernike side file)")
    m = m.view(n, -1)
    if m.shape[1] != d_in:
        raise ValueError(f"data.moments has {m.shape[1]} columns, the model expects {d_in}")
    return m.to(torch.float32)


class MomentsOnlyNet(nn.Module):
    """No-graph baseline: data.moments [B, d_in] -> fc1 -> activation -> bn1 -> fc2 -> activation -> bn2 -> out
    [B, 1] (the clone's MLP order, widths 512 and 256). forward reads data.moments and nothing else of the batch."""

    def __init__(self, d_in, hidden=MOMENTS_ONLY_HIDDEN, act="leaky_relu", mode=None):
        super().__init__()
        self.d_in, self.hidden, self.act, self.input_moments = int(d_in), tuple(int(h) for h in hidden), act, mode
        if mode is not None and input_moments_dim(mode) != self.d_in:
            raise ValueError(f"--input-moments {mode} has d_in {input_moments_dim(mode)}, not {self.d_in}")
        if len(self.hidden) != 2:
            raise ValueError("MomentsOnlyNet takes two hidden widths")
        self.activation = activation_function_dict[act]
        self.fc1 = nn.Linear(self.d_in, self.hidden[0])
        self.bn1 = nn.BatchNorm1d(self.hidden[0])
        self.fc2 = nn.Linear(self.hidden[0], self.hidden[1])
        self.bn2 = nn.BatchNorm1d(self.hidden[1])
        self.out = nn.Linear(self.hidden[1], 1)

    def forward(self, data):
        m = getattr(data, "moments", None)
        if m is None:
            raise RuntimeError("MomentsOnlyNet needs data.moments")
        x = m.view(m.shape[0], -1).to(torch.float32)
        if x.shape[1] != self.d_in:
            raise ValueError(f"data.moments has {x.shape[1]} columns, MomentsOnlyNet expects {self.d_in}")
        x = self.bn1(self.activation(self.fc1(x)))
        x = self.bn2(self.activation(self.fc2(x)))
        return self.out(x)


def moments_only_from_state_dict(sd):
    """(d_in, (hidden1, hidden2)) of a MomentsOnlyNet state dict (fc1.weight [h1, d_in], fc2.weight [h2, h1]), or
    None when sd is not one (any encoder / GNN key, or no fc1 / fc2 / bn1)."""
    keys = {k.split(".", 1)[0] for k in sd}
    if keys != {"fc1", "bn1", "fc2", "bn2", "out"}:
        return None
    w1, w2 = sd["fc1.weight"], sd["fc2.weight"]
    return int(w1.shape[1]), (int(w1.shape[0]), int(w2.shape[0]))


def input_moments_from_state_dict(sd):
    """(d_in, proj) of a wrapper's MomentsInput with a projection (moments_in.fc.weight [proj, d_in]); None when the
    state dict has no moments_in.* keys (no moments input, or a raw concatenation with proj 0: the enlarged
    affinity.fc1 alone tells, see fc1_input_width)."""
    w = sd.get(f"{MOMENTS_IN}.fc.weight")
    return None if w is None else (int(w.shape[1]), int(w.shape[0]))


def side_stats_digest(side):
    """sha1 of a Zernike side file's standardisation (pocket / ligand mean and sd as float32 bytes): the trainer
    records it in the sidecar so the evaluator can refuse another side file's statistics for a model with moments
    inputs."""
    import hashlib

    import numpy as np

    h = hashlib.sha1()
    for key in ("pocket_mean", "pocket_sd", "ligand_mean", "ligand_sd"):
        h.update(np.ascontiguousarray(side[key], dtype=np.float32).tobytes())
    return h.hexdigest()


def fc1_input_width(sd):
    """Input width of the pointwise MLP's fc1 in a wrapper state dict (D, or D + the moments-input width)."""
    return int(sd[f"{AFFINITY}.fc1.weight"].shape[1])


def init_linear(module, init):
    """Apply the clone's init_weights to every nn.Linear of module (init_weights would call xavier_normal_ on
    a LayerNorm's 1-d weight and raise); LayerNorm keeps its default (weight 1, bias 0)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init(m)
    return module


def _rename_keys(sd, rename):
    """New OrderedDict with rename(key, strict=True) applied to every key (None drops the entry) and the
    per-module `_metadata` (version info that BatchNorm reads when loading) renamed with strict=False."""
    out = OrderedDict()
    for key, value in sd.items():
        new = rename(key, strict=True)
        if new is not None:
            out[new] = value
    meta = getattr(sd, "_metadata", None)
    if meta is not None:
        out._metadata = OrderedDict()
        for key, value in meta.items():
            new = "" if key == "" else rename(key, strict=False)
            if new is not None:
                out._metadata[new] = value
    return out


def _plain_key(key, strict):
    head = key.split(".", 1)[0]
    if head in ENCODER_KEYS:
        return f"{ENCODER}.{key}"
    if head in AFFINITY_KEYS:
        return f"{AFFINITY}.{key}"
    if strict:
        raise KeyError(f"unexpected GATv2Net state-dict key {key!r}")
    return None


def _wrapped_key(key, strict):
    head, _, rest = key.partition(".")
    if head in HEAD_PREFIXES:
        return None
    if head == MOMENTS_IN and strict:
        raise KeyError(f"{key!r}: a wrapper with moments inputs has an enlarged fc1 and has no plain GATv2Net form")
    allowed = {ENCODER: ENCODER_KEYS, AFFINITY: AFFINITY_KEYS}.get(head)
    if allowed is not None and rest and rest.split(".", 1)[0] in allowed:
        return rest
    if strict:
        raise KeyError(f"unexpected wrapper state-dict key {key!r}")
    return None  # metadata of the containers themselves ("encoder", "affinity")


def plain_to_wrapped(sd):
    """A plain GATv2Net state dict -> the wrapper's encoder.* / affinity.* keys (same order, same tensors)."""
    return _rename_keys(sd, _plain_key)


def wrapped_to_plain(sd):
    """The wrapper's state dict -> a plain GATv2Net state dict (siamese.* / pair.* / moments.* dropped, order and
    tensors kept)."""
    return _rename_keys(sd, _wrapped_key)


def is_wrapped_state_dict(sd):
    """True for a state dict written by AffinityWithSiamese (encoder.* / affinity.* keys)."""
    return any(k.startswith((f"{ENCODER}.", f"{AFFINITY}.")) for k in sd)


def has_siamese_keys(sd):
    return any(k.startswith(f"{SIAMESE}.") for k in sd)


def siamese_hidden_from_state_dict(sd):
    """The head's hidden width from siamese.fc1.weight [hidden, 2 * dim], or None without a head."""
    w = sd.get(f"{SIAMESE}.fc1.weight")
    return None if w is None else int(w.shape[0])


def has_pair_keys(sd):
    return any(k.startswith(f"{PAIR}.") for k in sd)


def pair_hidden_from_state_dict(sd):
    """The pair head's hidden width from pair.fc1.weight [hidden, 3 * dim], or None without a pair head."""
    w = sd.get(f"{PAIR}.fc1.weight")
    return None if w is None else int(w.shape[0])


def moments_from_state_dict(sd):
    """{target: (hidden, out)} of the moments heads in a wrapper state dict (moments.<target>.fc1.weight
    [hidden, dim], moments.<target>.out.weight [out, hidden]); {} without moments heads."""
    out = {}
    for target in MOMENTS_TARGETS:
        w1 = sd.get(f"{MOMENTS}.{target}.fc1.weight")
        w2 = sd.get(f"{MOMENTS}.{target}.out.weight")
        if w1 is not None and w2 is not None:
            out[target] = (int(w1.shape[0]), int(w2.shape[0]))
    unknown = {k.split(".")[1] for k in sd if k.startswith(f"{MOMENTS}.")} - set(out)
    if unknown:
        raise KeyError(f"unexpected moments heads {sorted(unknown)} (known: {', '.join(MOMENTS_TARGETS)})")
    return out


class AffinityWithSiamese(nn.Module):
    """GATv2Net split into encoder + affinity, plus optional heads on the pooled embedding: a SiameseHead
    (arm S), a PairHead and MomentsHeads ({"pocket" | "ligand": MomentsHead}, arm P), and optional moments inputs
    to the pointwise head (moments_in: a MomentsInput, arm Z).

    embed(data) -> z [B, D]; affinity(z) -> [B, 1] (the MLP on its input); affinity_from_z(z, data) -> [B, 1] (the
    MLP on z, or on cat(z, moments_in(data.moments)) with moments inputs); forward(data) =
    affinity_from_z(embed(data), data) (the pointwise prediction, identical to GATv2Net.forward without moments
    inputs); pair_delta(z, i, j) = siamese(z[i], z[j]) [n_pairs]; pair_score(z, a, c) = pair(z[a], z[c]) [n_pairs];
    moments_pred(z, target) = moments[target](z) [B, out]. With moments_in the GATv2Net's fc1 is REPLACED (in the
    GATv2Net too) by a fresh nn.Linear(D + moments_in.out_dim, fc1.out_features): initialise it after wrapping."""

    def __init__(self, gnn, siamese=None, pair=None, moments=None, moments_in=None):
        super().__init__()
        self.encoder = Encoder(gnn)
        if moments_in is not None:
            if gnn.fc1.in_features != self.encoder.dim:
                raise ValueError("the GATv2Net's fc1 is already enlarged (wrapped twice?)")
            gnn.fc1 = nn.Linear(self.encoder.dim + moments_in.out_dim, gnn.fc1.out_features)
        self.affinity = AffinityMLP(gnn)
        if siamese is not None and siamese.dim != self.encoder.dim:
            raise ValueError(f"SiameseHead dim {siamese.dim} != embedding dim {self.encoder.dim}")
        self.siamese = siamese
        if pair is not None and pair.dim != self.encoder.dim:
            raise ValueError(f"PairHead dim {pair.dim} != embedding dim {self.encoder.dim}")
        self.pair = pair
        if moments:
            bad = set(moments) - set(MOMENTS_TARGETS)
            if bad:
                raise ValueError(f"unknown moments targets {sorted(bad)}")
            for target, head in moments.items():
                if head.dim != self.encoder.dim:
                    raise ValueError(f"MomentsHead {target} dim {head.dim} != embedding dim {self.encoder.dim}")
            # fixed order (pocket, ligand) whatever the dict order, so the state dict is deterministic
            self.moments = nn.ModuleDict({t: moments[t] for t in MOMENTS_TARGETS if t in moments})
        else:
            self.moments = None
        self.moments_in = moments_in  # registered last: models without it keep the arm-S / arm-P state dict

    @property
    def moments_targets(self):
        return tuple(self.moments.keys()) if self.moments is not None else ()

    @property
    def input_moments(self):
        """The --input-moments mode of the pointwise head's moments input ("none" without one)."""
        if self.moments_in is None:
            return "none"
        return self.moments_in.mode if self.moments_in.mode is not None else "unknown"

    @property
    def dim(self):
        return self.encoder.dim

    def embed(self, data):
        return self.encoder(data)

    def affinity_from_z(self, z, data=None):
        """The pointwise prediction [B, 1] from the pooled embedding z (and data.moments with moments inputs)."""
        if self.moments_in is None:
            return self.affinity(z)
        m = batch_moments(data, z.shape[0], self.moments_in.d_in).to(z.device)
        return self.affinity(torch.cat([z, self.moments_in(m)], dim=1))

    def forward(self, data):
        return self.affinity_from_z(self.embed(data), data)

    def pair_delta(self, z, i, j):
        if self.siamese is None:
            raise RuntimeError("this model has no Siamese head")
        return self.siamese(z[i], z[j])

    def pair_score(self, z, a, c):
        if self.pair is None:
            raise RuntimeError("this model has no pair head")
        return self.pair(z[a], z[c])

    def moments_pred(self, z, target):
        if target not in self.moments_targets:
            raise RuntimeError(f"this model has no {target} moments head")
        return self.moments[target](z)

    def from_plain_state_dict(self, sd):
        """Load a plain GATv2Net state dict into encoder + affinity (the heads are left as they are); every
        encoder / affinity key must be present and no other key may appear. Returns self. An error with moments
        inputs (the enlarged fc1 cannot take a plain fc1)."""
        if self.moments_in is not None:
            raise RuntimeError("a wrapper with moments inputs cannot load a plain GATv2Net state dict (fc1 widths "
                               "differ)")
        result = self.load_state_dict(plain_to_wrapped(sd), strict=False)
        missing = [k for k in result.missing_keys if k.split(".", 1)[0] not in HEAD_PREFIXES]
        if missing or result.unexpected_keys:
            raise KeyError(f"plain state dict does not match the wrapper: missing {missing}, "
                           f"unexpected {list(result.unexpected_keys)}")
        return self

    def to_plain_state_dict(self):
        """encoder + affinity as a plain GATv2Net state dict (loads into GATv2Net as is); an error with moments inputs
        (the enlarged fc1 has no GATv2Net form)."""
        if self.moments_in is not None:
            raise RuntimeError("a wrapper with moments inputs has no plain GATv2Net state dict")
        return wrapped_to_plain(self.state_dict())
