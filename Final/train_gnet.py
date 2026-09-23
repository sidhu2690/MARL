"""
train_gnet.py
--------------------------------------------------------------------
Joint magnitude+direction Transformer training on precomputed
(X, E, Q) npy triples produced by datagen.py.

Splits the flat sample pool into train/test (default 90/10) and
reports both training and test loss every epoch.

Launch (single GPU):
    python train_gnet.py --data_root ./data --batch_size 64 --epochs 50

Launch (multi-GPU DDP, e.g. 4 GPUs on one node):
    torchrun --nproc_per_node=4 train_gnet.py \
        --data_root ./data --batch_size 64 --epochs 50
--------------------------------------------------------------------
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ----------------------------------------------------------------------
# DDP helpers
# ----------------------------------------------------------------------
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def setup_ddp():
    """Initialize process group if launched via torchrun (env vars present)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        return device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False


def cleanup_ddp(distributed):
    if distributed:
        dist.destroy_process_group()


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ----------------------------------------------------------------------
# Dataset: flat pool of individually-saved (X, E, Q) npy files
# ----------------------------------------------------------------------
class GDataset(Dataset):
    """
    Expects:
        data_root/X/<name>.npy   shape (3, GRID, GRID)
        data_root/E/<name>.npy   shape (GRID, GRID)
        data_root/Q/<name>.npy   shape (GRID, GRID)
    Files are matched by filename across the three folders.
    """

    def __init__(self, data_root):
        self.x_dir = os.path.join(data_root, "X")
        self.e_dir = os.path.join(data_root, "E")
        self.q_dir = os.path.join(data_root, "Q")

        x_files = sorted(os.listdir(self.x_dir))
        self.names = [f for f in x_files if f.endswith(".npy")]

        if len(self.names) == 0:
            raise RuntimeError(f"No .npy files found under {self.x_dir}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        X = np.load(os.path.join(self.x_dir, name))  # (3, G, G)
        E = np.load(os.path.join(self.e_dir, name))  # (G, G)
        Q = np.load(os.path.join(self.q_dir, name))  # (G, G)
        X = torch.from_numpy(X.astype(np.float32))
        E = torch.from_numpy(E.astype(np.float32))
        Q = torch.from_numpy(Q.astype(np.float32))
        return X, E, Q


def split_dataset(dataset, test_frac, seed):
    n = len(dataset)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, test_idx.tolist())


# ----------------------------------------------------------------------
# Joint magnitude + direction transformer
# ----------------------------------------------------------------------
class GTransformer(nn.Module):
    """
    Single transformer that predicts, per grid cell:
      - channel 0: raw energy (regressed against squared error E)
      - channel 1: direction logit (regressed against sign target Q via BCE)
    """

    def __init__(self, in_channels=3, grid=100, d_model=32, nhead=4,
                 num_layers=2, dim_feedforward=64, dropout=0.0):
        super().__init__()
        self.grid = grid
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, grid * grid, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 2)  # [energy, dir_logit]

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        tokens = self.input_proj(tokens) + self.pos_embed
        out = self.encoder(tokens)                      # (B, H*W, d_model)
        out = self.output_proj(out)                      # (B, H*W, 2)
        out = out.view(B, H, W, 2)
        energy = out[..., 0]      # (B, H, W)
        dir_logit = out[..., 1]   # (B, H, W)
        return energy, dir_logit


# ----------------------------------------------------------------------
# Loss computation (shared between train/eval)
# ----------------------------------------------------------------------
def compute_losses(model, X, E, Q, bce, mse_weight, bce_weight):
    energy_pred, dir_logit = model(X)

    visited = (X[:, 2] > 0).float()

    # ---- energy (magnitude) loss: visited/unvisited balanced MSE ----
    n_vis = visited.sum().clamp(min=1.0)
    n_unvis = (1 - visited).sum().clamp(min=1.0)
    w_energy = visited / n_vis + (1 - visited) / n_unvis
    mse_loss = ((energy_pred - E) ** 2 * w_energy).sum() / 2.0

    # ---- direction loss: visited-only BCE, normalized by #visited ----
    target01 = (Q + 1.0) / 2.0
    per_cell_bce = bce(dir_logit, target01)
    w_dir = visited / visited.sum().clamp(min=1.0)
    dir_loss = (per_cell_bce * w_dir).sum()

    loss = mse_weight * mse_loss + bce_weight * dir_loss

    with torch.no_grad():
        pred_sign = torch.where(dir_logit >= 0, 1.0, -1.0)
        n_correct = ((pred_sign == Q).float() * visited).sum()
        n_total = visited.sum()

    return loss, mse_loss, dir_loss, n_correct, n_total


def run_epoch(model, loader, optimizer, bce, mse_weight, bce_weight, device, train_mode):
    model.train(train_mode)
    tot_loss = tot_mse = tot_bce = 0.0
    n_batches = 0
    n_correct = n_total = 0.0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for X, E, Q in loader:
            X = X.to(device, non_blocking=True)
            E = E.to(device, non_blocking=True)
            Q = Q.to(device, non_blocking=True)

            loss, mse_loss, dir_loss, correct, total = compute_losses(
                model, X, E, Q, bce, mse_weight, bce_weight
            )

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss += loss.item()
            tot_mse += mse_loss.item()
            tot_bce += dir_loss.item()
            n_correct += correct.item()
            n_total += total.item()
            n_batches += 1

    return tot_loss, tot_mse, tot_bce, n_batches, n_correct, n_total


def reduce_stats(stats, device, distributed):
    t = torch.tensor(stats, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train(args):
    device, distributed = setup_ddp()
    rank = get_rank()
    world_size = get_world_size()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print0(f"[setup] distributed={distributed}  world_size={world_size}  device={device}")

    # -------------------- dataset / split --------------------
    full_dataset = GDataset(args.data_root)
    print0(f"[data] total samples found: {len(full_dataset)}")

    train_set, test_set = split_dataset(full_dataset, args.test_frac, args.seed)
    print0(f"[data] train={len(train_set)}  test={len(test_set)}")

    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                            shuffle=True, seed=args.seed)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank,
                                           shuffle=False)
        train_shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # -------------------- model --------------------
    model = GTransformer(
        in_channels=3, grid=args.grid, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_test_loss = float("inf")
    bad_epochs = 0

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        # ---- train ----
        train_stats = run_epoch(
            model, train_loader, optimizer, bce,
            args.mse_weight, args.bce_weight, device, train_mode=True,
        )
        train_stats = reduce_stats(train_stats, device, distributed)
        tr_loss, tr_mse, tr_bce, tr_nb, tr_correct, tr_total = train_stats
        tr_avg_loss = tr_loss / max(tr_nb, 1)
        tr_avg_mse = tr_mse / max(tr_nb, 1)
        tr_avg_bce = tr_bce / max(tr_nb, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # ---- test ----
        test_stats = run_epoch(
            model, test_loader, optimizer, bce,
            args.mse_weight, args.bce_weight, device, train_mode=False,
        )
        test_stats = reduce_stats(test_stats, device, distributed)
        te_loss, te_mse, te_bce, te_nb, te_correct, te_total = test_stats
        te_avg_loss = te_loss / max(te_nb, 1)
        te_avg_mse = te_mse / max(te_nb, 1)
        te_avg_bce = te_bce / max(te_nb, 1)
        te_acc = te_correct / max(te_total, 1)

        print0(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train: loss {tr_avg_loss:.6f} mse {tr_avg_mse:.6f} bce {tr_avg_bce:.6f} acc {tr_acc:.3f}  |  "
            f"test: loss {te_avg_loss:.6f} mse {te_avg_mse:.6f} bce {te_avg_bce:.6f} acc {te_acc:.3f}"
        )

        if best_test_loss - te_avg_loss > args.min_delta:
            best_test_loss = te_avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print0(f"[train] test loss plateaued at epoch {epoch+1}, stopping early")
                break

    # -------------------- save (rank 0 only) --------------------
    if is_main_process():
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save(model_to_save.state_dict(), args.save_path)
        print0(f"saved joint magnitude+direction model to {args.save_path}")

    cleanup_ddp(distributed)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train joint magnitude+direction GTransformer")

    # data
    p.add_argument("--data_root", type=str, required=True,
                    help="root dir containing X/, E/, Q/ subfolders of .npy files")
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--test_frac", type=float, default=0.1)

    # model hyperparams
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)

    # training hyperparams
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--bce_weight", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=40)

    # output
    p.add_argument("--save_path", type=str, default="gnet_transformer_joint.pt")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
