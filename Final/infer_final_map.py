"""
compare_local_global.py
--------------------------------------------------------------------
Final pipeline: loads the trained alpha_net and the joint gnet
(magnitude+direction transformer), simulates fresh episodes, builds
a snapshot, predicts a combined local+global effect map, and compares
it against the brute-force ground truth (computed by hypothetically
sending an honest report to every single cell and measuring the
resulting *global* loss change, assuming alpha_net is perfect).

Metrics reported per episode and averaged across episodes:
  - KL divergence between normalized ground-truth map and predicted map
  - MAE between ground-truth map and predicted map

Also times each stage of the pipeline:
  - episode simulation / snapshot building
  - local prediction (joint gnet forward pass)
  - global effect prediction
  - brute-force ground-truth computation

Usage:
    python compare_local_global.py \
        --alpha_net_path alpha_net.pt \
        --gnet_path gnet_transformer_joint.pt \
        --n_episodes 5
--------------------------------------------------------------------
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Environment simulation helpers
# ----------------------------------------------------------------------
def make_true_map(rng, grid):
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    field = np.zeros((grid, grid))
    for _ in range(5):
        cx, cy = rng.uniform(0, grid, 2)
        amp = rng.uniform(0.5, 1.0)
        sigma = rng.uniform(1.5, 3.5)
        field += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field


def make_adv_mask(rng, n_robots, n_adv):
    mask = np.array([False] * (n_robots - n_adv) + [True] * n_adv)
    rng.shuffle(mask)
    return mask


def random_walk(rng, grid, steps):
    pos = rng.integers(0, grid, size=2)
    cells = []
    for _ in range(steps):
        while True:
            move = rng.choice([-1, 0, 1], size=2)
            new_pos = np.clip(pos + move, 0, grid - 1)
            if not np.array_equal(new_pos, pos):
                break
        pos = new_pos
        cells.append(tuple(pos))
    return cells


def report_value(rng, true_val, adversarial, cell):
    if adversarial:
        bias_sign = 1 if (hash(cell) % 2 == 0) else -1
        val = true_val + bias_sign * rng.uniform(0.15, 0.3) + rng.normal(0, 0.05)
    else:
        val = true_val + rng.normal(0, 0.02)
    return float(np.clip(val, 0, 1))


def error_scale_factor(U):
    return (2.0 * U + 1.0) / (U + 1.0) ** 2


PRIOR = 0


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class AlphaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class GTransformer(nn.Module):
    """Joint magnitude (energy) + direction (logit) transformer."""

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
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.input_proj(tokens) + self.pos_embed
        out = self.encoder(tokens)
        out = self.output_proj(out).view(B, H, W, 2)
        energy = out[..., 0]
        dir_logit = out[..., 1]
        return energy, dir_logit


# ----------------------------------------------------------------------
# Episode simulation -> snapshot
# ----------------------------------------------------------------------
def build_agent_maps(cell_reports, n_robots):
    agent_sum = [dict() for _ in range(n_robots)]
    agent_cnt = [dict() for _ in range(n_robots)]
    for cell, reports in cell_reports.items():
        for i, r in reports:
            agent_sum[i][cell] = agent_sum[i].get(cell, 0.0) + r
            agent_cnt[i][cell] = agent_cnt[i].get(cell, 0) + 1
    return agent_sum, agent_cnt


def simulate_snapshot(rng, alpha_net, device, grid, n_robots, adv_frac_max,
                       steps, epsilon, contra_thresh):
    TRUE_MAP = make_true_map(rng, grid)
    n_adv = rng.integers(1, int(n_robots * adv_frac_max) + 1)
    is_adv = make_adv_mask(rng, n_robots, n_adv)
    walks = [random_walk(rng, grid, steps) for _ in range(n_robots)]

    c = np.zeros(n_robots)
    D_sum = np.zeros(n_robots)
    D_n = np.zeros(n_robots)
    cell_reports = {}

    for t in range(steps):
        for i in range(n_robots):
            cell = walks[i][t]
            r_i = report_value(rng, TRUE_MAP[cell], is_adv[i], cell)
            others = cell_reports.get(cell, [])
            if others:
                residuals = [abs(r_i - r_j) for _, r_j in others]
                D_sum[i] += float(np.mean(residuals))
                D_n[i] += 1
                for j, r_j in others:
                    if abs(r_i - r_j) > contra_thresh:
                        c[i] += 1
                        c[j] += 1
            cell_reports.setdefault(cell, []).append((i, r_i))

    mu_c, sigma_c = c.mean(), c.std()
    z = (c - mu_c) / (sigma_c + epsilon)
    D = np.divide(D_sum, D_n, out=np.zeros_like(D_sum), where=D_n > 0)
    c_norm = c / steps
    feats = torch.tensor(np.stack([c_norm, z, D], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        alpha = alpha_net(feats).cpu().numpy()

    W_sum = np.zeros((grid, grid))
    A_sum = np.zeros((grid, grid))
    N_map = np.zeros((grid, grid))
    for cell, reports in cell_reports.items():
        idx = np.array([r for r, _ in reports])
        vals = np.array([v for _, v in reports])
        a = alpha[idx]
        W_sum[cell] = (a * vals).sum()
        A_sum[cell] = a.sum()
        N_map[cell] = len(reports)

    M_t = np.where(A_sum > 0, W_sum / np.maximum(A_sum, 1e-8), PRIOR)
    agent_sum, agent_cnt = build_agent_maps(cell_reports, n_robots)

    return dict(
        TRUE_MAP=TRUE_MAP, cell_reports=cell_reports, c=c, D_sum=D_sum, D_n=D_n,
        denom=steps, alpha=alpha, W_sum=W_sum, A_sum=A_sum, M_t=M_t, U_t=A_sum,
        N_t=N_map, agent_sum=agent_sum, agent_cnt=agent_cnt,
    )


# ----------------------------------------------------------------------
# Local prediction (single joint-model forward pass)
# ----------------------------------------------------------------------
def predict_local_and_signed(gnet, X_t, device):
    with torch.no_grad():
        x = torch.tensor(X_t, dtype=torch.float32, device=device).unsqueeze(0)
        energy_pred, dir_logit = gnet(x)
        energy_pred = energy_pred.squeeze(0).cpu().numpy()
        dir_logit = dir_logit.squeeze(0).cpu().numpy()
    energy = np.clip(energy_pred, 0, None)
    q = np.where(dir_logit >= 0, 1.0, -1.0)
    e_signed = q * np.sqrt(energy)
    U_t = X_t[1]
    R_local = energy * error_scale_factor(U_t)
    return R_local, e_signed


# ----------------------------------------------------------------------
# Global effect prediction (uses alpha_net, assumed perfect)
# ----------------------------------------------------------------------
def compute_lambda_total(agent_sum, agent_cnt, M_t, U_t, e_signed, n_robots):
    lam = np.zeros(n_robots)
    for i in range(n_robots):
        for cell, s in agent_sum[i].items():
            r_bar = s / agent_cnt[i][cell]
            u, m = U_t[cell], M_t[cell]
            if u > 0:
                lam[i] += e_signed[cell] * (r_bar - m) / u
    return lam


def predict_global_effect(alpha_net, cell_reports, agent_sum, agent_cnt,
                           c, D_sum, D_n, denom, alpha, lam_total, T_hat,
                           device, contra_thresh, epsilon, grid):
    feats, meta = [], []
    for s, reports in cell_reports.items():
        reporters = sorted({i for i, _ in reports})
        if not reporters:
            continue
        that = T_hat[s]
        c_hyp = c.copy()
        for i in reporters:
            r_bar = agent_sum[i][s] / agent_cnt[i][s]
            c_hyp[i] = c[i] + float(abs(that - r_bar) > contra_thresh)
        mu, sigma = c_hyp.mean(), c_hyp.std()
        for i in reporters:
            r_bar = agent_sum[i][s] / agent_cnt[i][s]
            D_i_new = (D_sum[i] + abs(r_bar - that)) / (D_n[i] + 1)
            z_i_new = (c_hyp[i] - mu) / (sigma + epsilon)
            feats.append([c_hyp[i] / denom, z_i_new, D_i_new])
            meta.append((s, i))

    G_global = np.zeros((grid, grid))
    if not feats:
        return G_global

    feats = torch.tensor(np.array(feats, dtype=np.float32), device=device)
    with torch.no_grad():
        alpha_new = alpha_net(feats).cpu().numpy()

    for (s, i), a_new in zip(meta, alpha_new):
        d_alpha = a_new - alpha[i]
        G_global[s] -= lam_total[i] * d_alpha

    return G_global


# ----------------------------------------------------------------------
# Brute-force ground truth: send an honest report to every cell,
# measure the resulting GLOBAL loss change (local + global combined)
# ----------------------------------------------------------------------
def ground_truth_gain(alpha_net, snap, device, contra_thresh, epsilon, grid):
    TRUE_MAP, cell_reports = snap["TRUE_MAP"], snap["cell_reports"]
    agent_sum, agent_cnt = snap["agent_sum"], snap["agent_cnt"]
    c, D_sum, D_n, denom = snap["c"], snap["D_sum"], snap["D_n"], snap["denom"]
    alpha, W_sum, A_sum, M_t = snap["alpha"], snap["W_sum"], snap["A_sum"], snap["M_t"]

    Y_true = np.zeros((grid, grid))

    for s, reports in cell_reports.items():
        reporters = sorted({i for i, _ in reports})
        Tval = TRUE_MAP[s]
        c_hyp = c.copy()
        for i in reporters:
            r_bar = agent_sum[i][s] / agent_cnt[i][s]
            c_hyp[i] = c[i] + float(abs(Tval - r_bar) > contra_thresh)
        mu, sigma = c_hyp.mean(), c_hyp.std()

        feats = []
        for i in reporters:
            r_bar = agent_sum[i][s] / agent_cnt[i][s]
            D_i_new = (D_sum[i] + abs(r_bar - Tval)) / (D_n[i] + 1)
            z_i_new = (c_hyp[i] - mu) / (sigma + epsilon)
            feats.append([c_hyp[i] / denom, z_i_new, D_i_new])
        feats = torch.tensor(np.array(feats, dtype=np.float32), device=device)
        with torch.no_grad():
            alpha_new_vals = alpha_net(feats).cpu().numpy()
        alpha_new = dict(zip(reporters, alpha_new_vals))

        W_after, A_after, affected = W_sum.copy(), A_sum.copy(), set()
        for i in reporters:
            d_alpha = alpha_new[i] - alpha[i]
            for cell in agent_sum[i]:
                W_after[cell] += d_alpha * agent_sum[i][cell]
                A_after[cell] += d_alpha * agent_cnt[i][cell]
                affected.add(cell)
        W_after[s] += Tval
        A_after[s] += 1.0
        affected.add(s)

        old_loss = sum((TRUE_MAP[cell] - M_t[cell]) ** 2 for cell in affected)
        new_loss = sum(
            (TRUE_MAP[cell] - (W_after[cell] / A_after[cell] if A_after[cell] > 0 else PRIOR)) ** 2
            for cell in affected
        )
        Y_true[s] = old_loss - new_loss
    return Y_true


# ----------------------------------------------------------------------
# Comparison metrics
# ----------------------------------------------------------------------
def normalize_nonneg(y):
    y = np.clip(y, 0, None)
    s = y.sum()
    if s <= 1e-12:
        return np.full_like(y, 1.0 / y.size)
    return y / s


def kl_divergence(p_true, p_pred, eps=1e-8):
    p_true = np.clip(p_true, eps, None)
    p_pred = np.clip(p_pred, eps, None)
    return float(np.sum(p_true * np.log(p_true / p_pred)))


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[device] using {device}")

    alpha_net = AlphaNet().to(device)
    alpha_net.load_state_dict(torch.load(args.alpha_net_path, map_location=device))
    alpha_net.eval()

    gnet = GTransformer(
        in_channels=3, grid=args.grid, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
    ).to(device)
    gnet.load_state_dict(torch.load(args.gnet_path, map_location=device))
    gnet.eval()

    kl_list, mae_list = [], []
    last = None  # for final plot

    for ep in range(args.n_episodes):
        # ---- 1) simulate episode / build snapshot ----
        t0 = time.perf_counter()
        snap = simulate_snapshot(
            rng, alpha_net, device, args.grid, args.n_robots, args.adv_frac_max,
            args.steps, args.epsilon, args.contra_thresh,
        )
        t_data = time.perf_counter() - t0

        X_t = np.stack([snap["M_t"], snap["U_t"], snap["N_t"]]).astype(np.float32)

        # ---- 2) local prediction ----
        t0 = time.perf_counter()
        R_local, e_signed = predict_local_and_signed(gnet, X_t, device)
        t_local = time.perf_counter() - t0

        # ---- 3) global effect prediction ----
        t0 = time.perf_counter()
        T_hat = snap["M_t"] - e_signed
        lam_total = compute_lambda_total(
            snap["agent_sum"], snap["agent_cnt"], snap["M_t"], snap["U_t"],
            e_signed, args.n_robots,
        )
        G_global = predict_global_effect(
            alpha_net, snap["cell_reports"], snap["agent_sum"], snap["agent_cnt"],
            snap["c"], snap["D_sum"], snap["D_n"], snap["denom"], snap["alpha"],
            lam_total, T_hat, device, args.contra_thresh, args.epsilon, args.grid,
        )
        t_global = time.perf_counter() - t0

        y_pred_total = R_local + G_global

        # ---- 4) brute-force ground truth ----
        t0 = time.perf_counter()
        Y_true = ground_truth_gain(
            alpha_net, snap, device, args.contra_thresh, args.epsilon, args.grid,
        )
        t_brute = time.perf_counter() - t0

        # ---- 5) compare ----
        mask = snap["N_t"] > 0
        y_true_v = Y_true[mask]
        y_pred_v = y_pred_total[mask]

        kl = kl_divergence(normalize_nonneg(Y_true), normalize_nonneg(y_pred_total))
        mae = float(np.mean(np.abs(y_true_v - y_pred_v)))
        kl_list.append(kl)
        mae_list.append(mae)

        print(
            f"[episode {ep+1}/{args.n_episodes}] "
            f"data={t_data:.3f}s  local={t_local:.3f}s  global={t_global:.3f}s  "
            f"brute_force={t_brute:.3f}s  |  KL={kl:.4f}  MAE={mae:.6f}  "
            f"(visited cells={mask.sum()})"
        )

        last = dict(snap=snap, R_local=R_local, G_global=G_global,
                    y_pred_total=y_pred_total, Y_true=Y_true, mask=mask)

    print("\n=== summary over all episodes ===")
    print(f"mean KL  = {np.mean(kl_list):.4f}")
    print(f"mean MAE = {np.mean(mae_list):.6f}")

    # ---- plot comparison for the last episode ----
    snap = last["snap"]
    mask = last["mask"]
    R_local_m = np.where(mask, last["R_local"], np.nan)
    y_pred_total_m = np.where(mask, last["y_pred_total"], np.nan)
    Y_true_m = np.where(mask, last["Y_true"], np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axes[0].imshow(snap["TRUE_MAP"], cmap="viridis")
    axes[0].set_title("Ground truth T(s)")
    im1 = axes[1].imshow(Y_true_m, cmap="magma")
    axes[1].set_title("Y_true (brute-force local+global)")
    im2 = axes[2].imshow(R_local_m, cmap="magma")
    axes[2].set_title("Predicted (local only)")
    im3 = axes[3].imshow(y_pred_total_m, cmap="magma")
    axes[3].set_title("Predicted (local + global)")
    for ax, im in zip(axes, [im0, im1, im2, im3]):
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=130)
    print(f"saved comparison plot to {args.plot_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare predicted local+global map against brute-force ground truth")

    p.add_argument("--alpha_net_path", type=str, required=True)
    p.add_argument("--gnet_path", type=str, required=True)

    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--n_robots", type=int, default=15)
    p.add_argument("--adv_frac_max", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--contra_thresh", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--n_episodes", type=int, default=5)

    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=64)

    p.add_argument("--plot_path", type=str, default="local_global_comparison.png")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
