import os
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Config — must match the alpha_net training config
# ----------------------------------------------------------------------
GRID = 100
N_ROBOTS = 15
ADV_FRAC_MAX = 0.3
STEPS = 1000
EPSILON = 0.05
CONTRA_THRESH = 0.2
SEED = 40

SAMPLES_PER_EPISODE = 100
N_EPISODES = 50000

OUT_DIR = "data"
X_DIR = os.path.join(OUT_DIR, "X")
E_DIR = os.path.join(OUT_DIR, "E")
Q_DIR = os.path.join(OUT_DIR, "Q")
for d in (X_DIR, E_DIR, Q_DIR):
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEVICE}")


# ----------------------------------------------------------------------
# Environment helpers (identical to training script)
# ----------------------------------------------------------------------
def make_true_map(grid):
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    field = np.zeros((grid, grid))
    for _ in range(5):
        cx, cy = rng.uniform(0, grid, 2)
        amp = rng.uniform(0.5, 1.0)
        sigma = rng.uniform(1.5, 3.5)
        field += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field


def make_adv_mask(n_robots, n_adv):
    mask = np.array([False] * (n_robots - n_adv) + [True] * n_adv)
    rng.shuffle(mask)
    return mask


def random_walk(grid, steps):
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


def report_value(true_val, adversarial, cell):
    if adversarial:
        bias_sign = 1 if (hash(cell) % 2 == 0) else -1
        val = true_val + bias_sign * rng.uniform(0.15, 0.3) + rng.normal(0, 0.05)
    else:
        val = true_val + rng.normal(0, 0.02)
    return float(np.clip(val, 0, 1))


PRIOR = 0


def error_scale_factor(U):
    return (2.0 * U + 1.0) / (U + 1.0) ** 2


# ----------------------------------------------------------------------
# AlphaNet (must match training script's architecture)
# ----------------------------------------------------------------------
class AlphaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


# ----------------------------------------------------------------------
# Full-episode rollout (keeps every one of the 1000 steps)
# ----------------------------------------------------------------------
def rollout_episode_full(alpha_net):
    TRUE_MAP = make_true_map(GRID)
    n_adv = rng.integers(1, int(N_ROBOTS * ADV_FRAC_MAX) + 1)
    is_adv = make_adv_mask(N_ROBOTS, n_adv)
    walks = [random_walk(GRID, STEPS) for _ in range(N_ROBOTS)]

    c = np.zeros(N_ROBOTS)
    D_sum = np.zeros(N_ROBOTS)
    D_n = np.zeros(N_ROBOTS)
    cell_reports = {}

    X_all = np.empty((STEPS, 3, GRID, GRID), dtype=np.float32)
    E_all = np.empty((STEPS, GRID, GRID), dtype=np.float32)
    Q_all = np.empty((STEPS, GRID, GRID), dtype=np.float32)

    for t in range(STEPS):
        for i in range(N_ROBOTS):
            cell = walks[i][t]
            r_i = report_value(TRUE_MAP[cell], is_adv[i], cell)
            others = cell_reports.get(cell, [])
            if others:
                residuals = [abs(r_i - r_j) for _, r_j in others]
                D_sum[i] += float(np.mean(residuals))
                D_n[i] += 1
                for j, r_j in others:
                    if abs(r_i - r_j) > CONTRA_THRESH:
                        c[i] += 1
                        c[j] += 1
            cell_reports.setdefault(cell, []).append((i, r_i))

        mu_c, sigma_c = c.mean(), c.std()
        z = (c - mu_c) / (sigma_c + EPSILON)
        D = np.divide(D_sum, D_n, out=np.zeros_like(D_sum), where=D_n > 0)
        c_norm = c / (t + 1)

        feats = torch.tensor(
            np.stack([c_norm, z, D], axis=1), dtype=torch.float32, device=DEVICE
        )
        with torch.no_grad():
            alpha = alpha_net(feats).cpu().numpy()

        W_sum = np.zeros((GRID, GRID))
        A_sum = np.zeros((GRID, GRID))
        N_map = np.zeros((GRID, GRID))
        for cell, reports in cell_reports.items():
            idx = np.array([r for r, _ in reports])
            vals = np.array([v for _, v in reports])
            a = alpha[idx]
            W_sum[cell] = (a * vals).sum()
            A_sum[cell] = a.sum()
            N_map[cell] = len(reports)

        M_t = np.where(A_sum > 0, W_sum / np.maximum(A_sum, 1e-8), PRIOR)
        U_t = A_sum
        N_t = N_map

        e_t = TRUE_MAP - M_t
        err_old = e_t ** 2
        dir_t = np.where(M_t >= TRUE_MAP, 1.0, -1.0)

        X_all[t, 0] = M_t
        X_all[t, 1] = U_t
        X_all[t, 2] = N_t
        E_all[t] = err_old
        Q_all[t] = dir_t

    return X_all, E_all, Q_all


def main():
    alpha_net = AlphaNet().to(DEVICE)
    alpha_net.load_state_dict(torch.load("alpha_net.pt", map_location=DEVICE))
    alpha_net.eval()

    approx_gb_per_ep = (
        SAMPLES_PER_EPISODE * (3 + 1 + 1) * GRID * GRID * 4
    ) / (1024 ** 3)
    print(
        f"[datagen] ~{approx_gb_per_ep:.4f} GB / episode  "
        f"-> ~{approx_gb_per_ep * N_EPISODES:.1f} GB total for {N_EPISODES} episodes"
    )

    for ep in range(N_EPISODES):
        X_all, E_all, Q_all = rollout_episode_full(alpha_net)

        idx = rng.choice(STEPS, size=SAMPLES_PER_EPISODE, replace=False)
        idx.sort()

        X_sample = X_all[idx]
        E_sample = E_all[idx]
        Q_sample = Q_all[idx]

        np.save(os.path.join(X_DIR, f"ep{ep:05d}.npy"), X_sample)
        np.save(os.path.join(E_DIR, f"ep{ep:05d}.npy"), E_sample)
        np.save(os.path.join(Q_DIR, f"ep{ep:05d}.npy"), Q_sample)

        if (ep + 1) % 25 == 0:
            total_samples = (ep + 1) * SAMPLES_PER_EPISODE
            print(
                f"[datagen] episode {ep+1}/{N_EPISODES} done  "
                f"({total_samples} samples saved so far)"
            )

    print("done generating dataset")


if __name__ == "__main__":
    main()
