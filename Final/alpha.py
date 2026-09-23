import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Config — 15 robots, 30% max adversarial fraction, 100x100 grid
# ----------------------------------------------------------------------
GRID = 100
N_ROBOTS = 15
ADV_FRAC_MAX = 0.3
STEPS = 1000
EPSILON = 0.05
CONTRA_THRESH = 0.2
SEED = 40

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEVICE}")


# ----------------------------------------------------------------------
# Environment helpers
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


# ----------------------------------------------------------------------
# AlphaNet
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


def train_alpha_net(episodes=1000, lr=2e-3):
    alpha_net = AlphaNet().to(DEVICE)
    optimizer = torch.optim.Adam(alpha_net.parameters(), lr=lr)
    loss_history = []

    for ep in range(episodes):
        TRUE_MAP = make_true_map(GRID)
        n_adv = rng.integers(1, int(N_ROBOTS * ADV_FRAC_MAX) + 1)
        is_adv = make_adv_mask(N_ROBOTS, n_adv)
        walks = [random_walk(GRID, STEPS) for _ in range(N_ROBOTS)]

        c = np.zeros(N_ROBOTS)
        D_sum = np.zeros(N_ROBOTS)
        D_n = np.zeros(N_ROBOTS)
        cell_reports = {}
        ep_loss = 0.0

        for t in range(STEPS):
            touched_cells = set()

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
                touched_cells.add(cell)

            mu_c, sigma_c = c.mean(), c.std()
            z = (c - mu_c) / (sigma_c + EPSILON)
            D = np.divide(D_sum, D_n, out=np.zeros_like(D_sum), where=D_n > 0)
            c_norm = c / (t + 1)

            feats = torch.tensor(
                np.stack([c_norm, z, D], axis=1),
                dtype=torch.float32,
                device=DEVICE,
            )

            alpha = alpha_net(feats)

            step_loss = 0.0

            for cell in touched_cells:
                reports = cell_reports[cell]
                idx = torch.tensor([r for r, _ in reports], device=DEVICE)
                vals = torch.tensor(
                    [v for _, v in reports],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                a = alpha[idx]
                M = (a * vals).sum() / (a.sum() + EPSILON)

                step_loss = step_loss + (TRUE_MAP[cell] - M) ** 2

            step_loss = step_loss / len(touched_cells)

            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            ep_loss += step_loss.item()

        loss_history.append(ep_loss / STEPS)

        print(
            f"[alpha_net] episode {ep+1}/{episodes}  "
            f"avg loss {loss_history[-1]:.5f}"
        )

    return alpha_net, loss_history


if __name__ == "__main__":
    ALPHA_EPISODES = 1000

    print("=== training alpha_net (GRID=100, N_ROBOTS=15, ADV_FRAC_MAX=0.3) ===")
    alpha_net, alpha_loss = train_alpha_net(episodes=ALPHA_EPISODES)

    torch.save(alpha_net.state_dict(), "alpha_net.pt")
    print("\nsaved alpha_net.pt")
