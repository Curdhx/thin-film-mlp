"""AI for Thin Films -- MLP Mini Research Project (main pipeline).

Student-specific parameters:
    student number (last 6 digits) seed = 276029
    last two digits N = 29  ->  lambda_target = 450 + 10 * (29 mod 31) = 740 nm
    design_seed = seed + 1 = 276030

Workflow:
    1. TMM data generation (5000 samples, fixed 4000/500/500 split)
    2. MLP surrogate training (4-128-128-64-41, MSE)
    3. training-size experiment (500 / 1000 / 2000 / 4000)
    4. MLP-assisted screening (10000 candidates -> top10 -> TMM -> top5)
    5. figures (8) and result persistence
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tmm import (WAVELENGTHS, D_MIN, D_MAX, N_H, N_L, N_S,
                 spectrum_batch, reflectance)
from mlp import MLP

# ---------------- student-specific parameters ----------------
SEED = 276029
DESIGN_SEED = SEED + 1                       # 276030
N_STUDENT = SEED % 100                       # 29
LAMBDA_TARGET = 450 + 10 * (N_STUDENT % 31)  # 740 nm
TARGET_IDX = int(round((LAMBDA_TARGET - WAVELENGTHS[0]) / 10))

N_TOTAL, N_TRAIN, N_VAL, N_TEST = 5000, 4000, 500, 500
MLP_SIZES = [4, 128, 128, 64, 41]
EPOCHS = 500
BATCH_SIZE = 128
LR = 1e-3
TRAIN_SIZES = [500, 1000, 2000, 4000]

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def norm(D):
    """Map thicknesses from [40, 180] nm to [0, 1]."""
    return (D - D_MIN) / (D_MAX - D_MIN)


def mse_rmse(Y, P):
    mse = float(np.mean((Y - P) ** 2))
    return mse, float(np.sqrt(mse))


def main():
    print("=" * 72)
    print("AI for Thin Films -- MLP Mini Research Project")
    print("seed = %d, design_seed = %d, N = %d" % (SEED, DESIGN_SEED, N_STUDENT))
    print("lambda_target = %d nm (index %d, wavelength %.1f nm)"
          % (LAMBDA_TARGET, TARGET_IDX, WAVELENGTHS[TARGET_IDX]))
    print("=" * 72)

    # ---------------- 1. TMM data generation ----------------
    print("\n[1] TMM data generation (5000 samples, fixed split 4000/500/500)")
    rng = np.random.RandomState(SEED)
    D_all = rng.uniform(D_MIN, D_MAX, size=(N_TOTAL, 4))
    perm = rng.permutation(N_TOTAL)
    D_all = D_all[perm]                          # one fixed permutation
    print("    computing 5000 x 41 reflectance spectra via TMM ...")
    R_all = spectrum_batch(D_all)

    Xtr, Xva, Xte = (D_all[:N_TRAIN],
                     D_all[N_TRAIN:N_TRAIN + N_VAL],
                     D_all[N_TRAIN + N_VAL:])
    Ytr, Yva, Yte = (R_all[:N_TRAIN],
                     R_all[N_TRAIN:N_TRAIN + N_VAL],
                     R_all[N_TRAIN + N_VAL:])
    Xn_tr, Xn_va, Xn_te = norm(Xtr), norm(Xva), norm(Xte)
    print("    train %s, val %s, test %s" % (Xtr.shape, Xva.shape, Xte.shape))

    # ---------------- 2. MLP training ----------------
    print("\n[2] MLP training %s, Adam lr=%.0e, batch=%d, epochs=%d"
          % (MLP_SIZES, LR, BATCH_SIZE, EPOCHS))
    model = MLP(MLP_SIZES, seed=SEED)
    hist = model.train(Xn_tr, Ytr, Xval=Xn_va, Tval=Yva,
                       epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=0)
    Ptr = model.forward(Xn_tr)[0]
    Pva = model.forward(Xn_va)[0]
    Pte = model.forward(Xn_te)[0]
    print("    final metrics (MSE / RMSE):")
    m_tr = mse_rmse(Ytr, Ptr)
    m_va = mse_rmse(Yva, Pva)
    m_te = mse_rmse(Yte, Pte)
    print("      train  MSE=%.6e  RMSE=%.6e" % m_tr)
    print("      val    MSE=%.6e  RMSE=%.6e" % m_va)
    print("      test   MSE=%.6e  RMSE=%.6e" % m_te)

    # ---------------- 3. training-size experiment ----------------
    print("\n[3] training-size experiment %s" % TRAIN_SIZES)
    size_results = []
    for n in TRAIN_SIZES:
        sub = MLP(MLP_SIZES, seed=SEED)         # identical init for fair comparison
        sub.train(Xn_tr[:n], Ytr[:n], Xval=Xn_va, Tval=Yva,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=0,
                  verbose=False)
        mse, rmse = mse_rmse(Yte, sub.forward(Xn_te)[0])
        size_results.append((n, mse, rmse))
        print("    n_train=%5d: test MSE=%.6e  RMSE=%.6e" % (n, mse, rmse))

    # ---------------- 4. MLP-assisted screening ----------------
    print("\n[4] MLP-assisted screening (design_seed=%d, 10000 candidates, "
          "target %.0f nm)" % (DESIGN_SEED, LAMBDA_TARGET))
    rng_d = np.random.RandomState(DESIGN_SEED)
    D_cand = rng_d.uniform(D_MIN, D_MAX, size=(10000, 4))
    R_pred_all = model.forward(norm(D_cand))[0]
    R_pred_t = R_pred_all[:, TARGET_IDX]
    top10 = np.argsort(R_pred_t)[:10]           # minimize reflectance at target
    D_top10 = D_cand[top10]
    R_true_top10 = spectrum_batch(D_top10)[:, TARGET_IDX]
    R_pred_top10 = R_pred_t[top10]
    top5_local = np.argsort(R_true_top10)[:5]   # re-rank by TMM-verified value
    top5_global = top10[top5_local]

    print("    Top 5 designs (ranked by TMM-verified R at target):")
    print("    Rank |   d1     d2     d3     d4   |  MLP R     |  TMM R")
    for r in range(5):
        g = top5_global[r]
        j = top5_local[r]
        print("    %4d | %6.2f %6.2f %6.2f %6.2f | %.6f | %.6f"
              % (r + 1, D_cand[g, 0], D_cand[g, 1], D_cand[g, 2], D_cand[g, 3],
                 R_pred_t[g], R_true_top10[j]))

    # failure case (largest mean absolute error on the test set)
    err = np.abs(Yte - Pte).mean(axis=1)
    fail_idx = int(np.argmax(err))
    print("\n    failure case: test index %d, mean |error| = %.4f"
          % (fail_idx, err[fail_idx]))

    # ---------------- 5. figures ----------------
    print("\n[5] generating 8 figures ...")
    make_figures(model, hist, Xtr, Ytr, Xte, Yte, Pte, size_results,
                 D_cand, R_pred_all, top10, top5_global, top5_local,
                 D_top10, R_true_top10, err, fail_idx)

    # ---------------- persist ----------------
    np.savez("results.npz",
             D_all=D_all, R_all=R_all, D_cand=D_cand, test_pred=Pte,
             top5_idx=top5_global, top5_thickness=D_cand[top5_global],
             top5_mlp_R=R_pred_t[top5_global], top5_tmm_R=R_true_top10[top5_local],
             size_results=np.array(size_results),
             train_hist=np.array(hist["train"]), val_hist=np.array(hist["val"]),
             lambda_target=LAMBDA_TARGET, seed=SEED, design_seed=DESIGN_SEED)
    print("\nDone. results.npz saved; figures in '%s/'." % FIG_DIR)


# ---------------- figures ----------------
def make_figures(model, hist, Xtr, Ytr, Xte, Yte, Pte, size_results,
                 D_cand, R_pred_all, top10, top5_global, top5_local,
                 D_top10, R_true_top10, err, fail_idx):
    # ---- Figure 1: overall workflow (left-to-right) ----
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    steps = [
        ("TMM", "thickness \u2192 spectrum\n(physical model)"),
        ("Dataset", "5000 samples\n4000 / 500 / 500"),
        ("MLP surrogate", "4\u2013128\u2013128\u201364\u201341\nMSE loss"),
        ("Screening", "10000 candidates\n\u03bb_target = 740 nm"),
        ("TMM verify", "Top 5 designs\nfinal check"),
    ]
    n = len(steps)
    box_w, box_h = 2.0, 1.7
    gap = 0.55
    for i, (title, sub) in enumerate(steps):
        x = i * (box_w + gap)
        box = FancyBboxPatch((x, 0), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor="#e8f0fe", edgecolor="#1f4e9c",
                             linewidth=1.8)
        ax.add_patch(box)
        ax.text(x + box_w / 2, box_h * 0.72, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1f4e9c")
        ax.text(x + box_w / 2, box_h * 0.32, sub, ha="center", va="center",
                fontsize=9, color="#333333")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + box_w + 0.04, box_h / 2),
                                         (x + box_w + gap - 0.04, box_h / 2),
                                         arrowstyle="-|>", mutation_scale=24,
                                         linewidth=2.2, color="#1f4e9c"))
    total_w = n * box_w + (n - 1) * gap
    ax.set_xlim(-0.15, total_w + 0.15)
    ax.set_ylim(-0.35, box_h + 0.6)
    fig.savefig(os.path.join(FIG_DIR, "fig1_workflow.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: physical model + sample spectra ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    layers = [("Air", 1.00), ("H", N_H), ("L", N_L), ("H", N_H),
              ("L", N_L), ("Glass", N_S)]
    for i, (name, ni) in enumerate(layers):
        color = "#dbe7f5" if name in ("Air", "L") else (
            "#f5e6d0" if name == "Glass" else "#cfe3cf")
        ax1.add_patch(plt.Rectangle((0, i), 1.0, 1.0,
                                    facecolor=color, edgecolor="black"))
        ax1.text(1.05, i + 0.5, "%s (n=%.2f)" % (name, ni), va="center",
                 fontsize=8)
    ax1.set_xlim(0, 2.0)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.set_title("Air/H/L/H/L/Glass stack")
    rng = np.random.RandomState(1)
    for k in range(5):
        ax2.plot(WAVELENGTHS, Ytr[k], lw=1.2, alpha=0.8)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflectance")
    ax2.set_title("Sample TMM spectra (train)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_model_data.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: MLP architecture ----
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes = MLP_SIZES
    display = [4, 6, 6, 5, 6]   # visual node counts (real sizes labeled)
    xs = np.linspace(0, 4, len(sizes))
    node_pos = []
    for l, nd in enumerate(display):
        ys = np.linspace(0, 1, nd)
        node_pos.append((xs[l], ys))
        for y in ys:
            ax.add_patch(plt.Circle((xs[l], y), 0.03,
                                    facecolor="#e8f0fe", edgecolor="black"))
        ax.text(xs[l], -0.08, "%d" % sizes[l], ha="center", fontsize=9)
    for l in range(len(sizes) - 1):
        for y1 in node_pos[l][1]:
            for y2 in node_pos[l + 1][1]:
                ax.plot([xs[l], xs[l + 1]], [y1, y2], color="lightgray",
                        lw=0.5, alpha=0.5)
    ax.text(0, 1.12, "Input\n4 thicknesses", ha="center", fontsize=8)
    ax.text(4, 1.12, "Output\n41 reflectance", ha="center", fontsize=8)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-0.2, 1.25)
    ax.axis("off")
    fig.savefig(os.path.join(FIG_DIR, "fig3_mlp_arch.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: training/validation loss ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["train"], label="Training MSE", lw=1.5)
    ax.plot(hist["val"], label="Validation MSE", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "fig4_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 5: TMM vs MLP on 3 test samples ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(Xte), 3, replace=False)
    for k, idx in enumerate(sample_idx):
        axes[k].plot(WAVELENGTHS, Yte[idx], "o-", ms=4, lw=1.5,
                     label="TMM")
        axes[k].plot(WAVELENGTHS, Pte[idx], "s--", ms=3, lw=1.2,
                     label="MLP")
        axes[k].set_xlabel("Wavelength (nm)")
        axes[k].set_ylabel("Reflectance")
        axes[k].legend(fontsize=8)
        axes[k].set_title("test #%d" % idx)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_prediction.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 6: training-set size effect ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [s[0] for s in size_results]
    rmses = [s[2] for s in size_results]
    ax.plot(ns, rmses, "o-", lw=1.5)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Test RMSE")
    ax.set_xticks(ns)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "fig6_train_size.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 7: selected design, MLP vs TMM ----
    g_best = top5_global[0]                     # best by TMM-verified R
    d_best = D_cand[g_best]
    R_mlp_best = R_pred_all[g_best]
    R_tmm_best = spectrum_batch(d_best[None, :])[0]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(WAVELENGTHS, R_tmm_best, "o-", ms=4, lw=1.5, label="TMM")
    ax.plot(WAVELENGTHS, R_mlp_best, "s--", ms=3, lw=1.2, label="MLP")
    ax.axvline(LAMBDA_TARGET, color="red", ls=":", lw=1.5,
               label=r"$\lambda_{target}$=%.0f nm" % LAMBDA_TARGET)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig7_design.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 8: representative failure case ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(WAVELENGTHS, Yte[fail_idx], "o-", ms=4, lw=1.5, label="TMM")
    ax.plot(WAVELENGTHS, Pte[fail_idx], "s--", ms=3, lw=1.2, label="MLP")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig8_failure.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
