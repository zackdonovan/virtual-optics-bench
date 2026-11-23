"""Render an illustrative transfer-function figure, not scored benchmark data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    f = np.geomspace(.5, 10000, 640)
    w = 2 * np.pi * f
    tau_opt, tau_det, tau_th = 30e-6, 220e-6, 4e-3
    opt = 1 / (1 + 1j * w * tau_opt)
    det = 1 / (1 + 1j * w * tau_det)
    th = 1 / (1 + 1j * w * tau_th)
    dark = "#142329"
    muted = "#54696f"
    bg = "#f8faf9"
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.3), facecolor=bg)
    fig.text(.085, .93, "Same static response. Different dynamics.", fontsize=20,
             weight="semibold", color=dark)
    fig.text(.085, .875, "Illustrative optical-cavity model  /  equal static gain, three mechanisms",
             fontsize=10, color=muted)
    for ax in axs:
        ax.set_facecolor(bg)
        ax.set_xscale("log")
        ax.grid(axis="y", color="#d6dfdc", linewidth=.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#9db0ae")
        ax.tick_params(colors=muted)
        ax.set_xlabel("modulation frequency (Hz)", color=dark)
        ax.axvspan(1 / (2 * np.pi * tau_det), f[-1], color="#789088", alpha=.11, lw=0)
    for name, frac, color in (("Kerr", 0, "#005b79"), ("Mixed", .5, "#926427"), ("Thermal", 1, "#a43454")):
        response = opt * det * (1 - frac + frac * th)
        axs[0].plot(f, abs(response), lw=2.6, color=color, label=name)
        axs[1].plot(f, np.rad2deg(np.unwrap(np.angle(response))), lw=2.6, color=color)
    axs[0].set_ylabel("measured amplitude / static amplitude", color=dark)
    axs[1].set_ylabel("unwrapped phase (degrees)", color=dark)
    axs[0].legend(loc="lower left", frameon=False, labelcolor=dark)
    axs[1].annotate("above detector\ncorner frequency", xy=(1700, -45), xytext=(62, -10),
                    arrowprops={"arrowstyle": "-", "color": muted}, color=muted, fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, top=.79, bottom=.15, wspace=.31)
    out = Path(__file__).resolve().parents[1] / "figures" / "transfer-functions.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
