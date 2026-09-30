"""
Graphical abstract for the QSARCert v2 manuscript (Elsevier: 531 x 1328 px minimum, readable at
5 x 13 cm). Built only from validation/results/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
INK, INK2 = "#0b0b0b", "#52514e"
NEW, OLD = "#2a78d6", "#e87ba4"


def main():
    d = pd.read_csv(os.path.join(RES, "benchmark.csv"))
    scan = pd.read_csv(os.path.join(RES, "offset_scan.csv"))
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                                "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                                "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
    a = fig.add_axes([0.08, 0.22, 0.38, 0.52])
    b = fig.add_axes([0.60, 0.22, 0.38, 0.52])

    checks = [("domain", "ad110_status", "ad_status"), ("Y-random.", "yr110_status", "yr_status"),
              ("leakage", "lk110_status", "lk_status")]
    x = np.arange(len(checks))
    fail_old = [(d[o] == "FAIL").mean() for _, o, _ in checks]
    fail_new = [(d[n] == "FAIL").mean() for _, _, n in checks]
    na_new = [(d[n] == "NOT_APPLICABLE").mean() for _, _, n in checks]
    a.bar(x - 0.19, fail_old, 0.36, color=OLD, label="1.1.0: FAIL")
    a.bar(x + 0.19, fail_new, 0.36, color=NEW, label="1.2: FAIL")
    a.bar(x + 0.19, na_new, 0.36, bottom=fail_new, color="#b9d3f2", label="1.2: not applicable")
    a.set_xticks(x, [c for c, _, _ in checks], fontsize=7)
    a.set_yticks([0, 0.5, 1], ["0", "50%", "100%"], fontsize=7)
    a.set_ylim(0, 1.32)
    a.set_title("Verdicts on 180 real models", fontsize=8, color=INK)
    a.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=3, borderaxespad=0.0, handlelength=1.0, columnspacing=0.8)

    g = scan.groupby("shift_sd")[["centred", "uncentred"]].mean()
    b.plot(g.index, g.centred, color=NEW, marker="o", ms=2.5, label="centred $R_0^2$")
    b.plot(g.index, g.uncentred, color=OLD, marker="^", ms=2.5, label="uncentred $R_0^2$")
    b.set_ylim(0, 0.6)
    b.set_yticks([0, 0.25, 0.5], ["0", "25%", "50%"], fontsize=7)
    b.set_xticks([-3, 0, 3], ["-3", "0", "+3"], fontsize=7)
    b.set_xlabel("response origin shifted (SD)", fontsize=7)
    b.set_title("Golbraikh-Tropsha pass rate", fontsize=8, color=INK)
    b.legend(fontsize=6, frameon=False, loc="lower center", borderaxespad=0.1)
    fig.text(0.5, 0.975, "QSARCert: QSAR validation checks, correct but not always informative",
             ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    out = os.path.join(HERE, "figures", "graphical_abstract")
    fig.savefig(out + ".png", dpi=400)
    fig.savefig(out + ".pdf")
    from PIL import Image
    im = Image.open(out + ".png").convert("RGB")
    im.save(out + ".tif", compression="tiff_lzw", dpi=(400, 400))
    print("graphical abstract", im.size[0], "x", im.size[1], "px")


if __name__ == "__main__":
    main()
