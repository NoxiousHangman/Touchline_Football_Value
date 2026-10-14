"""Export a standalone Matplotlib evaluation figure, without a desktop GUI."""

import os
from pathlib import Path
import tempfile

# This app's sandbox may not allow Matplotlib's usual user-cache directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "football-value-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ATTRIBUTION


def evaluation_plot(heldout, report: dict, path: Path) -> Path:
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#f6f8fb"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        actual = heldout.market_value_gbp / 1e6
        predicted = heldout.predicted_market_value_gbp / 1e6
        axes[0, 0].scatter(actual, predicted, color="#166c80", alpha=.65, s=25)
        bounds = [0, max(250, actual.max() * 1.05, predicted.max() * 1.05)]
        axes[0, 0].plot(bounds, bounds, "--", color="#8f9caf", linewidth=1)
        axes[0, 0].set(title="Actual vs predicted", xlabel="Actual market value (£m)", ylabel="Predicted value (£m)",
                       xlim=bounds, ylim=bounds)
        axes[0, 1].scatter(predicted, heldout.residual_gbp / 1e6, color="#166c80", alpha=.65, s=25)
        axes[0, 1].axhline(0, color="#8f9caf", linestyle="--")
        axes[0, 1].set(title="Residuals", xlabel="Predicted value (£m)", ylabel="Actual − predicted (£m)", xlim=bounds)
        axes[1, 0].scatter(heldout.age, actual, color="#166c80", alpha=.65, s=25)
        axes[1, 0].set(title="Age and observed value", xlabel="Age at feature observation", ylabel="Actual market value (£m)", ylim=bounds)
        positions = np.arange(2)
        model = [report["metrics"][key] / 1e6 for key in ("mae_gbp", "rmse_gbp")]
        baseline = [report["median_baseline_metrics"][key] / 1e6 for key in ("mae_gbp", "rmse_gbp")]
        axes[1, 1].bar(positions - .18, model, width=.36, color="#166c80", label="Linear Regression")
        axes[1, 1].bar(positions + .18, baseline, width=.36, color="#b8c3d4", label="Training-median baseline")
        axes[1, 1].set_xticks(positions, ["MAE", "RMSE"])
        axes[1, 1].set(title="Holdout errors · lower is better", ylabel="Error (£m)")
        axes[1, 1].legend(fontsize=8)
        title = "SYNTHETIC DEMO · not real valuation accuracy" if report["is_synthetic"] else "Football market-value model · chronological holdout"
        fig.suptitle(title + f"\nTrain: {report['train_rows']} rows  |  Test: {report['test_rows']} rows", fontsize=15)
        source = ("Transfermarkt via dcaribou/transfermarkt-datasets" if report.get("player_id_namespace") == "transfermarkt"
                  else ATTRIBUTION + " · Valuation source recorded in metrics.json")
        credit = "Invented players, statistics and valuations · software demonstration only" if report["is_synthetic"] else source
        fig.supxlabel(credit, fontsize=8)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return path
