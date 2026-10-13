from pathlib import Path
import matplotlib.pyplot as plt
import shap


def save_shap_summary(model, X_transformed, feature_names, output_path, max_display=20):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.Explainer(model, X_transformed, feature_names=feature_names)
    values = explainer(X_transformed)
    shap.plots.beeswarm(values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    return values
