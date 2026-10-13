import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import load_hmda, prepare_outcome, select_columns
from .features import build_preprocessor
from .models import candidate_models
from .evaluation import classification_metrics, metrics_frame


def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    project, spec = cfg["project"], cfg["modeling"]
    df = prepare_outcome(load_hmda(project["raw_file"]))
    df = select_columns(df, spec["numeric_features"], spec["categorical_features"], spec["audit_features"], spec["exclude_leakage"])
    usable_num = [c for c in spec["numeric_features"] if c in df.columns]
    usable_cat = [c for c in spec["categorical_features"] if c in df.columns]
    X = df[usable_num + usable_cat]
    y = df["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=project["test_size"], stratify=y, random_state=project["random_state"])
    results = {}
    Path('models').mkdir(exist_ok=True)
    Path('reports/tables').mkdir(parents=True, exist_ok=True)
    for name, model in candidate_models(project["random_state"]).items():
        pipe = Pipeline([("preprocessor", build_preprocessor(usable_num, usable_cat)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        results[name] = classification_metrics(y_test, pred, score)
        joblib.dump(pipe, f"models/{name}.joblib")
    table = metrics_frame(results)
    table.to_csv('reports/tables/model_performance.csv')
    print(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    run(args.config)
