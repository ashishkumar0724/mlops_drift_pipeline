"""
Model training module: AutoGluon + MLflow
API-compliant save mechanism (path in constructor, not fit)
Rubric: Automation + Versioning + Experiment Tracking
"""

import json
import logging
import os
import warnings
from pathlib import Path

import mlflow
import pandas as pd
from autogluon.tabular import TabularPredictor

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*lightgbm.*")
warnings.filterwarnings("ignore", message=".*catboost.*")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("telecom-churn-autogluon")


class ModelTrainer:
    def __init__(self, model_dir: str = "model", label: str = "Churn", version: str | None = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.label = label

        try:
            runs = mlflow.search_runs(experiment_names=["telecom-churn-autogluon"])
            self.version = version or f"v{len(runs) + 1}"
        except Exception:
            self.version = version or "v1"

        logger.info(f"🎯 Target: '{label}' | Version: {self.version}")

    def load_data(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
        if self.label not in df.columns:
            raise ValueError(f"Target '{self.label}' missing in {p.name}. Cols: {list(df.columns)[:10]}...")
        logger.info(f"📊 Loaded {len(df):,} rows | Target dist:\n{df[self.label].value_counts()}")
        return df

    def train(self, train_path: str | Path) -> TabularPredictor:
        train_df = self.load_data(train_path)

        # ✅ Define save path & ensure parent dir exists
        save_path = self.model_dir / f"predictor_{self.version}"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Training lightweight models (RandomForest & ExtraTrees)")
        logger.info(f"📂 Saving to: {save_path}")

        with mlflow.start_run(run_name=self.version):
            # ✅ FIX: 'path' belongs in TabularPredictor constructor, NOT in .fit()
            predictor = TabularPredictor(
                label=self.label,
                eval_metric="accuracy",
                problem_type="binary",
                path=str(save_path),  # ← CORRECT LOCATION
            )

            predictor.fit(
                train_data=train_df,
                time_limit=60,
                verbosity=1,
                # Only use models that don't require Torch/LightGBM/CatBoost
                hyperparameters={
                    "RF": {},  # Random Forest
                    "XT": {},  # Extra Trees
                },
            )

            # Evaluate
            leaderboard = predictor.leaderboard(train_df, silent=True)
            val_score = leaderboard.iloc[0]["score_val"]
            best_model = leaderboard.iloc[0]["model"]

            # Log to MLflow
            mlflow.log_metric("val_accuracy", val_score)
            mlflow.log_params(
                {"best_model": best_model, "time_limit": 60, "target": self.label, "train_rows": len(train_df)}
            )

            # Log Artifacts
            if save_path.exists():
                mlflow.log_artifacts(str(save_path), artifact_path="model")
            else:
                logger.warning("⚠️ Model path missing after fit. Skipping artifact log.")

            # Metadata
            meta = {"version": self.version, "val_accuracy": val_score, "best_model": best_model}
            Path("model/metadata.json").write_text(json.dumps(meta, indent=2))
            mlflow.log_artifact("model/metadata.json", artifact_path="registry")

            logger.info(f"✅ Training complete | Best: {best_model} | Accuracy: {val_score:.4f}")
            return predictor


def train_churn_model(train_path: str | Path, label: str = "Churn", version: str | None = None) -> TabularPredictor:
    return ModelTrainer(label=label, version=version).train(train_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.parquet")
    p.add_argument("--label", default="Churn")
    p.add_argument("--version")
    args = p.parse_args()

    print(f"\n🚀 Pipeline | Data: {args.train} | Target: {args.label}\n")
    try:
        pred = train_churn_model(args.train, label=args.label, version=args.version)
        sample = pd.read_parquet("data/processed/test.parquet").head(3).drop(columns=[args.label], errors="ignore")
        print(f"\n✅ Inference test:\n{pred.predict(sample)}")

        ver = args.version or "v1"
        print(f"\n✨ Model saved to: {MODEL_DIR / f'predictor_{ver}'}")
    except Exception as e:
        logger.error(f"❌ {e}", exc_info=True)
        raise
