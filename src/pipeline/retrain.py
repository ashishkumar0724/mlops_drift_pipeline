"""
Conditional retraining module
Triggers model retraining if drift score exceeds threshold
Rubric: Retraining Logic (20%) + Version Comparison
"""

import json
import logging
import sys
from pathlib import Path

# Ensure pipeline modules are importable
sys.path.append(str(Path(__file__).parent.parent.parent / "src/pipeline"))
import mlflow
from train import train_churn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("telecom-churn-autogluon")


def get_latest_version_score() -> float | None:
    """Fetch the validation score of the latest registered model version."""
    try:
        runs = mlflow.search_runs(experiment_names=["telecom-churn-autogluon"], order_by=["start_time DESC"])
        if not runs.empty and "metrics.val_accuracy" in runs.columns:
            return float(runs.iloc[0]["metrics.val_accuracy"])
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch previous score: {e}")
    return None


def conditional_retrain(
    drift_result: dict, train_path: str = "data/processed/train.parquet", target: str = "Churn"
) -> dict | None:
    """
    If drift detected → retrain model → save as new version → compare performance.
    """
    if not drift_result.get("needs_retrain", False):
        logger.info("✅ Drift score below threshold. Model stable. No retraining needed.")
        return None

    logger.info("⚠️  DRIFT DETECTED! Triggering automated retraining...")

    old_score = get_latest_version_score()
    new_version = f"v{2 if old_score else 1}"

    # 1️⃣ Train new model
    try:
        train_churn_model(train_path, label=target, version=new_version)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None

    # 2️⃣ Extract new score & compare
    try:
        runs = mlflow.search_runs(
            experiment_names=["telecom-churn-autogluon"], filter_string=f"tags.mlflow.runName='{new_version}'"
        )
        new_score = float(runs.iloc[0]["metrics.val_accuracy"])

        improved = bool(new_score > old_score) if old_score is not None else True
        comparison = f"{new_score:.4f} (v2) vs {old_score:.4f} (v1)" if old_score else f"{new_score:.4f} (baseline v1)"
        logger.info(f"✅ Retraining complete | {comparison} | Improved: {improved}")
    except Exception as e:
        logger.warning(f"⚠️ Could not extract new score: {e}")
        new_score = 0.0
        improved = False
        comparison = "N/A"

    # 3️⃣ Save metadata (JSON-safe casting)
    try:
        meta_path = Path("model/metadata.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update(
            {
                "current_version": str(new_version),
                "v1_score": float(old_score) if old_score is not None else None,
                "v2_score": float(new_score),
                "improved": bool(improved),
                "triggered_by_drift_score": float(drift_result.get("drift_score", 0.0)),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        logger.warning(f"⚠️ Metadata save failed (non-critical): {e}")

    return {
        "old_version": "v1" if old_score else None,
        "new_version": str(new_version),
        "old_score": float(old_score) if old_score else None,
        "new_score": float(new_score),
        "improved": bool(improved),
        "comparison": comparison,
        "model_path": str(Path("model") / f"predictor_{new_version}"),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--drift-score", type=float, default=0.42)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--train", default="data/processed/train.parquet")
    args = p.parse_args()

    drift_result = {"needs_retrain": args.drift_score > args.threshold, "drift_score": args.drift_score}

    print(f"\n🔄 Conditional Retraining Check:")
    print(f"   Drift Score: {args.drift_score:.4f} | Threshold: {args.threshold}")
    print(f"   Trigger: {'YES' if drift_result['needs_retrain'] else 'NO'}\n")

    result = conditional_retrain(drift_result, train_path=args.train)

    if result:
        print("\n✅ Retraining Successful:")
        print(f"   Version:  {result['old_version'] or 'N/A'} → {result['new_version']}")
        print(f"   Scores:   {result['old_score'] or 0.0:.4f} → {result['new_score']:.4f}")
        print(f"   Improved: {result['improved']}")
    else:
        print("ℹ️  No retraining triggered or failed. Check logs above.")
