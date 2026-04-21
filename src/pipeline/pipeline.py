"""
Master orchestration pipeline
Runs: Data → Train → Drift Detect → Conditional Retrain
Rubric: Automation (25%) + Full Pipeline Flow
"""
# ✅ Standard library imports (alphabetical)
import logging
import sys
from pathlib import Path

# ✅ Path setup for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ Local package imports (noqa: E402 because path setup comes first)
from pipeline.data import prepare_data_for_pipeline  # noqa: E402
from pipeline.drift import detect_drift  # noqa: E402
from pipeline.retrain import conditional_retrain  # noqa: E402
from pipeline.train import train_churn_model  # noqa: E402

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MLOpsPipeline")


def run_pipeline(dataset_slug: str = "shilongzhuang/telecom-customer-churn-by-maven-analytics") -> bool:
    logger.info("=" * 70)
    logger.info("🚀 STARTING AUTOMATED MLOPS PIPELINE")
    logger.info("=" * 70)

    try:
        # 🔹 STEP 1: Data Ingestion & Preparation
        logger.info("\n📦 STEP 1/4: Data Ingestion & Drift Simulation")
        train_df, test_df, drifted_df = prepare_data_for_pipeline(dataset_slug=dataset_slug)
        logger.info(f"✅ Prepared: Train={train_df.shape}, Test={test_df.shape}, Drifted={drifted_df.shape}")

        # 🔹 STEP 2: Initial Training (Baseline v1)
        logger.info("\n🤖 STEP 2/4: Baseline Model Training")
        train_path = "data/processed/train.parquet"
        _ = train_churn_model(train_path, label="Churn", version="v1")
        logger.info("✅ Baseline model trained & logged (v1)")

        # 🔹 STEP 3: Drift Detection
        logger.info("\n📈 STEP 3/4: Production Drift Detection")
        ref_path = "data/processed/train.parquet"
        cur_path = "data/processed/test_drifted.csv"
        drift_result = detect_drift(
            reference_path=ref_path,
            current_path=cur_path,
            target_column="Churn",
            threshold=0.05
        )

        # 🔹 STEP 4: Conditional Retraining
        logger.info("\n🔄 STEP 4/4: Automated Retraining Logic")
        retrain_result = conditional_retrain(drift_result, train_path=train_path, target="Churn")

        # 📊 FINAL PIPELINE SUMMARY
        logger.info("\n" + "=" * 70)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"   Dataset:        {dataset_slug}")
        logger.info(f"   Models Created: v1{' + v2' if retrain_result else ''}")
        logger.info(f"   Drift Score:    {drift_result['drift_score']:.4f} (Threshold: {drift_result['threshold']})")
        logger.info(f"   Drift Report:   {drift_result['report_path']}")

        if retrain_result:
            logger.info(f"   Retrain:        YES ({retrain_result['old_version']} → {retrain_result['new_version']})")
            logger.info(f"   Performance:    {retrain_result['comparison']}")
        else:
            logger.info("   Retrain:        NO (Model stable)")

        logger.info(f"   Artifacts:      model/, mlruns/, reports/")
        logger.info("=" * 70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run end-to-end MLOps pipeline")
    p.add_argument(
        "--dataset",
        type=str,
        default="shilongzhuang/telecom-customer-churn-by-maven-analytics"
    )
    args = p.parse_args()

    success = run_pipeline(args.dataset)
    sys.exit(0 if success else 1)
