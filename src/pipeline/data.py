"""
Data ingestion module: Kaggle download via kagglehub + target creation + drift simulation
Dataset: Telecom Customer Churn (Maven Analytics)
Rubric: Automation + Drift Detection foundations
"""

import glob
import logging
import random
import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
for d in [DATA_DIR, RAW_DIR, PROC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 🎯 DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DEFAULT_DATASET_SLUG = "shilongzhuang/telecom-customer-churn-by-maven-analytics"
DEFAULT_TARGET_COLUMN = "Churn"  # We will CREATE this binary column
DEFAULT_DRIFT_COLUMNS: list[str] | None = None
DEFAULT_DRIFT_MAGNITUDE = 0.3
DEFAULT_TEST_SIZE = 0.2
# ═══════════════════════════════════════════════════════════════


class DataManager:
    def __init__(self, data_dir: str = "data", seed: int = 42, **kwargs):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        self.dataset_slug = kwargs.get("dataset_slug", DEFAULT_DATASET_SLUG)
        self.target_column = kwargs.get("target_column", DEFAULT_TARGET_COLUMN)
        self.drift_columns = kwargs.get("drift_columns", DEFAULT_DRIFT_COLUMNS)
        self.drift_magnitude = kwargs.get("drift_magnitude", DEFAULT_DRIFT_MAGNITUDE)
        self.test_size = kwargs.get("test_size", DEFAULT_TEST_SIZE)

    def download_dataset(
        self, dataset_slug: str | None = None, output_filename: str = "raw_data.csv", force_redownload: bool = False
    ) -> Path:
        """Download dataset from Kaggle using kagglehub (exact structure you requested)."""
        slug = dataset_slug or self.dataset_slug
        out = PROC_DIR / output_filename

        if out.exists() and not force_redownload:
            logger.info(f"✅ Cached: {out}")
            return out

        logger.info(f"📥 Downloading {slug} via kagglehub...")
        try:
            dl_path = kagglehub.dataset_download(slug)
            csvs = [f for f in glob.glob(f"{dl_path}/*.csv") if Path(f).is_file()]
            if not csvs:
                raise FileNotFoundError(f"No CSV files found in Kaggle download for {slug}")
            shutil.copy2(csvs[0], out)
            logger.info(f"✅ Downloaded & saved to {out}")
            return out
        except Exception as e:
            logger.error(f"❌ Kaggle download failed: {e}")
            raise

    def load_data(self, filepath: str | Path) -> pd.DataFrame:
        """Load CSV/Parquet with basic validation."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
        logger.info(f"📊 Loaded {len(df):,} rows, {len(df.columns)} columns from {p.name}")
        return df

    def create_binary_churn_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized binary 'Churn' column from 'Customer Status'."""
        if "Customer Status" not in df.columns:
            raise ValueError("Missing 'Customer Status' column for target creation")

        status = df["Customer Status"].astype(str).str.strip().str.lower()
        df["Churn"] = (status == "churned").astype(int)

        logger.info(f"✅ Created binary 'Churn' from 'Customer Status'")
        logger.info(f"   Distribution: {df['Churn'].value_counts().to_dict()}")
        return df

    def simulate_drift(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_cols: list[str] | None = None,
        mag: float | None = None,
        out_name: str = "test_drifted.csv",
    ) -> pd.DataFrame:
        df_d = df.copy()
        magnitude = mag or self.drift_magnitude
        cols = (
            drift_cols
            or self.drift_columns
            or [
                c
                for c in df_d.select_dtypes(include="number").columns
                if c != target_col and not c.lower().startswith(("customer", "id", "index", "zip"))
            ]
        )
        if not cols:
            logger.warning("⚠️ No numeric cols for drift")
            return df_d

        logger.info(f"🔄 Drift sim (mag={magnitude}) on {len(cols)} cols")
        for c in cols:
            if c not in df_d.columns or pd.api.types.is_string_dtype(df_d[c]):
                continue
            if df_d[c].isna().all():
                continue
            std = df_d[c].std()
            if pd.isna(std) or std == 0:
                continue
            shift = random.choice([-1, 1]) * magnitude * std
            scale = 1 + random.uniform(-magnitude / 2, magnitude / 2)
            df_d[c] = (df_d[c] * scale) + shift
            df_d[c] = df_d[c].clip(df_d[c].quantile(0.01), df_d[c].quantile(0.99))

        out = PROC_DIR / out_name
        df_d.to_csv(out, index=False)
        logger.info(f"💾 Saved drifted: {out}")
        return df_d

    def prepare_datasets(
        self, dataset_slug: str | None = None, force_redownload: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 1. Download (signature now matches call)
        raw_path = self.download_dataset(dataset_slug, force_redownload=force_redownload)

        # 2. Load
        df = self.load_data(raw_path)

        # 3. Create binary target BEFORE split
        df = self.create_binary_churn_target(df)

        # 4. Stratified split
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.seed, stratify=df["Churn"])

        # 5. Save processed
        train_df.to_parquet(PROC_DIR / "train.parquet", index=False)
        test_df.to_parquet(PROC_DIR / "test.parquet", index=False)
        logger.info(f"✅ Saved train/test to {PROC_DIR}")

        # 6. Simulate drift on test set
        drifted_df = self.simulate_drift(test_df, target_col="Churn")
        logger.info("✅ Dataset prep complete")
        return train_df, test_df, drifted_df


def prepare_data_for_pipeline(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return DataManager(**kwargs).prepare_datasets()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET_SLUG)
    p.add_argument("--drift-mag", type=float, default=DEFAULT_DRIFT_MAGNITUDE)
    args = p.parse_args()
    train, test, drifted = prepare_data_for_pipeline(dataset_slug=args.dataset, drift_magnitude=args.drift_mag)
    print(f"\n📦 Summary: Train={train.shape}, Test={test.shape}, Drifted={drifted.shape}")
    print(f"   Target 'Churn' exists: {'Churn' in train.columns}")
    print(f"   Distribution: {train['Churn'].value_counts().to_dict()}")
