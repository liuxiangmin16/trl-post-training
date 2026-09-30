"""预览本地 OpenR1-Math parquet 数据集的结构和完整样本。"""
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pa_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_OPENR1_DIR = ROOT / "data" / "raw" / "OpenR1"
PREVIEW_PATH = ROOT / "data" / "sample_preview" / "openr1_raw_sample.json"


def get_parquet_files():
    files = sorted(RAW_OPENR1_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"OpenR1 parquet 不存在: {RAW_OPENR1_DIR}")
    return files


def preview_openr1(sample_rows=1):
    """用 PyArrow 读取完整数据集信息，用 pandas 展示未截断的样本。"""
    files = get_parquet_files()
    dataset = pa_dataset.dataset([str(path) for path in files], format="parquet")
    sample_df = dataset.head(sample_rows).to_pandas()

    print("\n===== OpenR1 raw dataset =====")
    print(f"parquet files: {len(files)}")
    print(f"rows: {dataset.count_rows()}")
    print("schema:")
    print(dataset.schema)
    print("\npandas dtypes:")
    print(sample_df.dtypes)

    # pandas 默认会截断长字符串和 list；这里完整打印所有列内容。
    with pd.option_context(
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.max_seq_items", None,
        "display.width", None,
    ):
        print("\nfull sample:")
        print(sample_df.to_string(index=False))

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREVIEW_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            sample_df.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nsaved full sample: {PREVIEW_PATH}")


if __name__ == "__main__":
    preview_openr1()
