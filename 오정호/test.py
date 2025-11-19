# oulad_ablation_gpu.py
# -*- coding: utf-8 -*-
"""
OULAD feature ablation + multi-model evaluation (GPU 지원 버전)
- 모델: XGBoost (GPU / CPU 지원)
- baseline + 여러 feature 제거 조합(ablation)
- GroupKFold(id_student) 5-fold
- 각 모델 × 시나리오별로
    - fold별 성능표 CSV 저장
    - fold별 성능(표 + AUC/ACC 라인 그래프) PNG 저장
- 전체 요약 결과는 oulad_ablation_results.csv 로 저장

Usage 예시:
    # CPU만 사용 (기본)
    python oulad_ablation_gpu.py

    # 1번 GPU 사용 (사용자 기준 1~4 → CUDA index 0~3)
    python oulad_ablation_gpu.py --gpu-id 1

    # 4번 GPU 사용
    python oulad_ablation_gpu.py --gpu-id 4

    # feature 조합에서 최대 2개까지 제거 (조합 수 줄이기)
    python oulad_ablation_gpu.py --max-drop 2
"""

import os
import argparse
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

# =========================
# 1. Argument parsing & GPU 설정
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="OULAD feature ablation (GPU-enabled XGBoost only)")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="OULAD data directory (기본: data/)"
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=-1,
        help="1~4: 사용할 GPU 번호, -1 또는 0: GPU 사용 안 함 (CPU만)."
    )

    parser.add_argument(
        "--max-drop",
        type=int,
        default=3,
        help="한 시나리오에서 제거할 최대 feature 개수 (기본: 3, 너무 크게 하면 조합 폭발 주의)"
    )

    return parser.parse_args()


args = parse_args()

# GPU 설정: 사용자 기준 1~4 → CUDA_VISIBLE_DEVICES 0~3
if args.gpu_id is not None and args.gpu_id >= 1:
    cuda_idx = args.gpu_id - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
    print(f"[GPU] Using logical GPU index {args.gpu_id} -> CUDA_VISIBLE_DEVICES={cuda_idx}")
else:
    # CPU만 사용
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[GPU] GPU disabled, using CPU only.")

DATA_DIR = args.data_dir
OUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_DROP = max(1, args.max_drop)  # 최소 1개 이상은 제거 가능하도록

# =========================
# 2. Imports (GPU env 세팅 후)
# =========================

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일 저장용 백엔드
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# XGBoost만 사용
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("[INFO] xgboost available.")
except Exception:
    HAS_XGB = False
    print("[ERROR] xgboost not available. Please install xgboost.")
    # 나머지 부분에서 에러를 던지도록 두고 진행


# =========================
# 3. 데이터 로딩
# =========================

def load_features(data_dir: str) -> pd.DataFrame:
    """processed/oulad_features.parquet 또는 csv 로딩"""
    processed_dir = os.path.join(data_dir, "processed")
    parquet_path = os.path.join(processed_dir, "oulad_features.parquet")
    csv_path = os.path.join(processed_dir, "oulad_features.csv")

    if os.path.exists(parquet_path):
        print(f"[DATA] Loading {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"[DATA] Loading {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"processed OULAD features not found. "
            f"Expected: {parquet_path} or {csv_path}"
        )

    return df


df_use = load_features(DATA_DIR)

required_cols = [
    "id_student", "is_late", "submission_delay",
    "weight", "studied_credits", "num_of_prev_attempts",
    "prev_score", "prev_delay",
    "assessment_type", "gender", "age_band",
]

missing = [c for c in required_cols if c not in df_use.columns]
if missing:
    raise ValueError(f"Required columns missing in df_use: {missing}")

print("[DATA] df_use shape:", df_use.shape)


# =========================
# 4. 기본 feature 설정
# =========================

# 숫자형 5개
BASE_NUM_COLS = [
    "weight",
    "studied_credits",
    "num_of_prev_attempts",
    "prev_score",
    "prev_delay",
]

# 범주형 3개
BASE_CAT_COLS = [
    "assessment_type",
    "gender",
    "age_band",
]

ALL_FEATURES = BASE_NUM_COLS + BASE_CAT_COLS

# 타깃과 그룹
y_cls = df_use["is_late"].astype(int)
groups = df_use["id_student"]


# =========================
# 5. 모델 정의 (GPU 설정 포함) — XGBoost만 사용
# =========================

def build_model_dict(use_gpu: bool):
    """
    XGBoost만 사용 (GPU / CPU 모두 지원).
    """
    if not HAS_XGB:
        raise RuntimeError(
            "xgboost가 설치되어 있지 않습니다. "
            "pip install xgboost==1.7.6 같은 방식으로 설치해 주세요."
        )

    models = {}

    if use_gpu:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            eval_metric="logloss",
            use_label_encoder=False,
        )
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
        )

    return models


USE_GPU = (args.gpu_id is not None and args.gpu_id >= 1)
model_dict = build_model_dict(USE_GPU)


# =========================
# 6. Ablation 시나리오 생성 (여러 feature 조합)
# =========================

def make_scenarios(base_num_cols, base_cat_cols, max_drop: int):
    """
    ablation 시나리오 생성:
    - baseline: 아무 것도 제거 안 함
    - 1개 제거, 2개 제거, ..., max_drop개 제거 조합까지 모두 생성
      (단, 전체 feature 개수보다 큰 max_drop은 잘라서 사용)
    """
    features = list(base_num_cols) + list(base_cat_cols)
    total_feat = len(features)
    max_k = min(max_drop, total_feat)

    scenarios = []

    # baseline
    scenarios.append({
        "name": "baseline",
        "drop_features": []
    })

    # k개 제거 (1..max_k)
    for k in range(1, max_k + 1):
        for comb in combinations(features, k):
            drop_list = list(comb)
            # 시나리오 이름: "-feat1-feat2-..."
            name = "-" + "-".join(drop_list)
            scenarios.append({
                "name": name,
                "drop_features": drop_list,
            })

    print(f"[SCENARIO] 총 {len(scenarios)}개 시나리오 생성 "
          f"(baseline + 1~{max_k}개 제거 조합)")
    return scenarios


# =========================
# 7. Fold별 성능표 & 그래프 저장 유틸
# =========================

def save_fold_results_and_plot(model_name, scenario_name, auc_list, f1_list, acc_list, out_dir):
    """
    각 모델 × 시나리오에 대해
    - fold별 AUC/F1/ACC 표를 CSV로 저장
    - 같은 내용을 PNG로 표 + AUC/ACC 라인 그래프로 저장
    """
    model_dir = os.path.join(out_dir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    folds = list(range(1, len(auc_list) + 1))
    df_fold = pd.DataFrame({
        "fold": folds,
        "auc": auc_list,
        "f1": f1_list,
        "acc": acc_list,
    })

    # CSV 저장
    csv_path = os.path.join(model_dir, f"{scenario_name}_fold_metrics.csv")
    df_fold.to_csv(csv_path, index=False)

    # 표 + 라인 그래프 이미지 저장
    png_path = os.path.join(model_dir, f"{scenario_name}_fold_metrics.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (1) 표(table)
    axes[0].axis("off")
    table = axes[0].table(
        cellText=np.round(df_fold.values, 4),
        colLabels=df_fold.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.3)
    axes[0].set_title("Fold Metrics Table")

    # (2) 라인 플롯 (AUC & ACC)
    axes[1].plot(folds, auc_list, marker="o", label="AUC")
    axes[1].plot(folds, acc_list, marker="s", label="ACC")
    axes[1].set_xlabel("Fold (epoch-like)")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(folds)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title("AUC & ACC per Fold")

    fig.suptitle(f"{model_name} - {scenario_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# =========================
# 8. Ablation 실행
# =========================

def run_ablation_experiments(
    df_all: pd.DataFrame,
    base_num_cols,
    base_cat_cols,
    y,
    groups,
    out_csv_path: str,
    out_dir: str,
    max_drop: int,
):
    scenarios = make_scenarios(base_num_cols, base_cat_cols, max_drop=max_drop)
    gkf = GroupKFold(n_splits=5)

    records = []

    for sc in scenarios:
        drop_set = set(sc["drop_features"])
        cur_num = [c for c in base_num_cols if c not in drop_set]
        cur_cat = [c for c in base_cat_cols if c not in drop_set]

        if not cur_num and not cur_cat:
            print(f"[WARN] Scenario {sc['name']} skipped (no features left).")
            continue

        print("\n======================================")
        print(f"Scenario: {sc['name']}")
        print("  사용 num cols :", cur_num)
        print("  사용 cat cols :", cur_cat)

        X = df_all[cur_num + cur_cat].copy()

        # 숫자형은 numeric으로 캐스팅 후 NaN→0
        for c in cur_num:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        # 범주형은 string + UNK 처리
        for c in cur_cat:
            X[c] = X[c].astype("string").fillna("UNK").replace("nan", "UNK")

        preprocess = ColumnTransformer(
            transformers=[
                ("num", "passthrough", cur_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cur_cat),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        for model_name, model in model_dict.items():
            print(f"\n  ▶ Model: {model_name}")
            pipe = Pipeline([
                ("prep", preprocess),
                ("model", model)
            ])

            auc_list, f1_list, acc_list = [], [], []

            for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
                pipe.fit(X.iloc[tr], y.iloc[tr])

                proba = pipe.predict_proba(X.iloc[va])[:, 1]
                pred = (proba >= 0.5).astype(int)

                auc = roc_auc_score(y.iloc[va], proba)
                f1 = f1_score(y.iloc[va], pred)
                acc = accuracy_score(y.iloc[va], pred)

                auc_list.append(auc)
                f1_list.append(f1)
                acc_list.append(acc)

                print(f"    Fold {fold}: AUC={auc:.4f}  F1={f1:.4f}  ACC={acc:.4f}")

            # fold별 성능 저장 + 그래프
            save_fold_results_and_plot(
                model_name=model_name,
                scenario_name=sc["name"],
                auc_list=auc_list,
                f1_list=f1_list,
                acc_list=acc_list,
                out_dir=out_dir,
            )

            rec = {
                "scenario": sc["name"],
                "dropped_features": ",".join(sc["drop_features"]),
                "model": model_name,
                "mean_auc": float(np.mean(auc_list)),
                "mean_f1": float(np.mean(f1_list)),
                "mean_acc": float(np.mean(acc_list)),
            }
            records.append(rec)
            print(f"  ✅ {model_name} @ {sc['name']}  ->  "
                  f"AUC={rec['mean_auc']:.4f}, F1={rec['mean_f1']:.4f}, ACC={rec['mean_acc']:.4f}")

    result_df = pd.DataFrame(records)
    result_df.to_csv(out_csv_path, index=False)
    print("\n[RESULT] Ablation results saved to:", out_csv_path)
    return result_df


# =========================
# 9. main
# =========================

def main():
    out_csv = os.path.join(OUT_DIR, "oulad_ablation_results.csv")
    result_df = run_ablation_experiments(
        df_all=df_use,
        base_num_cols=BASE_NUM_COLS,
        base_cat_cols=BASE_CAT_COLS,
        y=y_cls,
        groups=groups,
        out_csv_path=out_csv,
        out_dir=OUT_DIR,
        max_drop=MAX_DROP,
    )

    print("\n[HEAD of results]")
    print(result_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
