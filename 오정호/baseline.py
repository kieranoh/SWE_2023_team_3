#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")



import kagglehub
HAS_KAGGLEHUB = True


# =========================
# 1. Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="OULAD feature generation + baseline/ablation evaluation (multi-model)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["generate_data", "baseline", "ablation"],
        help="실행 모드: generate_data / baseline / ablation"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="OULAD data root directory (processed/oulad_features.parquet 이 여기 아래에 생성됨)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="generate_data 모드에서 사용할 raw OULAD CSV 디렉터리. "
             "지정한 경로에 studentInfo.csv 등이 있으면 Kaggle 다운로드를 건너뜀."
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="XGBoost 에 대해 GPU 사용 (CUDA). scikit-learn 모델은 항상 CPU."
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="--use_gpu 일 때 사용할 CUDA GPU 인덱스 (기본: 0 → CUDA_VISIBLE_DEVICES='0')"
    )

    parser.add_argument(
        "--max-drop",
        type=int,
        default=3,
        help="(ablation 모드에서만 의미 있음) 한 시나리오에서 제거할 최대 feature 개수 (기본: 3)"
    )

    return parser.parse_args()


# =========================
# 2. 공통 import
# =========================

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

# XGBoost만 GPU 지원
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("[INFO] xgboost available.")
except Exception:
    HAS_XGB = False
    print("[WARN] xgboost not available. XGBoost 모델은 제외됩니다.")


# =========================
# 3. 공통 feature 설정
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

REQUIRED_COLS = [
    "id_student", "is_late", "submission_delay",
] + BASE_NUM_COLS + BASE_CAT_COLS


# =========================
# 4. Raw OULAD → feature 테이블 생성 (generate_data)
# =========================

def find_raw_dir(data_path: str):
    """
    raw OULAD CSV 디렉터리 결정:
      - data_path 가 주어졌고, 그 안에 studentInfo.csv 가 있으면 그대로 사용
      - 아니면 kagglehub.dataset_download(...) 사용
    """
    if data_path and os.path.isdir(data_path):
        candidate = os.path.join(data_path, "studentInfo.csv")
        if os.path.exists(candidate):
            print(f"[DATA] Using existing raw OULAD at: {data_path}")
            return data_path
        else:
            print(f"[WARN] {data_path} 는 존재하지만 studentInfo.csv 가 없습니다. Kagglehub 로 다시 받겠습니다.")

    if not HAS_KAGGLEHUB:
        raise RuntimeError(
            "kagglehub 가 설치되어 있지 않습니다. "
            "pip install kagglehub 후 다시 시도하거나, "
            "--data_path 로 raw CSV 디렉터리를 직접 지정해 주세요."
        )

    print("[DATA] Downloading OULAD from Kaggle via kagglehub...")
    raw_dir = kagglehub.dataset_download("anlgrbz/student-demographics-online-education-dataoulad")
    print("Path to dataset files:", raw_dir)
    return raw_dir


def generate_oulad_features(data_dir: str, data_path: str):
    """
    Kaggle OULAD raw CSV → 우리가 쓰는 feature 테이블 생성 후 parquet 저장
    """
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    features_parquet = os.path.join(processed_dir, "oulad_features.parquet")

    # 이미 생성되어 있으면 재생성하지 않음
    if os.path.exists(features_parquet):
        print(f"[SKIP] {features_parquet} 이미 존재합니다. 생성 과정을 건너뜁니다.")
        return

    raw_dir = find_raw_dir(data_path)

    # --- CSV 로딩 ---
    print("[DATA] Loading raw OULAD csv files...")

    assessments_path = os.path.join(raw_dir, "assessments.csv")
    student_assess_path = os.path.join(raw_dir, "studentAssessment.csv")
    student_info_path = os.path.join(raw_dir, "studentInfo.csv")

    if not (os.path.exists(assessments_path) and
            os.path.exists(student_assess_path) and
            os.path.exists(student_info_path)):
        raise FileNotFoundError(
            f"OULAD raw CSV 일부를 찾을 수 없습니다.\n"
            f"  - {assessments_path}\n"
            f"  - {student_assess_path}\n"
            f"  - {student_info_path}"
        )

    assessments = pd.read_csv(assessments_path)
    student_assess = pd.read_csv(student_assess_path)
    student_info = pd.read_csv(student_info_path)

    print("[DATA] assessments shape:", assessments.shape)
    print("[DATA] studentAssessment shape:", student_assess.shape)
    print("[DATA] studentInfo shape:", student_info.shape)

    # --- studentAssessment + assessments merge ---
    print("[MERGE] Merging studentAssessment with assessments...")
    sa = student_assess.merge(
        assessments[
            ["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]
        ],
        on="id_assessment",
        how="left",
    )

    # --- 지각 여부 및 지각 일수 계산 ---
    print("[FEATURE] Computing submission_delay and is_late...")
    sa["date"] = pd.to_numeric(sa["date"], errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")

    sa["submission_delay"] = (sa["date_submitted"] - sa["date"]).clip(lower=0)
    sa["submission_delay"] = sa["submission_delay"].fillna(0).astype(float)
    sa["is_late"] = (sa["submission_delay"] > 0).astype(int)

    # --- studentInfo merge (demographics + 학습 이력) ---
    print("[MERGE] Merging with studentInfo (demographics)...")
    merged = sa.merge(
        student_info[
            [
                "code_module",
                "code_presentation",
                "id_student",
                "gender",
                "age_band",
                "num_of_prev_attempts",
                "studied_credits",
            ]
        ],
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    # --- prev_score, prev_delay 계산 ---
    print("[FEATURE] Computing prev_score and prev_delay...")

    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    merged["submission_delay"] = pd.to_numeric(merged["submission_delay"], errors="coerce").fillna(0.0)

    merged = merged.sort_values(
        ["id_student", "code_module", "code_presentation", "date", "id_assessment"]
    )

    group_cols = ["id_student", "code_module", "code_presentation"]

    merged["cum_score"] = merged.groupby(group_cols)["score"].cumsum()
    merged["cum_delay"] = merged.groupby(group_cols)["submission_delay"].cumsum()
    merged["cum_count"] = merged.groupby(group_cols).cumcount() + 1

    merged["prev_count"] = merged["cum_count"] - 1
    merged["prev_score_sum"] = merged["cum_score"] - merged["score"]
    merged["prev_delay_sum"] = merged["cum_delay"] - merged["submission_delay"]

    merged["prev_score"] = np.where(
        merged["prev_count"] > 0,
        merged["prev_score_sum"] / merged["prev_count"],
        0.0,
    )
    merged["prev_delay"] = np.where(
        merged["prev_count"] > 0,
        merged["prev_delay_sum"] / merged["prev_count"],
        0.0,
    )

    # --- 필요 컬럼만 선택 ---
    print("[FEATURE] Selecting final feature columns...")

    final_cols = [
        "id_student",
        "is_late",
        "submission_delay",
        "weight",
        "studied_credits",
        "num_of_prev_attempts",
        "prev_score",
        "prev_delay",
        "assessment_type",
        "gender",
        "age_band",
    ]

    df_features = merged[final_cols].copy()

    df_features["id_student"] = df_features["id_student"].astype(int)
    df_features["is_late"] = df_features["is_late"].astype(int)

    print("[DATA] Final feature dataframe shape:", df_features.shape)

    # --- parquet 저장 ---
    print(f"[SAVE] Saving features to {features_parquet}")
    df_features.to_parquet(features_parquet, index=False)
    print("[DONE] Feature generation completed.")


# =========================
# 5. Ablation / Baseline 공통 함수들
# =========================

def load_features(data_dir: str) -> pd.DataFrame:
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

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing in df_use: {missing}")

    return df


def build_model_dict(use_gpu: bool):
    """
    6개 모델:
      - LogisticRegression
      - RandomForest
      - GradientBoosting
      - ExtraTrees
      - KNN
      - XGBoost (옵션으로 GPU)
    """
    models = {}

    # 1) Logistic Regression
    models["LogisticRegression"] = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )

    # 2) Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )

    # 3) Gradient Boosting
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )

    # 4) Extra Trees
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )

    # 5) KNN
    models["KNN"] = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        n_jobs=-1,
    )

    # 6) XGBoost
    if HAS_XGB:
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


def make_scenarios(base_num_cols, base_cat_cols, max_drop: int):
    """
    ablation 시나리오 생성:
    - baseline: 아무 것도 제거 안 함
    - max_drop <= 0 이면 baseline 만 생성
    - max_drop >= 1 이면 1~max_drop개 제거 조합 추가
    """
    features = list(base_num_cols) + list(base_cat_cols)
    total_feat = len(features)
    max_k = min(max_drop, total_feat)

    scenarios = []

    scenarios.append({
        "name": "baseline",
        "drop_features": []
    })

    if max_k <= 0:
        print("[SCENARIO] max_drop <= 0 이므로 baseline 시나리오만 사용합니다.")
        print(f"[SCENARIO] 총 {len(scenarios)}개 시나리오 생성")
        return scenarios

    for k in range(1, max_k + 1):
        for comb in combinations(features, k):
            drop_list = list(comb)
            name = "-" + "-".join(drop_list)
            scenarios.append({
                "name": name,
                "drop_features": drop_list,
            })

    print(f"[SCENARIO] 총 {len(scenarios)}개 시나리오 생성 "
          f"(baseline + 1~{max_k}개 제거 조합)")
    return scenarios


def save_fold_results_and_plot(model_name, scenario_name, auc_list, f1_list, acc_list, out_dir):
    model_dir = os.path.join(out_dir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    folds = list(range(1, len(auc_list) + 1))
    df_fold = pd.DataFrame({
        "fold": folds,
        "auc": auc_list,
        "f1": f1_list,
        "acc": acc_list,
    })

    csv_path = os.path.join(model_dir, f"{scenario_name}_fold_metrics.csv")
    df_fold.to_csv(csv_path, index=False)

    png_path = os.path.join(model_dir, f"{scenario_name}_fold_metrics.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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

    axes[1].plot(folds, auc_list, marker="o", label="AUC")
    axes[1].plot(folds, acc_list, marker="s", label="ACC")
    axes[1].set_xlabel("Fold")
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


def run_experiments(
    df_all: pd.DataFrame,
    base_num_cols,
    base_cat_cols,
    y,
    groups,
    out_csv_path: str,
    out_dir: str,
    max_drop: int,
    model_dict=None,
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

        for c in cur_num:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

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
    print("\n[RESULT] Results saved to:", out_csv_path)
    return result_df


# =========================
# 6. main
# =========================

def main():
    args = parse_args()

    DATA_DIR = args.data_dir
    OUT_DIR = os.path.join(DATA_DIR, "processed")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.mode == "generate_data":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[MODE] generate_data - feature parquet 생성만 수행합니다.")
        generate_oulad_features(DATA_DIR, args.data_path)
        return

    # baseline / ablation 공통 GPU 설정
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[GPU] Using CUDA_VISIBLE_DEVICES={args.gpu_id} (XGBoost만 GPU 사용)")
        use_gpu = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[GPU] GPU disabled, all models use CPU only.")
        use_gpu = False

    df_use = load_features(DATA_DIR)
    print("[DATA] df_use shape:", df_use.shape)

    y_cls = df_use["is_late"].astype(int)
    groups = df_use["id_student"]

    model_dict = build_model_dict(use_gpu=use_gpu)

    if args.mode == "baseline":
        print("[MODE] baseline - feature 제거 없이 6개 모델 성능 측정.")
        max_drop = 0  # baseline만
        out_csv = os.path.join(OUT_DIR, "oulad_baseline_results.csv")
    else:
        print("[MODE] ablation - 여러 feature 조합 제거 + 6개 모델.")
        max_drop = max(1, args.max_drop)
        out_csv = os.path.join(OUT_DIR, "oulad_ablation_results.csv")

    result_df = run_experiments(
        df_all=df_use,
        base_num_cols=BASE_NUM_COLS,
        base_cat_cols=BASE_CAT_COLS,
        y=y_cls,
        groups=groups,
        out_csv_path=out_csv,
        out_dir=OUT_DIR,
        max_drop=max_drop,
        model_dict=model_dict,
    )

    print("\n[HEAD of results]")
    print(result_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
