## baseline.py

 - generation + baseline / ablation evaluation (multi-model, XGBoost GPU optional)
   
- 모드:
  1) generate_data
     - Kagglehub 또는 --data_path 로 받은 OULAD raw CSV를 이용해
       우리가 쓰는 feature 테이블을 만들어 parquet 로 저장.
     - 출력: <data-dir>/processed/oulad_features.parquet
    
   2) baseline
     - feature 제거 없이 baseline feature 셋으로
       아래 6개 모델 성능 측정:
         - LogisticRegression
         - RandomForest
         - GradientBoosting
         - ExtraTrees
         - KNN
         - XGBoost (옵션으로 GPU)
     - GroupKFold(id_student) 5-fold
     - 각 모델 × 시나리오(=baseline 1개) 별 fold metrics CSV/PNG,
       전체 요약 CSV 저장.

   3) ablation
     - baseline과 동일한 6개 모델 사용
     - 여러 feature 제거 조합으로 ablation (max_drop까지)
     - GroupKFold(id_student) 5-fold
     - 각 모델 × 시나리오 별 fold metrics CSV/PNG,
       전체 요약 CSV 저장.
       
```

Usage 예시:

  - 1. 데이터 생성
  python oulad_ablation_gpu.py \
      --mode generate_data \
      --data-dir data \
      --data_path /path/to/raw_oulad_or_empty

  - 2. baseline 측정 (6개 모델, CPU)
  python oulad_ablation_gpu.py \
      --mode baseline \
      --data-dir data

  - 3. baseline 측정 (6개 모델, XGBoost만 GPU 0)
  python oulad_ablation_gpu.py \
      --mode baseline \
      --data-dir data \
      --use_gpu --gpu_id 0

  - 4. ablation (여러 feature 제거 조합, 6개 모델, CPU)
  python oulad_ablation_gpu.py \
      --mode ablation \
      --data-dir data \
      --max-drop 3

  - 5. ablation (XGBoost만 GPU 1)
  python oulad_ablation_gpu.py \
      --mode ablation \
      --data-dir data \
      --max-drop 3 \
      --use_gpu --gpu_id 1
```


## result.csv

```
input = ["weight", "studied_credits", "num_of_prev_attempts", "prev_score", "prev_delay","assessment_type", "gender", "age_band",]

output = ["is_late"]
```


|Model|baseline Auc|
|---|---|
|LogisticRegression|0.8423|
|RandomForest|0.8913|
|GradientBoosting|0.9103|
|ExtraTrees	|0.8749|
|KNN|0.8579|

하나의 feature에 대해서만 ablation 실횅 

## xgboost_ablation.csv

```
input = ["weight", "studied_credits", "num_of_prev_attempts", "prev_score", "prev_delay","assessment_type", "gender", "age_band",]

output = ["is_late"]
```

GPU 이용 학습실행


최대 3개 feature에 대해서만 ablation 실횅 

