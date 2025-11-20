
## 1. 미루기 방지 프로젝트 (Procrastination Prevention Project)

1. 스냅샷 단위(tabular) 데이터셋 구성  
2. Data leakage 없는 split & sanity check  
3. Tree 기반 모델 + 간단 MLP baseline 성능 확인  

---

## 2. 데이터 파이프라인 개요

### 2.1 Snapshot 단위 데이터셋

`src/data/snapshot_builder.py` 에서 OULAD 원본 CSV들을 사용하여,
각 row가 다음을 의미하는 **스냅샷 단위 레코드**를 만듭니다.

> (student_id, module_presentation, task_id, snapshot_day_rel)
> → 해당 시점에서 아직 과제를 안 냈다면 **48시간 내 미루기 위험 여부**를 label로 둠

주요 처리:

* OULAD 원본 테이블 로드:

  * `studentInfo.csv`
  * `assessments.csv`
  * `studentAssessment.csv`
  * `studentVle.csv`
* 과제(assessment)의 due day, weight 등 정제
* `snapshot_day` 를 일정 간격(every k days, 논의된 설정 기준)으로 생성
* 각 스냅샷에서:

  * 아직 과제를 제출하지 않은 학생만 남김 (`submitted_by_snapshot == 0`)
  * **미래 정보(submission_time 등)는 label 정의에만 사용**하고 feature에는 넣지 않음

최종 결과:
`data/processed/oulad_snapshots.parquet`

---

## 3. Feature & Label 정의

### 3.1 Label

`LABEL_COL = "label_warn_H48"`

* 의미:
  스냅샷 시점 기준으로 **48시간 이내에 “위험” 상태에 있는지**를 나타내는 이진 label

  * `1`: 48시간 안에 과제를 제출하지 못할 위험 (정의는 snapshot_builder 내부 로직 기반)
  * `0`: 상대적으로 안전

추가로 분석에만 쓰이는 label 예:

* `label_late_final`: 최종적으로 늦게 제출 or 미제출 여부

---

### 3.2 Numeric Features

현재 cross-domain 전이를 염두에 둔 **도메인 일반적인 feature 셋**:

```python
NUMERIC_FEATURES = [
    "time_to_deadline_days",    # 스냅샷 시점에서 마감일까지 남은 일수 (0~7 등)
    "time_to_deadline_hours",   # 위의 시간 단위 버전
    "past_num_tasks",           # 지금까지 경험한 과제 수
    "past_late_ratio",          # 지금까지 늦게 제출한 과제 비율
    "past_avg_buffer_days",     # 과거 과제들을 얼마나 여유 있게 냈는지 평균 (deadline - submit)
    "vle_clicks_total",         # 지금까지 해당 과목 VLE 클릭 누적
    "vle_clicks_last_3d",       # 최근 3일 VLE 활동량
    "vle_clicks_last_7d",       # 최근 7일 VLE 활동량
    "num_tasks_due_next_3d",    # 앞으로 3일 내 마감될 과제 수 (workload)
    "num_tasks_due_next_7d",    # 앞으로 7일 내 마감될 과제 수
    "studied_credits",          # 해당 학생이 수강/이수 중인 전체 credit 수
]
```

모든 numeric feature는 `NaN → 0.0` 처리,
없는 컬럼은 아예 0으로 새로 생성하여 **SKKU 데이터에서도 같은 스키마 유지**를 할 수 있도록 해야할 듯듯

### 3.3 Categorical Features (현재 OULAD 스냅샷엔 없음)

```python
CATEGORICAL_FEATURES = [
    "task_type",  # 과제 유형 (assignment / quiz / exam 등)이 있다면 사용 예정
]
```

* 현재 OULAD 스냅샷 데이터에는 `task_type` 컬럼이 없어서
  → OneHotEncoder는 빈 카테고리로 처리되고, 사실상 numeric-only 입력으로 학습 중.

---

## 4. 학습/평가 파이프라인

### 4.1 데이터 로드 & 필터링

`src/training/train_oulad.py` 에서:

1. **스냅샷 로드**

```python
df = load_snapshot_df("data/processed/oulad_snapshots.parquet")
```

2. **기본 필터**

```python
df = basic_filter(df)
# - submitted_by_snapshot == 0 (아직 과제 안 낸 상태만)
# - label_warn_H48 notna()
```

3. **Sanity check**

* `submitted_by_snapshot == 1` 인 경우에도,

  * `submission_day <= snapshot_day` 조건 위반 케이스가 없는지 확인
* `label_warn_H48` 비율, `time_to_deadline_days` 범위 출력

4. **Train/Val/Test split (student 단위)**

```python
df_train, df_val, df_test = split_by_student(df)
```

* 같은 학생이 train, test 양쪽에 섞이지 않도록
  **student_id 기준으로 split** (data leakage 방지)

5. **Feature matrix 생성**

```python
X_train, y_train, X_val, y_val, X_test, y_test, enc = prepare_features(df_train, df_val, df_test)
```

* numeric: 그대로 사용 (결측 0 처리)
* categorical: OneHotEncoder (현재는 사용할 카테고리 없음)
* 최종 `X = [numeric || onehot]`

---

### 4.2 Baseline 모델들

#### 4.2.1 RandomForestClassifier

```python
train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
```

* `n_estimators=300`, `class_weight="balanced"` 등 설정
* 결과(Full feature):

  * **Test ROC-AUC ≈ 0.9956**
  * **Test PR-AUC ≈ 0.9932**
  * Accuracy ≈ 0.968

#### 4.2.2 LightGBM / XGBoost

* 옵션: `--no_xgb`, `--no_lgbm` 으로 끌 수 있음
* 하이퍼파라미터는 보수적인 기본값 + `class_weight` 또는 `scale_pos_weight` 사용
* 성능은 RF와 비슷한 upper bound 수준 (대략 AUC 0.995±)

#### 4.2.3 Simple MLP (PyTorch)

```python
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        ...
```

* BCEWithLogitsLoss + `pos_weight` 사용 (class imbalance 보정)
* Epoch 20 기준 결과:

  * **Train ROC-AUC ≈ 0.9948, PR-AUC ≈ 0.9919**
  * **Val ROC-AUC ≈ 0.9947, PR-AUC ≈ 0.9916**
  * **Test ROC-AUC ≈ 0.9947, PR-AUC ≈ 0.9916**

→ Tree 기반 모델과 거의 동급 성능을 달성하는 **Deep tabular baseline** 구축 완료.

---

## 5. Sanity Check 실험 (Predictive Sanity)

`--run_sanity_predictive` 옵션으로 실행 시 RF 기반으로 다음 실험을 자동 수행

### 5.1 라벨 섞기 (Shuffled Labels)

* Train 라벨을 랜덤 셔플한 후 RF 학습
* Val/Test 결과:

  * ROC-AUC ≈ 0.52, PR-AUC ≈ 0.40

→ 파이프라인에 구조적 버그가 있거나 leakage로 인해 1.0에 가깝게 나오는 것은 아님을 확인.

---

### 5.2 `time_to_deadline_days` **하나만 사용**

* Feature를 `["time_to_deadline_days"]` 하나만 사용한 RF

* Test 성능:

  * **ROC-AUC ≈ 0.9864**
  * **PR-AUC ≈ 0.9627**
  * Accuracy ≈ 0.967

→ **deadline 정보 하나만으로도 상당 부분 label을 맞출 수 있음**
→ 현재 정의된 `label_warn_H48`가 deadline 구조에 강하게 묶여 있다는 것을 의미.

---

### 5.3 deadline feature 제거 (`no time_to_deadline_*`)

* `time_to_deadline_days`, `time_to_deadline_hours`를 빼고
  나머지 feature만 사용한 RF

* Test 성능:

  * **ROC-AUC ≈ 0.8861**
  * **PR-AUC ≈ 0.8602**

→ deadline 없이도 **0.88 수준**의 AUC가 나오므로,
과거 late 비율, VLE 활동량, 앞으로 남은 workload 등으로
**학생의 “미루기 성향”을 꽤 잘 잡아낼 수 있음을 시사**.

---

## 6. 실행 방법

### 6.1 OULAD 스냅샷 생성

실제 인자값은 `snapshot_builder.py`의 `main()` 참고

```bash
python -m src.data.snapshot_builder \
  --raw_dir data/raw/oulad \
  --out_path data/processed/oulad_snapshots.parquet
```

* `raw_dir` 아래에 OULAD CSV들이 있어야 함.
* 실행 후 `data/processed/oulad_snapshots.parquet` 생성.

### 6.2 Baseline 학습 (로컬/Colab 공통)

필수 라이브러리 (전형적인 예):

```bash
pip install pandas numpy scikit-learn matplotlib torch xgboost lightgbm
```

기본 실행:

```bash
python -m src.training.train_oulad \
  --snapshot_path data/processed/oulad_snapshots.parquet \
  --plots_dir outputs/plots_oulad \
  --run_sanity_predictive
```

옵션 예시:

* MLP만 (RF/XGB/LGBM 끄고 싶을 때):

  ```bash
  python -m src.training.train_oulad \
    --snapshot_path data/processed/oulad_snapshots.parquet \
    --no_plots \
    --no_rf \
    --no_xgb \
    --no_lgbm
  ```

* 부스팅만 보고 싶다면 `--no_mlp` 추가 등으로 조합 가능.

---

## 7. 앞으로 할 일 / 확장 방향?

1. **난이도 고정 실험**
   * 예: `time_to_deadline_days == 3` 또는 `== 5`인 subset만 뽑아서
     RF/MLP 성능 측정 → “같은 D-x 시점에서 누가 더 위험한가?”를 평가.

2. **Student / Task Representation 학습**
   * Student encoder: 과거 스냅샷 sequence → embedding `z_student`
   * Task encoder: weight, course, 과제 설명(BERT) → embedding `z_task`
   * Snapshot predictor: `[z_student || z_task || x_state]`로 `warn_H48` 예측.

3. **SKKU 데이터로 Transfer**

   * OULAD에서 encoder pretrain
   * 성대 데이터로 fine-tuning
   * “OULAD로 pretrain vs scratch” 비교, OOD generalization 분석.

```
::contentReference[oaicite:0]{index=0}
```
