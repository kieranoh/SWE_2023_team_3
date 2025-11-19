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