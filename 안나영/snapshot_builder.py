 # src/data/snapshot_builder.py

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


# -----------------------------------
# 설정값 (필요하면 config로 빼도 됨)
# -----------------------------------

SNAPSHOT_OFFSETS_DAYS: List[int] = [7, 3, 1, 0]   # 마감 7/3/1/0일 전 시점에서 스냅샷 생성
WARN_HOURS: int = 48                               # 48시간 경고 라벨용


# -----------------------------------
# 1. RAW 데이터 로딩
# -----------------------------------

def load_oulad(raw_dir: Path,
               modules: Optional[List[str]] = None,
               presentations: Optional[List[str]] = None
               ) -> Dict[str, pd.DataFrame]:
    """
    OULAD 원본 CSV들을 읽어서 기본적인 타입/필터링만 수행.
    modules / presentations 리스트를 주면 특정 모듈만 필터링 (속도 위해 추천).
    """
    raw_dir = Path(raw_dir)

    assessments = pd.read_csv(raw_dir / "assessments.csv")
    student_assessment = pd.read_csv(raw_dir / "studentAssessment.csv")
    student_vle = pd.read_csv(
        raw_dir / "studentVle.csv",
        engine="python",          # C 엔진 대신 Python 엔진 사용
        on_bad_lines="skip",      # 깨진 줄은 그냥 건너뛰기
        header=0,
        names=[
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ],
    )
    student_reg = pd.read_csv(raw_dir / "studentRegistration.csv")
    student_info = pd.read_csv(raw_dir / "studentInfo.csv")

    # 필요한 모듈/프레젠테이션만 필터링 (없으면 전체 사용)
    if modules is not None:
        assessments = assessments[assessments["code_module"].isin(modules)]
        student_vle = student_vle[student_vle["code_module"].isin(modules)]
        student_reg = student_reg[student_reg["code_module"].isin(modules)]
        student_info = student_info[student_info["code_module"].isin(modules)]

    if presentations is not None:
        assessments = assessments[assessments["code_presentation"].isin(presentations)]
        student_vle = student_vle[student_vle["code_presentation"].isin(presentations)]
        student_reg = student_reg[student_reg["code_presentation"].isin(presentations)]
        student_info = student_info[student_info["code_presentation"].isin(presentations)]

    return {
        "assessments": assessments,
        "student_assessment": student_assessment,
        "student_vle": student_vle,
        "student_reg": student_reg,
        "student_info": student_info,
    }


# -----------------------------------
# 2. 기본 테이블 전처리
# -----------------------------------

def prepare_assessments(assessments: pd.DataFrame) -> pd.DataFrame:
    """
    assessments.csv를 기반으로 과제/코스 정보를 정리.
    date = 모듈 시작일로부터의 마감일(day) → due_day로 rename
    """
    df = assessments.copy()

    # course_id: module + presentation을 하나의 key로 묶음
    df["course_id"] = df["code_module"] + "_" + df["code_presentation"]

    df = df.rename(columns={
        "id_assessment": "task_id",
        "assessment_type": "task_type",
        "date": "due_day",
    })

    before = len(df)
    df = df[np.isfinite(df["due_day"])].copy()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[prepare_assessments] dropped {dropped} assessments with missing due_day")

    # weight가 NaN이면 0으로 처리 (비중 없는 과제라고 보고)
    df["weight"] = df["weight"].fillna(0.0)
    
    # weight를 0~1 사이 비율로 변환
    df["weight_frac"] = df["weight"] / 100.0

    # 마감일을 int로 캐스팅
    df["due_day"] = df["due_day"].astype(int)

    return df[[
        "course_id", "code_module", "code_presentation",
        "task_id", "task_type", "due_day", "weight", "weight_frac"
    ]]


def prepare_registrations(student_reg: pd.DataFrame) -> pd.DataFrame:
    """
    studentRegistration.csv에서 course_id, student_id, 등록기간만 정리.
    """
    df = student_reg.copy()
    df["course_id"] = df["code_module"] + "_" + df["code_presentation"]
    df = df.rename(columns={"id_student": "student_id"})

    # date_registration / date_unregistration는 snapshot 필터링에 쓸 수 있으나,
    # 여기서는 course에 속한 학생 리스트를 만드는 데 집중
    return df[["course_id", "student_id", "date_registration", "date_unregistration"]]


def prepare_student_assessment(student_assessment: pd.DataFrame,
                               assessments_pre: pd.DataFrame) -> pd.DataFrame:
    """
    studentAssessment + assessments를 merge해서
    (학생, 과제, 코스, 마감일, 제출일, 지연 여부 등) 테이블 생성.
    """
    sa = student_assessment.copy().rename(columns={"id_student": "student_id",
                                                   "id_assessment": "task_id",
                                                   "date_submitted": "submission_day"})
    # 과제 메타 정보 붙이기 (course_id, due_day, weight 등)
    sa = sa.merge(
        assessments_pre[["course_id", "task_id", "due_day", "weight_frac"]],
        on="task_id",
        how="left"
    )

    # 제출 안 한 경우 submission_day가 NaN임
    sa["submitted_final"] = sa["submission_day"].notna()

    # 미제출도 위험케이스로 보고 싶으면: late_final = (제출했고 마감 이후) OR (제출 안 함)
    sa["late_final"] = np.where(
        sa["submitted_final"],
        sa["submission_day"] > sa["due_day"],
        True,   # 미제출
    )

    # buffer_days: 마감 기준 제출 시간 여유(+) 또는 지연(-) (미제출은 NaN)
    sa["buffer_days"] = np.where(
        sa["submitted_final"],
        sa["due_day"] - sa["submission_day"],    # 양수면 마감 전에 제출
        np.nan
    )

    return sa[[
        "course_id", "student_id", "task_id",
        "due_day", "submission_day",
        "submitted_final", "late_final", "buffer_days",
        "weight_frac", "score", "is_banked"
    ]]


def prepare_student_vle(student_vle: pd.DataFrame) -> pd.DataFrame:
    """
    studentVle.csv에서 (course, student, day) 단위로 daily_clicks와 누적 클릭 계산.
    """
    sv = student_vle.copy().rename(columns={
        "id_student": "student_id",
        "date": "day"
    })
    sv["course_id"] = sv["code_module"] + "_" + sv["code_presentation"]

    # (course, student, day) 단위 daily_clicks 합산
    sv_daily = (
        sv.groupby(["course_id", "student_id", "day"], as_index=False)
          .agg(daily_clicks=("sum_click", "sum"))
          .sort_values(["course_id", "student_id", "day"])
    )

    # 누적 클릭 (cum_clicks)
    sv_daily["cum_clicks"] = (
        sv_daily
        .groupby(["course_id", "student_id"])["daily_clicks"]
        .cumsum()
    )

    return sv_daily  # columns: course_id, student_id, day, daily_clicks, cum_clicks


def prepare_student_info(student_info: pd.DataFrame) -> pd.DataFrame:
    """
    studentInfo.csv에서 static feature 일부만 사용.
    (gender, age_band, highest_education, studied_credits, disability 정도)
    """
    si = student_info.copy().rename(columns={"id_student": "student_id"})
    si["course_id"] = si["code_module"] + "_" + si["code_presentation"]

    return si[[
        "course_id", "student_id",
        "gender", "age_band", "highest_education",
        "studied_credits", "disability", "final_result"
    ]]


# -----------------------------------
# 3. 스냅샷 생성
# -----------------------------------

def build_assessment_snapshots(assessments_pre: pd.DataFrame,
                               snapshot_offsets: List[int]) -> pd.DataFrame:
    """
    과제 수준에서 (task_id, snapshot_day) 조합 생성.
    snapshot_day = due_day - offset (offset: 마감까지 남은 일 수)
    """
    offsets_df = pd.DataFrame({"offset_days": snapshot_offsets})

    # cross join: 각 과제마다 offset_days 리스트를 붙임
    a = assessments_pre[["course_id", "task_id", "due_day"]].copy()
    a["key"] = 1
    offsets_df["key"] = 1
    cross = a.merge(offsets_df, on="key").drop(columns="key")

    # snapshot_day = due_day - offset
    cross["snapshot_day"] = cross["due_day"] - cross["offset_days"]

    # 모듈 시작일(0일) 이전 스냅샷은 버림
    cross = cross[cross["snapshot_day"] >= 0].copy()

    # time_to_deadline_days = offset_days
    cross["time_to_deadline_days"] = cross["offset_days"]
    cross["time_to_deadline_hours"] = cross["time_to_deadline_days"] * 24

    return cross[[
        "course_id", "task_id", "snapshot_day",
        "time_to_deadline_days", "time_to_deadline_hours",
        "due_day"
    ]]


def attach_students_to_snapshots(snapshots_task: pd.DataFrame,
                                 registrations: pd.DataFrame) -> pd.DataFrame:
    """
    (과제, 스냅샷) 레벨에 course에 등록된 학생들 붙이기.
    → (student_id, task_id, snapshot_day)가 완성됨.
    """
    students = registrations[["course_id", "student_id"]].drop_duplicates()
    snapshots = snapshots_task.merge(students, on="course_id", how="inner")

    # snapshot id는 나중에 편의를 위해
    snapshots["snapshot_id"] = (
        snapshots["student_id"].astype(str) + "_" +
        snapshots["task_id"].astype(str) + "_" +
        snapshots["snapshot_day"].astype(str)
    )
    return snapshots


def add_labels_and_submission_status(snapshots: pd.DataFrame,
                                     sa_pre: pd.DataFrame) -> pd.DataFrame:
    """
    스냅샷에 제출/지연 여부, warn_H48 라벨 붙이기.
    """
    df = snapshots.merge(
        sa_pre[["course_id", "student_id", "task_id",
                "due_day", "submission_day",
                "submitted_final", "late_final"]],
        on=["course_id", "student_id", "task_id"],
        how="left"
    )

    # NaN (해당 과제에 기록이 없는 학생) → '제출 안 함'
    df["submitted_final"] = df["submitted_final"].fillna(False)
    df["late_final"] = df["late_final"].fillna(True)  # 미제출 = late_final=True 취급
    # submission_day NaN은 매우 늦게 제출했다고 가정할 수 있지만, 여기서는 단순히 비교에서 제외

    # snapshot 시점까지 제출했는지
    df["submitted_by_snapshot"] = np.where(
        df["submitted_final"] & df["submission_day"].notna(),
        df["submission_day"] <= df["snapshot_day"],
        False
    )

    # 기본 late 라벨 (최종 결과 기준)
    df["label_late_final"] = df["late_final"].astype(int)

    # H48 경고용 라벨:
    #   - 마감까지 48시간 이내
    #   - 아직 제출 안 함
    #   - 결국 late/미제출
    warn_days = WARN_HOURS / 24.0

    df["label_warn_H48"] = np.where(
        (df["time_to_deadline_days"] <= warn_days) &
        (~df["submitted_by_snapshot"]) &
        (df["late_final"]),
        1,
        0
    )

    return df


# -----------------------------------
# 4. 과거 과제 히스토리 feature (학생별 / 코스별)
# -----------------------------------

def build_past_assessment_history(sa_pre: pd.DataFrame) -> pd.DataFrame:
    """
    (course_id, student_id, due_day) 기준으로 하루치 과제 정보 요약 후
    누적(cumsum)을 구해서 snapshot에서 merge_asof로 붙일 준비를 한다.
    """
    # 하루에 같은 날 마감인 과제 여러 개 있을 수 있으니 day단위로 합산
    daily = (
        sa_pre.groupby(["course_id", "student_id", "due_day"], as_index=False)
              .agg(
                  tasks_on_day=("task_id", "count"),
                  late_on_day=("late_final", lambda x: x.sum()),
                  submitted_on_day=("submitted_final", lambda x: x.sum()),
                  buffer_sum_on_day=("buffer_days", lambda x: np.nansum(x))
              )
              .rename(columns={"due_day": "day"})
              .sort_values(["course_id", "student_id", "day"])
    )
    # day는 merge_asof 키가 될 것이므로 int64로 맞춰준다
    daily["day"] = daily["day"].astype("int64")

    # 누적합 (이 날까지 끝난 과제 기준)
    daily["cum_tasks"] = daily.groupby(["course_id", "student_id"])["tasks_on_day"].cumsum()
    daily["cum_late"] = daily.groupby(["course_id", "student_id"])["late_on_day"].cumsum()
    daily["cum_submitted"] = daily.groupby(["course_id", "student_id"])["submitted_on_day"].cumsum()
    daily["cum_buffer_sum"] = daily.groupby(["course_id", "student_id"])["buffer_sum_on_day"].cumsum()

    return daily


def add_past_history_features(snapshots: pd.DataFrame,
                              history_daily: pd.DataFrame) -> pd.DataFrame:
    """
    각 snapshot에 대해 "이 snapshot 이전에 마감된 과제들"만 사용하여
    과거 late 비율 / 평균 buffer를 계산해서 붙인다.
    """
    df = snapshots.copy()

    # snapshot 이전까지의 히스토리 → history_day = snapshot_day - 1
    df["history_day"] = df["snapshot_day"] - 1

    # merge_asof는 키 타입이 동일해야 하므로 int64로 통일
    df["history_day"] = df["history_day"].astype("int64")
    hist = history_daily.copy()
    hist["day"] = hist["day"].astype("int64")

    # merge_asof: history_day 기준으로 day <= history_day인 마지막 row를 가져옴
    merged = pd.merge_asof(
        df.sort_values("history_day"),
        history_daily.sort_values("day"),
        left_on="history_day",
        right_on="day",
        by=["course_id", "student_id"],
        direction="backward"
    )

    # 과거 과제가 하나도 없는 경우 NaN → 0으로 치환
    for col in ["cum_tasks", "cum_late", "cum_submitted", "cum_buffer_sum"]:
        merged[col] = merged[col].fillna(0.0)

    # past_* feature 계산
    merged["past_num_tasks"] = merged["cum_tasks"]
    merged["past_late_ratio"] = np.where(
        merged["cum_tasks"] > 0,
        merged["cum_late"] / merged["cum_tasks"],
        0.0
    )
    merged["past_avg_buffer_days"] = np.where(
        merged["cum_submitted"] > 0,
        merged["cum_buffer_sum"] / merged["cum_submitted"],
        0.0
    )

    return merged


# -----------------------------------
# 5. VLE 클릭 기반 feature
# -----------------------------------

def add_vle_click_features(snapshots: pd.DataFrame,
                           sv_daily: pd.DataFrame,
                           horizons_days: List[int] = [3, 7]) -> pd.DataFrame:
    """
    학생별 / 코스별 VLE 클릭 누적(cum_clicks)을 사용해서
    - snapshot까지의 총 클릭 수
    - 최근 k일(3,7 등) 클릭 수
    를 계산해서 붙인다.
    """
    df = snapshots.copy()

    # snapshot 시점까지 총 클릭 수
    # merge_asof: snapshot_day 기준으로 day <= snapshot_day 인 마지막 cum_clicks
    base = pd.merge_asof(
        df.sort_values("snapshot_day"),
        sv_daily.sort_values("day"),
        left_on="snapshot_day",
        right_on="day",
        by=["course_id", "student_id"],
        direction="backward"
    )
    base["cum_clicks"] = base["cum_clicks"].fillna(0.0)
    base = base.rename(columns={"cum_clicks": "vle_clicks_total"})

    # horizon마다 최근 k일 클릭 수 계산
    for h in horizons_days:
        col_name = f"vle_clicks_last_{h}d"

        # 기준 시점 = snapshot_day - h
        base[f"snapshot_day_minus_{h}"] = base["snapshot_day"] - h

        prev = pd.merge_asof(
            base.sort_values(f"snapshot_day_minus_{h}"),
            sv_daily.sort_values("day"),
            left_on=f"snapshot_day_minus_{h}",
            right_on="day",
            by=["course_id", "student_id"],
            direction="backward"
        )

        prev["cum_clicks"] = prev["cum_clicks"].fillna(0.0)
        # last_h = cum_now - cum_(snapshot_day-h)
        base[col_name] = base["vle_clicks_total"] - prev["cum_clicks"]

        # 음수 방지 (이론상 없지만 수치적 이유로)
        base[col_name] = base[col_name].clip(lower=0.0)

    return base


# -----------------------------------
# 6. Workload feature (앞으로 마감될 과제 수)
# -----------------------------------

def add_workload_features(snapshots: pd.DataFrame,
                          assessments_pre: pd.DataFrame,
                          horizons_days: List[int] = [3, 7]) -> pd.DataFrame:
    """
    각 snapshot에 대해 같은 course 내에서
    - 앞으로 k일 안에 마감되는 과제 수
    를 계산해서 붙인다.
    과제 수가 많지 않아서 course-level에서 day 중복 제거 후 루프 돌리는 방식으로 구현.
    """
    df = snapshots.copy()

    # course / day 레벨로 unique snapshot_day만
    unique_cd = df[["course_id", "snapshot_day"]].drop_duplicates()

    rows = []
    for course_id, grp in unique_cd.groupby("course_id"):
        due_days = assessments_pre.loc[assessments_pre["course_id"] == course_id, "due_day"].values
        if len(due_days) == 0:
            continue
        for snapshot_day in grp["snapshot_day"]:
            row = {"course_id": course_id, "snapshot_day": snapshot_day}
            for h in horizons_days:
                mask = (due_days > snapshot_day) & (due_days <= snapshot_day + h)
                row[f"num_tasks_due_next_{h}d"] = int(mask.sum())
            rows.append(row)

    workload = pd.DataFrame(rows)
    df = df.merge(workload, on=["course_id", "snapshot_day"], how="left")

    # NaN → 0
    for h in horizons_days:
        col = f"num_tasks_due_next_{h}d"
        df[col] = df[col].fillna(0).astype(int)

    return df


# -----------------------------------
# 7. studentInfo static feature 붙이기
# -----------------------------------

def add_student_static_features(snapshots: pd.DataFrame,
                                student_info_pre: pd.DataFrame) -> pd.DataFrame:
    """
    성별, 나이대, 교육수준, disability 등 static feature를 붙인다.
    """
    df = snapshots.merge(
        student_info_pre,
        on=["course_id", "student_id"],
        how="left"
    )
    return df


# -----------------------------------
# 8. 전체 파이프라인
# -----------------------------------

def build_snapshot_dataset(raw_dir: str | Path,
                           output_path: str | Path,
                           modules: Optional[List[str]] = None,
                           presentations: Optional[List[str]] = None) -> None:
    """
    전체 파이프라인 실행:
    - OULAD 로딩
    - 기본 테이블 전처리
    - snapshot 생성
    - label 및 feature 엔지니어링
    - parquet/CSV로 저장
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    print("[1] Load raw OULAD CSVs...")
    data = load_oulad(raw_dir, modules=modules, presentations=presentations)
    assessments = data["assessments"]
    student_assessment = data["student_assessment"]
    student_vle = data["student_vle"]
    student_reg = data["student_reg"]
    student_info = data["student_info"]

    print("[2] Prepare base tables...")
    assessments_pre = prepare_assessments(assessments)
    reg_pre = prepare_registrations(student_reg)
    sa_pre = prepare_student_assessment(student_assessment, assessments_pre)
    sv_daily = prepare_student_vle(student_vle)
    si_pre = prepare_student_info(student_info)

    print("[3] Build (task, snapshot) table...")
    snapshots_task = build_assessment_snapshots(assessments_pre, SNAPSHOT_OFFSETS_DAYS)

    print("[4] Attach students to snapshots...")
    snapshots = attach_students_to_snapshots(snapshots_task, reg_pre)

    print("[5] Add labels (late, warn_H48)...")
    snapshots = add_labels_and_submission_status(snapshots, sa_pre)

    print("[6] Build past assessment history...")
    history_daily = build_past_assessment_history(sa_pre)
    snapshots = add_past_history_features(snapshots, history_daily)

    print("[7] Add VLE click features...")
    snapshots = add_vle_click_features(snapshots, sv_daily, horizons_days=[3, 7])

    print("[8] Add workload features...")
    snapshots = add_workload_features(snapshots, assessments_pre, horizons_days=[3, 7])

    print("[9] Add student static features...")
    snapshots = add_student_static_features(snapshots, si_pre)

    print(f"[10] Save snapshot dataset to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        snapshots.to_parquet(output_path, index=False)
    else:
        snapshots.to_csv(output_path, index=False)

    print("[Done] Snapshot dataset shape:", snapshots.shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_path", type=str, default="data/processed/oulad_snapshots.parquet")
    parser.add_argument("--modules", type=str, nargs="*", default=None,
                        help="Optional list of code_module to include, e.g. AAA BBB")
    parser.add_argument("--presentations", type=str, nargs="*", default=None,
                        help="Optional list of code_presentation to include, e.g. 2013J 2014B")
    args = parser.parse_args()

    build_snapshot_dataset(
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        modules=args.modules,
        presentations=args.presentations,
    )
