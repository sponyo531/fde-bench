"""
核岛安装工单排产调度评估器

EXTRACTOR_SPEC:
  plan_file: 任务级计划.csv
  required_columns: [施工对象编码, 图纸编码, 房间, 施工对象, 开始时间, 结束时间, 分配工人数, "工期(小时)", 所需工人数, 房间容量, start_hour, end_hour, actual_work_hours]
  extra_files:
    - name: 每小时人力使用.csv
      required_columns: [小时, 忙碌人数, 空闲人数, 总人数]
    - name: 图纸级计划.csv
      required_columns: [图纸编码, 房间, 开始时间, 结束时间, "总工期(小时)", 平均人力]
  notes: >
    Agent 需要产出三张 CSV:
      1) 任务级计划.csv — 每行一个施工任务，字段含 start_hour/end_hour（
         以项目起始日 2024-01-02 的第 0 小时为起点、每天 8 工时的整数小时索引）、
         actual_work_hours(= 安装总点数 × 工效系数)、开始时间/结束时间(YYYY-MM-DD HH:MM)。
         提交必须与 data/combined_data.csv 严格同序、行数一致——评估器按行索引取真值
         校验分档工人数与工期公式；乱序会被判违反。时间区间语义为半开 [start_hour, end_hour)。
      2) 每小时人力使用.csv — 从 hour 0 到 max end_hour, 每小时一行；总人数 = 忙碌 + 空闲, 且非递减。
      3) 图纸级计划.csv — 按图纸编码聚合。
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ── 常量 ──────────────────────────────────────────────────────────────────────

DURATION_TOL = 1e-4  # 工期校验容差

TASK_ORDER = {
    "G01设备": 0, "G02设备": 1, "G02风管": 2, "G03设备": 3,
    "G04大管支架": 4, "G04大管": 5, "G04小管支架": 6, "G04小管": 7,
    "G05主托盘支架": 8, "G05主托盘": 9, "G05设备": 10, "G05箱盒": 11,
    "G05次托盘支架": 12, "G05次托盘": 13, "G05电缆敷设": 14,
    "G05电缆端接": 15, "G06仪表架": 16, "G06仪表": 17,
}

EFFICIENCY_MAP = {
    "G01": 4.06, "G02": 1.38, "G03": 1.76,
    "G04": 1.35, "G05": 1.04, "G06": 1.56,
}


# ── 人力配置规则 ──────────────────────────────────────────────────────────────

def expected_workers(task_type: str, points: float, weight: float) -> int:
    """按 staffing_rules.txt 分档规则计算应分配工人数。"""
    if task_type == "G03设备":
        return 3 if weight < 1 else (6 if weight < 3 else 9)
    elif task_type in ("G04大管支架", "G04大管", "G04小管支架", "G04小管", "G02风管"):
        return 3 if points < 50 else 6
    elif task_type == "G02设备":
        return 3 if weight < 1 else (6 if weight < 3 else 9)
    elif task_type in ("G05次托盘", "G05次托盘支架", "G05主托盘", "G05主托盘支架",
                       "G06仪表架", "G06仪表"):
        return 3
    elif task_type == "G05电缆端接":
        return 2 if points < 10 else 3
    elif task_type == "G05电缆敷设":
        return 3 if points < 50 else (6 if points < 100 else 9)
    elif task_type in ("G05设备", "G05箱盒"):
        if points < 10: return 2
        elif points < 35: return 4
        elif points < 50: return 6
        else: return 8
    elif task_type == "G01设备":
        return 3 if points < 50 else (6 if points < 350 else 8)
    else:
        return 3  # fallback


def get_efficiency_coef(task_type: str) -> float:
    """从施工对象类型获取工效系数。"""
    for prefix, coef in EFFICIENCY_MAP.items():
        if task_type.startswith(prefix):
            return coef
    return 1.0


# ── CSV 读取 ──────────────────────────────────────────────────────────────────

def read_csv_safe(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法读取: {path}")


# ── 性能指标计算（从任务级计划重建 hourly 数据，不信任 agent 提交的每小时人力表）───

def rebuild_hourly_from_tasks(task_schedule):
    """从任务级计划重新生成每小时人力数据（半开区间 [start, end)）。"""
    max_hour = int(task_schedule["end_hour"].max())
    busy_arr = np.zeros(max_hour + 1, dtype=int)

    for _, row in task_schedule.iterrows():
        s = int(row["start_hour"])
        e = int(row["end_hour"])
        w = int(row["分配工人数"])
        busy_arr[s:e] += w

    # 总人数只增不减（累计最大值）
    total_arr = np.maximum.accumulate(busy_arr)

    return busy_arr, total_arr


def calculate_metrics(task_schedule, original_data=None):
    busy_arr, total_arr = rebuild_hourly_from_tasks(task_schedule)

    # 总工期(天)按相对小时轴 end_hour 计算。移交时间约束已换算到同一 start_hour
    # 轴上校验（见 validate() 第6项），因此 start_hour 无法被压缩到移交下界以下，
    # makespan 会如实反映移交造成的工期下界，不存在日历/小时轴脱节的刷分空间。
    makespan_hours = int(task_schedule["end_hour"].max())
    makespan_days = int(np.ceil(makespan_hours / 8))
    peak_workers = int(busy_arr.max())

    # total_worker_hours 从原始数据独立重算 (points × coef 的总和),
    # 不信 agent 自报的 actual_work_hours 列（防 gaming 该列拉高理论最小峰值刷分）。
    if original_data is not None and "安装总点数" in original_data.columns and "施工对象" in original_data.columns:
        work_amount = original_data["安装总点数"].fillna(0).astype(float) \
            * original_data["施工对象"].astype(str).apply(get_efficiency_coef)
        total_worker_hours = float(work_amount.sum())
    else:
        total_worker_hours = float(task_schedule["actual_work_hours"].sum())
    avg_workers = round(total_worker_hours / makespan_hours, 1) if makespan_hours > 0 else 0.0

    # 资源利用率：从重建的数据计算
    utilization_vals = [b / t for b, t in zip(busy_arr, total_arr) if t > 0]
    utilization = float(np.mean(utilization_vals)) if utilization_vals else 0.0

    # 日峰值波动系数
    num_days = int(np.ceil(len(busy_arr) / 8))
    daily_busy = []
    for day in range(num_days):
        s = day * 8
        e = min(s + 8, len(busy_arr))
        daily_busy.append(int(busy_arr[s:e].max()))

    if len(daily_busy) > 1:
        arr = np.array(daily_busy)
        fluctuation = round(float(np.std(arr) / np.mean(arr)), 3) if np.mean(arr) > 0 else 0.0
    else:
        fluctuation = 0.0

    return {
        "总工期(天)": makespan_days,
        "总工期(小时)": makespan_hours,
        "人力峰值(人)": peak_workers,
        "总人力成本(人·小时)": int(total_worker_hours),
        "平均人力(人力/时)": avg_workers,
        "资源利用率(%)": utilization,
        "人力波动系数": fluctuation,
        "总工人数(峰值)": int(total_arr.max()),
    }


# ── 约束校验 ──────────────────────────────────────────────────────────────────

def validate(hourly_stats, task_schedule, drawing_schedule, original_data):
    errors = []

    # 1. 总人数只增不减
    if "总人数" in hourly_stats.columns:
        if (hourly_stats["总人数"].diff().dropna() < 0).any():
            errors.append("总人数减少（应只增不减）")

    # 2. 忙碌 ≤ 总
    if all(c in hourly_stats.columns for c in ["忙碌人数", "总人数"]):
        if (hourly_stats["忙碌人数"] > hourly_stats["总人数"]).any():
            errors.append("忙碌人数超过总人数")

    # 3. 忙碌 + 空闲 = 总
    if all(c in hourly_stats.columns for c in ["忙碌人数", "空闲人数", "总人数"]):
        check = (hourly_stats["忙碌人数"] + hourly_stats["空闲人数"]) == hourly_stats["总人数"]
        if not check.all():
            errors.append("忙碌人数+空闲人数 ≠ 总人数")

    # 4. 施工顺序（同房间内不可倒序）
    if "房间" in task_schedule.columns and "开始时间" in task_schedule.columns:
        for room in task_schedule["房间"].unique():
            room_tasks = task_schedule[task_schedule["房间"] == room].sort_values("开始时间")
            prev_order = -1
            for _, task in room_tasks.iterrows():
                tt = task.get("施工对象", "")
                if tt in TASK_ORDER:
                    curr = TASK_ORDER[tt]
                    if curr < prev_order:
                        errors.append(f"房间{room}: 施工顺序违反 ({tt})")
                        break
                    prev_order = curr

    # 5. 日期格式
    for col in ["开始时间", "结束时间"]:
        if col in task_schedule.columns:
            parsed = pd.to_datetime(task_schedule[col], format="%Y-%m-%d %H:%M", errors="coerce")
            if parsed.isna().any():
                errors.append(f"任务级计划{col}日期格式错误")

    # 6. 移交时间约束（在 start_hour 相对小时轴上校验，与工期同基准）
    # 项目起始日 = 最早移交日 = 2024-01-02 对应 start_hour=0；每天 8 工时。
    # 房间移交日换算为 start_hour 下界 = (移交日 - 项目起始日).days * 8。
    # 任务的 start_hour 必须 >= 该房间的移交小时下界。这样移交约束与 makespan
    # 共用 start_hour 轴，无法靠日历列与 start_hour 脱节来虚降工期。
    if "移交时间" in original_data.columns and all(
        c in task_schedule.columns for c in ["房间", "start_hour"]
    ):
        room_handovers = pd.to_datetime(
            original_data.groupby("房间")["移交时间"].min(), errors="coerce"
        )
        project_start = room_handovers.min()  # = 2024-01-02
        WORK_HOURS_PER_DAY = 8
        room_min_hour = {
            room: max(0, (ho - project_start).days) * WORK_HOURS_PER_DAY
            for room, ho in room_handovers.items()
            if pd.notna(ho)
        }
        violations = 0
        for _, task in task_schedule.iterrows():
            room = task["房间"]
            if room in room_min_hour:
                try:
                    sh = int(task["start_hour"])
                except (ValueError, TypeError):
                    continue
                if sh < room_min_hour[room]:
                    violations += 1
        if violations > 0:
            errors.append(f"{violations}个任务在房间移交前开始（start_hour 早于移交小时下界）")

    # 7. 开始 < 结束 且工期 > 0
    if all(c in task_schedule.columns for c in ["开始时间", "结束时间", "工期(小时)"]):
        for _, task in task_schedule.iterrows():
            s = pd.to_datetime(task["开始时间"])
            e = pd.to_datetime(task["结束时间"])
            if s > e:
                errors.append(f"任务{task.get('施工对象编码','')}: 开始晚于结束")
                break
            if task["工期(小时)"] <= 0:
                errors.append(f"任务{task.get('施工对象编码','')}: 工期≤0")
                break

    # 8. 图纸级计划字段
    if not drawing_schedule.empty:
        req = ["图纸编码", "房间", "开始时间", "结束时间", "总工期(小时)", "平均人力"]
        missing = [c for c in req if c not in drawing_schedule.columns]
        if missing:
            errors.append(f"图纸级计划缺少字段: {missing}")

    # 9. 任务级计划字段
    if not task_schedule.empty:
        req = ["施工对象编码", "图纸编码", "房间", "施工对象", "开始时间", "结束时间",
               "分配工人数", "工期(小时)", "所需工人数", "房间容量",
               "start_hour", "end_hour", "actual_work_hours"]
        missing = [c for c in req if c not in task_schedule.columns]
        if missing:
            errors.append(f"任务级计划缺少字段: {missing}")

    # 10. 施工对象编码完整性
    if "施工对象编码" in original_data.columns and "施工对象编码" in task_schedule.columns:
        orig = set(original_data["施工对象编码"].dropna().astype(str).unique())
        sched = set(task_schedule["施工对象编码"].dropna().astype(str).unique())
        miss = orig - sched
        if miss:
            errors.append(f"{len(miss)}个施工对象编码缺失")

    # 11. 同房间不同施工类型严格串行（逐任务区间检查）
    if all(c in task_schedule.columns for c in ["房间", "施工对象", "start_hour", "end_hour"]):
        for room in task_schedule["房间"].unique():
            rt = task_schedule[task_schedule["房间"] == room]
            if rt.empty:
                continue
            # 按施工类型分组，收集每个任务的 [start, end) 区间
            type_intervals = {}
            for _, task in rt.iterrows():
                tt = task["施工对象"]
                s = int(task["start_hour"])
                e = int(task["end_hour"])
                type_intervals.setdefault(tt, []).append((s, e))
            types = list(type_intervals.keys())
            found = False
            for i in range(len(types)):
                if found:
                    break
                for j in range(i + 1, len(types)):
                    # 检查类型 i 的任意任务是否与类型 j 的任意任务时间重叠
                    overlap = False
                    for s1, e1 in type_intervals[types[i]]:
                        for s2, e2 in type_intervals[types[j]]:
                            if s1 < e2 and s2 < e1:
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        errors.append(
                            f"房间{room}: 不同施工类型时间重叠 "
                            f"({types[i]} vs {types[j]})"
                        )
                        found = True
                        break

    # 12. 同房间同时刻总工人数不超房间容量（半开区间 [start, end)）
    if all(c in task_schedule.columns for c in ["房间", "start_hour", "end_hour", "分配工人数", "房间容量"]):
        for room in task_schedule["房间"].unique():
            rt = task_schedule[task_schedule["房间"] == room]
            if rt.empty:
                continue
            capacity = int(rt["房间容量"].iloc[0])
            max_hour = int(rt["end_hour"].max())
            min_hour = int(rt["start_hour"].min())
            timeline = np.zeros(max_hour - min_hour + 1, dtype=int)
            for _, task in rt.iterrows():
                s = int(task["start_hour"]) - min_hour
                e = int(task["end_hour"]) - min_hour
                w = int(task["分配工人数"])
                timeline[s:e] += w
            peak = int(timeline.max())
            if peak > capacity:
                errors.append(f"房间{room}: 同时刻工人数{peak}超过容量{capacity}")

    # 13. 人力配置规则校验（所需工人数必须符合 staffing_rules.txt 分档表）
    # 提交的任务级计划与 original_data 严格同序、行数一致（each row = one task），
    # 且 (施工对象编码, 施工对象) 不唯一（数据有重复行，相同 key 可对应不同
    # 安装总点数/安装重量）。因此按行序对齐取本行真实的 points/weight，
    # 不能用 key join + iloc[0]（那会对所有重复行都取第一行的值，误判 valid 解）。
    # 反 hack: 若 agent 提交行序与原始不一致(orig code != sub code),视为违反,
    # 不能 continue（否则乱序即可跳过分档校验）。
    if all(c in task_schedule.columns for c in ["施工对象", "所需工人数"]):
        staffing_violations = 0
        staffing_examples = []
        for idx, row in task_schedule.iterrows():
            task_type = row["施工对象"]
            reported_needed = int(row["所需工人数"])
            if idx not in original_data.index:
                staffing_violations += 1
                if len(staffing_examples) < 3:
                    staffing_examples.append(
                        f"任务{row.get('施工对象编码','')}: 行索引{idx}越界，"
                        f"提交行数应与 combined_data.csv 一致且严格同序"
                    )
                continue
            orig_row = original_data.loc[idx]
            # 同序校验：本行 key 必须与原始数据对应行一致，否则视为违反
            if (str(orig_row.get("施工对象编码")) != str(row.get("施工对象编码"))
                    or str(orig_row.get("施工对象")) != str(task_type)):
                staffing_violations += 1
                if len(staffing_examples) < 3:
                    staffing_examples.append(
                        f"任务级计划第{idx}行 (编码{row.get('施工对象编码','')}/{task_type}) "
                        f"与 combined_data.csv 第{idx}行 (编码{orig_row.get('施工对象编码','')}/{orig_row.get('施工对象','')}) 不匹配；"
                        f"提交必须与原始数据严格同序"
                    )
                continue
            points = float(orig_row["安装总点数"])
            weight = float(orig_row["安装重量"])

            expected = expected_workers(task_type, points, weight)
            if reported_needed != expected:
                staffing_violations += 1
                if len(staffing_examples) < 3:
                    staffing_examples.append(
                        f"任务{row.get('施工对象编码','')}: {task_type} "
                        f"所需工人数{reported_needed}人，应为{expected}人"
                    )
        if staffing_violations > 0:
            errors.extend(staffing_examples)
            if staffing_violations > 3:
                errors.append(f"...共{staffing_violations}个任务所需工人数不符合分档规则或行序错乱")

    # 14. 工期公式校验（工期 = ceil(安装总点数 × 工效系数 ÷ 分配工人数)）
    # 同 #13：按行序对齐取本行真实 points，orig code != sub code 视为违反(不 continue)。
    if all(c in task_schedule.columns for c in ["施工对象", "分配工人数", "工期(小时)"]):
        duration_violations = 0
        duration_examples = []
        for idx, row in task_schedule.iterrows():
            task_type = row["施工对象"]
            workers = int(row["分配工人数"])
            reported_duration = float(row["工期(小时)"])

            if workers <= 0:
                duration_violations += 1
                if len(duration_examples) < 3:
                    duration_examples.append(
                        f"任务{row.get('施工对象编码','')}: 分配工人数≤0"
                    )
                continue
            if idx not in original_data.index:
                duration_violations += 1
                continue
            orig_row = original_data.loc[idx]
            if (str(orig_row.get("施工对象编码")) != str(row.get("施工对象编码"))
                    or str(orig_row.get("施工对象")) != str(task_type)):
                # 行序不一致，已在 #13 报告；此处不再重复报告，只计数
                duration_violations += 1
                continue
            points = float(orig_row["安装总点数"])
            coef = get_efficiency_coef(task_type)

            expected_duration = math.ceil(points * coef / workers)

            if abs(reported_duration - expected_duration) > DURATION_TOL:
                duration_violations += 1
                if len(duration_examples) < 3:
                    duration_examples.append(
                        f"任务{row.get('施工对象编码','')}: {task_type} "
                        f"报告工期{reported_duration}h，应为{expected_duration}h"
                    )
        if duration_violations > 0:
            errors.extend(duration_examples)
            if duration_violations > 3:
                errors.append(f"...共{duration_violations}个任务工期不符合公式或行序错乱")

    # 15. (已移除: actual_work_hours 公式校验)

    # 16. 房间容量值校验（不得超过 floor(房间面积 / 2.25)，偏高即判无效）
    # 同 #13/#14：按行序对齐取本行房间面积，orig code != sub code 视为违反。
    if all(c in task_schedule.columns for c in ["施工对象编码", "施工对象", "房间容量"]):
        cap_violations = 0
        cap_examples = []
        for idx, row in task_schedule.iterrows():
            if idx not in original_data.index:
                cap_violations += 1
                continue
            orig_row = original_data.loc[idx]
            if (str(orig_row.get("施工对象编码")) != str(row.get("施工对象编码"))
                    or str(orig_row.get("施工对象")) != str(row.get("施工对象"))):
                cap_violations += 1
                continue
            expected_cap = math.floor(float(orig_row["房间面积"]) / 2.25)
            reported_cap = int(row["房间容量"])
            if reported_cap > expected_cap:
                cap_violations += 1
                if len(cap_examples) < 3:
                    cap_examples.append(
                        f"任务{row.get('施工对象编码','')}: 房间容量={reported_cap}，"
                        f"上限为floor({orig_row['房间面积']}/2.25)={expected_cap}"
                    )
        if cap_violations > 0:
            errors.extend(cap_examples)
            if cap_violations > 3:
                errors.append(f"...共{cap_violations}个任务房间容量超过上限或行序错乱")

    # 17. (已移除: 与验证 13 合并)

    # 18. hour 索引内部一致性校验
    if all(c in task_schedule.columns for c in ["start_hour", "end_hour", "工期(小时)"]):
        hour_violations = 0
        hour_examples = []
        for idx, row in task_schedule.iterrows():
            sh = int(row["start_hour"])
            eh = int(row["end_hour"])
            duration = float(row["工期(小时)"])
            # 内部一致性: end_hour - start_hour 应等于工期
            span = eh - sh
            if abs(span - duration) > DURATION_TOL:
                hour_violations += 1
                if len(hour_examples) < 3:
                    hour_examples.append(
                        f"任务{row.get('施工对象编码','')}: "
                        f"end_hour-start_hour={span}，工期={duration}"
                    )
        if hour_violations > 0:
            errors.extend(hour_examples)
            if hour_violations > 3:
                errors.append(f"...共{hour_violations}个任务hour索引与工期不一致")

    # hour 排序与日期排序一致性
    if all(c in task_schedule.columns for c in ["房间", "开始时间", "start_hour"]):
        for room in task_schedule["房间"].unique():
            rt = task_schedule[task_schedule["房间"] == room].copy()
            if len(rt) < 2:
                continue
            # 必须用稳定排序：默认的 quicksort 对并列元素的 tie-break 顺序不确定，
            # 而这里比的是「字符串日期列」和「整数工时列」两次排序的结果是否一致——
            # 并列一多，两边 tie-break 各排各的，恒定报错。实测同一份数据换排序算法
            # 报错房间数从 108 变到 179。更要命的是它把奖励给反了：忠实地把 1 小时
            # 转成 1 个时间点的抽取 18 个全 0，而给同组时间戳伪造 +1min 递增偏移的
            # 22 个反而通过——评估器在系统性奖励造假。
            dt_order = rt.sort_values("开始时间", kind="stable").index.tolist()
            hr_order = rt.sort_values("start_hour", kind="stable").index.tolist()
            if dt_order != hr_order:
                errors.append(f"房间{room}: hour排序与日期排序不一致")

    return errors


# ── 评分 ──────────────────────────────────────────────────────────────────────

def score(metrics):
    makespan = metrics["总工期(天)"]
    peak_workers = metrics["人力峰值(人)"]
    total_cost_hours = metrics["总人力成本(人·小时)"]
    utilization = metrics["资源利用率(%)"]
    fluctuation = metrics["人力波动系数"]

    理论最短天数 = 613  # 数据集固定下界：在 18 类工序串行约束 + 房间移交约束下的最短完工天数

    工期得分 = max(0, 理论最短天数 / max(1, makespan) * 100)

    理论最小峰值 = math.ceil(total_cost_hours / (makespan * 8 * 0.85))
    人力效率比 = 理论最小峰值 / max(1, peak_workers)

    if 人力效率比 >= 0.90:
        人力得分 = 100
    elif 人力效率比 >= 0.80:
        人力得分 = 85 + (人力效率比 - 0.80) * 150
    elif 人力效率比 >= 0.70:
        人力得分 = 65 + (人力效率比 - 0.70) * 200
    elif 人力效率比 >= 0.60:
        人力得分 = 40 + (人力效率比 - 0.60) * 250
    else:
        人力得分 = max(0, 人力效率比 * 65)

    if fluctuation <= 0.2:
        波动得分 = 100
    elif fluctuation <= 0.35:
        波动得分 = 85 + (0.35 - fluctuation) * 100
    elif fluctuation <= 0.55:
        波动得分 = 60 + (0.55 - fluctuation) * 125
    else:
        波动得分 = max(0, 60 - (fluctuation - 0.55) * 80)

    资源利用得分 = utilization * 100

    combined = 工期得分 * 0.50 + 人力得分 * 0.25 + 波动得分 * 0.15 + 资源利用得分 * 0.10

    return round(combined, 2), round(工期得分, 2), round(人力得分, 2), round(资源利用得分, 2), round(波动得分, 2)


# ── 主评估函数 ────────────────────────────────────────────────────────────────

def evaluate(data_dir: Path, baseline_dir: Path, submission_dir: Path) -> dict:
    result = {
        "validity_score": 0.0,
        "quality_score": 0.0,
        "overall_score": 0.0,
        "errors": [],
    }

    try:
        input_path = data_dir / "combined_data.csv"
        original_data = read_csv_safe(str(input_path))

        hourly_stats = read_csv_safe(str(submission_dir / "每小时人力使用.csv"))
        task_schedule = read_csv_safe(str(submission_dir / "任务级计划.csv"))

        drawing_path = submission_dir / "图纸级计划.csv"
        drawing_schedule = read_csv_safe(str(drawing_path)) if drawing_path.exists() else pd.DataFrame()

        if task_schedule.empty:
            result["errors"] = ["任务级计划为空"]
            return result

        required = ["start_hour", "end_hour", "actual_work_hours"]
        missing = [c for c in required if c not in task_schedule.columns]
        if missing:
            result["errors"] = [f"任务级计划缺少列: {missing}"]
            return result

        metrics = calculate_metrics(task_schedule, original_data)
        errors = validate(hourly_stats, task_schedule, drawing_schedule, original_data)

        if errors:
            result["errors"] = errors
            return result

        combined, 工期, 人力, 资源, 波动 = score(metrics)

        # 用 baseline 归一化: quality = 当前综合分 / baseline 综合分 (higher_is_better, 不截断)
        with open(baseline_dir / "reference_metrics.json", "r", encoding="utf-8") as f:
            ref = json.load(f)
        baseline_score = float(ref["reference_value"])

        result["validity_score"] = 1.0
        result["quality_score"] = round(combined / baseline_score, 4) if baseline_score > 0 else 0.0
        result["overall_score"] = result["quality_score"]
        result["details"] = {
            "combined_score": combined,
            "工期得分": 工期,
            "人力得分": 人力,
            "资源利用得分": 资源,
            "波动得分": 波动,
            "reference_value": baseline_score,
            "metrics": metrics,
        }

    except Exception as e:
        result["errors"] = [f"{type(e).__name__}: {e}"]

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核电站施工排产评估器 V2")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        default=Path(__file__).resolve().parent / "baseline",
        type=Path,
    )
    parser.add_argument("--submission-dir", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(args.data_dir, args.baseline_dir, args.submission_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
