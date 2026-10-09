"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  schema: |
    求解产物为合法 VRP 配送方案, 即一组车辆路线, 每个路线是一个 trip 对象:
    {
      "tripId": <int>,
      "distance": <float>,              # 本趟总里程(米)
      "duration": <int>,                # 本趟总时长(分钟,含装卸)
      "depot": {...},                   # 仓库点(装货)
      "departPoint": {...},             # SITE_002(发车)
      "customers": [
        {
          "locationId": <str>,          # 必须指向 input 的 customer.locationId
          "name": <str>, "address": <str>,
          "arriveTime": <str>, "departTime": <str>,   # 形如 202603050830
          "distance": <float>, "duration": <int>, "chargeTime": <int>,
          "orders": [ {"order_id": <str>, "oilType": <int>, "oilCategory": <str>,
                       "wareId": <int>, "volume": <float>} ]
        }, ...
      ],
      "loadInfo": [
        {"wareId": <int>, "oilType": <int>, "oilCategory": <str>, "volume": <float>, "volumePercent": <float>}
      ]
    }
  notes: >
    多油仓多油品 VRP。产物是 trips 列表(整个 solution.json 就是 trips 数组或 {trips:[...]})。
    评估器按在线口径校验: 油仓-油品匹配 / 需求全覆盖 / 时间窗 / 车型准入 / 仓容量 / 总量一致,
    再按总行驶里程归一化给分(里程越小分数越高)。
    参考的 EXTRACTOR_SPEC 参考线上燃料配送交付物结构: 每条 trip 的 customers 需含 order_id 与 volume
    与 input 订单精确一致, wareId/loadType 须对应车型仓定义。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent


# ===================== 内联自 tests/validator_core.py（评估器自包含，原文件已删除） =====================
"""
Vehicle Routing Problem Evaluation and Validation
VRP 问题的验证和评估模块

验证所有约束条件：
1. 油仓类型与油品匹配
2. 所有客户需求被完全满足
3. 时间窗约束
4. 车辆类型匹配
5. 仓容量约束
"""
import pickle
import uuid
import json
from typing import List, Dict, Tuple
# 无需导入外部初始化脚本；直接校验 JSON 方案。
from datetime import datetime, timedelta
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import re
import ast
import logging
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- 时间窗工具函数（原 TimeWindow 类）--------------------
def parse_time_window(work_time_str: str) -> List[Tuple[str, str]]:
    """解析时间窗字符串，支持多个时间窗
    例如: "202603050800-202603051200|202603051400-202603051800"
    """
    windows = []
    time_ranges = work_time_str.split('|')
    for time_range in time_ranges:
        parts = time_range.split('-')
        if len(parts) == 2:
            windows.append((parts[0], parts[1]))
    return windows


def time_to_minutes(time_str: str, task_start_date: str) -> int:
    """将时间字符串转换为距离任务开始时间的分钟数
    time_str: "202603050800" -> 距离任务开始的分钟数
    """
    try:
        time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
        start_obj = datetime.strptime(task_start_date, "%Y%m%d%H%M")
        delta = time_obj - start_obj
        return int(delta.total_seconds() / 60)
    except:
        return 0


def _oid(order, where=""):
    """提交侧订单号:兼容 order_id / orderId 两种写法(输入数据用 order_id)。
    取不到时返回 None,由调用方记成校验错误,不再抛 KeyError 把评估打崩。"""
    if not isinstance(order, dict):
        return None
    for k in ("order_id", "orderId"):
        if k in order:
            return order[k]
    return None


def _fld(d, *names, default=None):
    """按候选名依次取字段,全都取不到时返回 default(不抛异常)。"""
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d:
            return d[n]
    return default

# -------------------- 验证辅助函数（原 VRPValidator 类的方法）--------------------

def _validate_fuel_type_consistency(input_data: Dict, output_data: List[Dict], errors: List[str], warnings: List[str]):
    """验证1：每个仓的燃料类型与装载油品一致"""
    for trip in output_data:
        for load_item in trip.get('loadInfo', []):
            ware_id = _fld(load_item, 'wareId')
            oil_type = _fld(load_item, 'oilType')
            # 注:oilCategory 只是给人看的中文油品名(92#/95#/柴油),判定一律以
            # oilType 这个整数编码为准 —— 仓的定义 ware['loadType'] 存的也是整数。
            # 曾在这里取过 oil_category 却从不使用,是个死字段:留着会让读代码的人
            # 以为它参与校验,也会让 agent 花力气去对齐一个不影响结果的字段。
            if ware_id is None or oil_type is None:
                errors.append(
                    f"Trip {_fld(trip, 'tripId', default='?')}: loadInfo 缺 wareId 或 oilType"
                )
                continue

            # 查找对应的仓定义
            found = False
            for vehicle in input_data['vehicles']:
                for ware in vehicle['vehcileType']:
                    if ware['wareId'] == ware_id:
                        if ware['loadType'] == oil_type:
                            found = True
                        else:
                            errors.append(
                                f"Trip {trip['tripId']}: 仓{ware_id}的燃料类型{oil_type}与定义{ware['loadType']}不符"
                            )
                        break
                if found:
                    break
            if not found:
                errors.append(f"Trip {_fld(trip, 'tripId', default='?')}: 仓{ware_id}未在车辆定义中找到")


def _validate_customer_demands_satisfied(input_data: Dict, output_data: List[Dict], errors: List[str], warnings: List[str]):
    """验证2：所有客户的需求被完全满足"""
    # 收集所有客户的需求
    customer_demands = {}
    for customer in input_data['customers']:
        customer_id = customer['locationId']
        customer_demands[customer_id] = {
            'orders': customer.get('orders', [])
        }

    # 收集所有配送的货物
    delivered_cargo = {}
    for trip in output_data:
        for customer in trip.get('customers', []):
            customer_id = _fld(customer, 'locationId')
            if customer_id is None:
                errors.append("提交里有一条客户记录没写 locationId")
                continue
            if customer_id not in delivered_cargo:
                delivered_cargo[customer_id] = []

            for order in customer.get('orders', []):
                delivered_cargo[customer_id].append(order)

    # 验证每个客户的需求
    for customer_id, demand in customer_demands.items():
        required_orders = {_oid(o): o for o in demand['orders']}
        delivered_orders = {}
        for o in delivered_cargo.get(customer_id, []):
            oid = _oid(o)
            if oid is None:
                errors.append(
                    f"客户 {customer_id}: 有一条订单没写订单号(字段名应为 order_id)"
                )
                continue
            delivered_orders[oid] = o

        # 检查是否所有订单都被交付
        for order_id, order in required_orders.items():
            if order_id not in delivered_orders:
                errors.append(
                    f"客户 {customer_id}: 订单 {order_id} 未被交付"
                )
            elif _fld(delivered_orders[order_id], 'volume') != order['volume']:
                errors.append(
                    f"客户 {customer_id}: 订单 {order_id} 的体积不匹配 "
                    f"(期望: {order['volume']}, 实际: {_fld(delivered_orders[order_id], 'volume')})"
                )


def _validate_time_windows(input_data: Dict, output_data: List[Dict], errors: List[str], warnings: List[str]):
    """验证3：所有客户的时间窗被遵守"""
    for trip in output_data:
        for customer in trip.get('customers', []):
            customer_id = _fld(customer, 'locationId')
            arrive_time = _fld(customer, 'arriveTime')
            depart_time = _fld(customer, 'departTime')
            if customer_id is None or arrive_time is None or depart_time is None:
                errors.append(
                    f"客户 {customer_id or '?'}: 缺 locationId / arriveTime / departTime 之一"
                )
                continue

            # 获取客户的时间窗
            customer_info = None
            for c in input_data['customers']:
                if c['locationId'] == customer_id:
                    customer_info = c
                    break

            if not customer_info:
                continue

            # 解析时间窗
            work_time = customer_info.get('workTime', '')
            time_windows = parse_time_window(work_time)

            # 检查到达时间是否在任何时间窗内
            arrive_minutes = time_to_minutes(arrive_time, input_data['taskStartDate'])
            service_duration = customer_info.get('serviceDuration', 90)
            service_end_minutes = arrive_minutes + service_duration

            time_window_valid = False
            for start_time_str, end_time_str in time_windows:
                start_minutes = time_to_minutes(start_time_str, input_data['taskStartDate'])
                end_minutes = time_to_minutes(end_time_str, input_data['taskStartDate'])

                # 检查服务结束时间是否在时间窗内
                if start_minutes <= arrive_minutes and service_end_minutes <= end_minutes:
                    time_window_valid = True
                    break

            if not time_window_valid:
                warnings.append(
                    f"客户 {customer_id}: 到达时间 {arrive_time} 可能不在时间窗内"
                )


def _validate_vehicle_type_matching(input_data: Dict, output_data: List[Dict], errors: List[str], warnings: List[str]):
    """验证4：所有客户的车辆类型匹配允许列表"""
    for customer in input_data['customers']:
        customer_id = customer['locationId']
        allowed_vehicles = customer.get('availableVehicles', [])

        if not allowed_vehicles:
            continue

        # 检查该客户是否被某辆允许的车辆类型配送
        found = False
        for trip in output_data:
            for cust in trip.get('customers', []):
                if _fld(cust, 'locationId') == customer_id:
                    vehicle_type = trip.get('vehicleType', 'unknown')
                    if vehicle_type in allowed_vehicles:
                        found = True
                    break
            if found:
                break

        if not found and allowed_vehicles:
            warnings.append(
                f"Customer {customer_id}: not served by allowed vehicle types {allowed_vehicles}"
            )


def _validate_warehouse_capacity(input_data: Dict, output_data: List[Dict], errors: List[str], warnings: List[str]):
    """验证5：按车型真实仓容与订单细类复算每趟装载。"""
    vehicle_defs = {str(v['name']): v for v in input_data.get('vehicles', [])}
    required_orders = {}
    for customer in input_data.get('customers', []):
        cid = str(customer.get('locationId'))
        for order in customer.get('orders', []):
            required_orders[(cid, str(_oid(order)))] = order

    for trip in output_data:
        trip_id = _fld(trip, 'tripId', default='?')
        vehicle_type = str(_fld(trip, 'vehicleType', default='')).strip()
        vehicle = vehicle_defs.get(vehicle_type)
        if vehicle is None:
            errors.append(
                f"Trip {trip_id}: vehicleType 必须是车型定义中的名称"
            )
            continue
        ware_defs = {int(w['wareId']): w for w in vehicle.get('vehcileType', [])}
        loaded_by_ware = {}
        for load_item in trip.get('loadInfo', []):
            volume = _fld(load_item, 'volume')
            warehouse_id = _fld(load_item, 'wareId')
            if volume is None or warehouse_id is None:
                errors.append(
                    f"Trip {trip_id}: loadInfo 缺 volume 或 wareId"
                )
                continue
            try:
                warehouse_id = int(warehouse_id)
                volume = float(volume)
            except (TypeError, ValueError):
                errors.append(f"Trip {trip_id}: 仓号或装载量不是有效数值")
                continue
            if warehouse_id not in ware_defs:
                errors.append(f"Trip {trip_id}: 车型 {vehicle_type} 没有仓 {warehouse_id}")
                continue
            if not math.isfinite(volume) or volume < 0:
                errors.append(
                    f"Trip {trip_id}: 仓 {warehouse_id} 装载量必须是非负有限数"
                )
                continue
            loaded_by_ware[warehouse_id] = loaded_by_ware.get(warehouse_id, 0.0) + volume

        delivered_by_ware = {}
        categories_by_ware = {}
        for customer in trip.get('customers', []):
            cid = str(_fld(customer, 'locationId', default=''))
            for order in customer.get('orders', []):
                oid = _oid(order)
                truth = required_orders.get((cid, str(oid)))
                warehouse_id = _fld(order, 'wareId')
                if warehouse_id is None:
                    errors.append(
                        f"Trip {trip_id}: 客户 {cid} 订单 {oid} 缺 wareId"
                    )
                    continue
                try:
                    warehouse_id = int(warehouse_id)
                except (TypeError, ValueError):
                    errors.append(f"Trip {trip_id}: 订单 {oid} 的 wareId 无效")
                    continue
                if warehouse_id not in ware_defs:
                    errors.append(
                        f"Trip {trip_id}: 订单 {oid} 分到车型 {vehicle_type} 不存在的仓 {warehouse_id}"
                    )
                    continue
                if truth is None:
                    continue  # 未知订单由需求对账分支报错
                expected_type = int(truth['oilType'])
                if int(ware_defs[warehouse_id]['loadType']) != expected_type:
                    errors.append(
                        f"Trip {trip_id}: 订单 {oid} 的油品大类与仓 {warehouse_id} 不匹配"
                    )
                categories_by_ware.setdefault(warehouse_id, set()).add(
                    str(truth.get('fuelCategory', '')).strip()
                )
                try:
                    delivered_by_ware[warehouse_id] = (
                        delivered_by_ware.get(warehouse_id, 0.0) + float(_fld(order, 'volume'))
                    )
                except (TypeError, ValueError):
                    errors.append(f"Trip {trip_id}: 订单 {oid} 的 volume 无效")

        for warehouse_id, categories in categories_by_ware.items():
            categories.discard('')
            if len(categories) > 1:
                errors.append(
                    f"Trip {trip_id}: 仓 {warehouse_id} 混装了多种细类油品 {sorted(categories)}"
                )

        for warehouse_id in set(loaded_by_ware) | set(delivered_by_ware):
            loaded = loaded_by_ware.get(warehouse_id, 0.0)
            delivered = delivered_by_ware.get(warehouse_id, 0.0)
            if abs(loaded - delivered) > 0.01:
                errors.append(
                    f"Trip {trip_id}: 仓 {warehouse_id} 装载 {loaded:.2f}L 与该仓交付 {delivered:.2f}L 不一致"
                )
            capacity = float(ware_defs[warehouse_id]['volume'])
            if loaded > capacity + 0.01:
                errors.append(
                    f"Trip {trip_id}: 仓 {warehouse_id} 装载 {loaded:.2f}L 超过真实仓容 {capacity:.2f}L"
                )


def _validate_total_volume_consistency(input_data: Dict, output_data: List[Dict], errors: List[str], warnings: List[str]):
    """验证6：客户总需求与配送总量一致"""
    total_required = 0
    for customer in input_data['customers']:
        for order in customer.get('orders', []):
            total_required += order['volume']

    total_delivered = 0
    for trip in output_data:
        for load_item in trip.get('loadInfo', []):
            total_delivered += float(_fld(load_item, 'volume', default=0) or 0)

    if abs(total_required - total_delivered) > 0.01:
        errors.append(
            f"Total volume mismatch: required={total_required:.2f}L, delivered={total_delivered:.2f}L"
        )


def validate_all(input_data: Dict, output_data: List[Dict]) -> Tuple[bool, List[str], List[str]]:
    """执行所有验证（原 VRPValidator.validate_all）"""
    errors = []
    warnings = []

    _validate_fuel_type_consistency(input_data, output_data, errors, warnings)
    _validate_customer_demands_satisfied(input_data, output_data, errors, warnings)
    _validate_time_windows(input_data, output_data, errors, warnings)
    _validate_vehicle_type_matching(input_data, output_data, errors, warnings)
    _validate_warehouse_capacity(input_data, output_data, errors, warnings)
    _validate_total_volume_consistency(input_data, output_data, errors, warnings)

    return len(errors) == 0, errors, warnings

# ---------------运行被评估代码--------------------
def run_with_timeout(program_path: str, function_name: str, args: dict, timeout_seconds: int = 2000) -> dict:
    """
    Run the program in a separate process with timeout
    """
    temp_path = f"{function_name}_{uuid.uuid4().hex}.pkl"
    with open(temp_path, "wb") as f:
        pickle.dump(args, f)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        if not os.path.isabs(program_path):
            program_path = os.path.abspath(program_path)
        clean_program_path = program_path.replace('\x0c', '')

        script = f"""
import sys
import os
import pickle
import traceback

sys.path.insert(0, os.path.dirname(r'{clean_program_path}'))

try:
    with open(r'{temp_path}', 'rb') as f:
        args = pickle.load(f)

    spec = __import__('importlib.util').util.spec_from_file_location("program", r'{clean_program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)

    entry = getattr(program, "{function_name}", None)
    if entry is None:
        raise AttributeError("new_born program.py doesn't have {function_name}")

    results = entry(**args)

    with open(r'{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    tb = traceback.format_exc()
    with open(r'{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': f'{{e}}', 'traceback': tb}}, f)
    sys.exit(0)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if "error" in results:
                    err = results.get("error", "")
                    tb = results.get("traceback", "")
                    raise RuntimeError(f"Program execution failed: {err}\nTraceback:\n{tb}")
                return results

            if exit_code != 0:
                raise RuntimeError(
                    f"Process exited with code {exit_code}\n"
                    f"---- STDOUT ----\n{stdout.decode(errors='ignore')}\n"
                    f"---- STDERR ----\n{stderr.decode(errors='ignore')}"
                )
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            try:
                os.unlink(results_path)
            except:
                pass


def normalize_cost_exp(total_cost, scale_factor=1271000):
    """
    使用指数衰减函数进行归一化
    """
    import math
    return math.exp(-total_cost / scale_factor) * 100

def score(results):
    """
    根据结果计算分数
    """
    total_distance = 0
    max_duration = 0
    min_duration = float('inf')
    for trip in results:
        total_distance += trip.get('distance', 0)
        duration = trip.get('duration', 0)
        max_duration = max(max_duration, duration)
        if duration > 0:
            min_duration = min(min_duration, duration)

    if min_duration == float('inf'):
        min_duration = 0

    if min_duration == float('inf'):
        min_duration = 0

    return {
        'combined_score': normalize_cost_exp(total_distance),
        'max_duration': max_duration,
        'min_duration': min_duration,
    }


def evaluate_solution(input_file: str = None, output_file: str = None,
                     input_data: Dict = None, output_data: List[Dict] = None) -> Tuple[bool, List[str], List[str]]:
    """
    评估VRP解的可行性和质量

    Args:
        input_file: 输入文件路径（JSON格式）
        output_file: 输出文件路径（JSON格式）
        input_data: 输入数据（直接传入）
        output_data: 输出数据（直接传入）

    Returns:
        Tuple[bool, List[str], List[str]]: (是否通过验证, 错误列表, 警告列表)
    """
    # 加载数据
    if input_data is None:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

    if output_data is None:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)

    # 执行验证
    is_valid, errors, warnings = validate_all(input_data, output_data)

    return is_valid, errors, warnings

def load_and_process_data(input_path: str):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            print(f"[OK] Input data loaded successfully")
            print(f"  - Customers: {len(input_data['customers'])}")
            print(f"  - Vehicle types: {len(input_data['vehicles'])}")
            return input_data
    except Exception as e:
        print(f"[ERROR] Failed to read input data: {e}")
        return


def _kv_evaluate(path_user_py: str):
    """主函数：读取输入数据，调用求解器，验证输出"""
    import sys

    """
        评估函数
        """
    metrics = {
        "validity": 0.0,
        "combined_score": 0.0,
        "error_info": {},
        "max_duration": 0.0,
        "min_duration": 0.0
    }

    function = "process_cvrp_with_assign"
    input_path = r'input_data.json'
    output_file = r'output_data.json'

    # # 文件路径
    # 本地调试时由调用方提供输入与输出路径。

    try:
        logger.info("=" * 60)
        logger.info("VRP Vehicle Routing Problem Evaluation")
        logger.info("=" * 60)

        # 步骤1：读取输入数据
        # 1. 加载和处理数据
        logger.info(f"开始加载数据: {input_path}")
        input_data = load_and_process_data(input_path)

        # 2. 准备调用参数
        args = {
            # "customers": input_data['customers'],
            # "depot": input_data['depots'][0] if input_data['depots'] else {},
            # "depart": input_data['departs'][0] if input_data['departs'] else {},
            # "vehicles": input_data['vehicles'],
            # "locations": input_data['locations'],
            # "speed_dict": {},
            # "back_flag": input_data.get('backFlag', True),
            # "task_start_date": input_data.get('taskStartDate', '202603050800'),
            # "connect_size": input_data.get('connectSize', 5)
            "input_data":input_data
        }

        # 3. 检查被评估的Python文件
        if not os.path.isabs(path_user_py):
            path_user_py = os.path.abspath(path_user_py)
        path_user_py = path_user_py.replace('\x0c', '')

        logger.info(f"开始评估: {path_user_py}")

        # 4. 运行被评估代码（process_cvrp_with_packing 返回的是路线列表）
        output_data = run_with_timeout(path_user_py, function, args, timeout_seconds=2400)
        # 保存输出数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[OK] Output data saved to {output_file}")

        ## 步骤2：调用求解器生成初始解
        # print("\n[Step 2] Calling solver to generate initial solution...")
        # try:
        #     # output_data = process_cvrp_with_assign(
        #     #     customers=input_data['customers'],
        #     #     depot=input_data['depots'][0] if input_data['depots'] else {},
        #     #     depart=input_data['departs'][0] if input_data['departs'] else {},
        #     #     vehicles=input_data['vehicles'],
        #     #     locations=input_data['locations'],
        #     #     speed_dict={},
        #     #     back_flag=input_data.get('backFlag', True),
        #     #     task_start_date=input_data.get('taskStartDate', '202603050800'),
        #     #     connect_size=input_data.get('connectSize', 5)
        #     # )
        #     # print(f"[OK] Initial solution generated successfully")
        #     # print(f"  - Routes generated: {len(output_data)}")
        #
        #     # # 保存输出数据
        #     # with open(output_file, 'w', encoding='utf-8') as f:
        #     #     json.dump(output_data, f, ensure_ascii=False, indent=2)
        #     # print(f"[OK] Output data saved to {output_file}")
        #
        # except Exception as e:
        #     print(f"[ERROR] Solver failed: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return

        # 步骤3：验证输出
        logger.info("\n[Step 3] Validating output solution...")
        try:
            is_valid, errors, warnings = evaluate_solution(
                input_data=input_data,
                output_data=output_data
            )

            if is_valid:
                logger.info("[OK] Validation passed!")
                metrics["validity"] = 1.0
                metrics.update(score(output_data))
            else:
                logger.error("[ERROR] Validation failed!")
                metrics["error_info"]['validation_errors'] = errors

            # # 打印错误
            # if errors:
            #     print(f"\nErrors ({len(errors)}):")
            #     for i, error in enumerate(errors, 1):
            #         print(f"  {i}. {error}")
            #
            # # 打印警告
            # if warnings:
            #     print(f"\nWarnings ({len(warnings)}):")
            #     for i, warning in enumerate(warnings, 1):
            #         print(f"  {i}. {warning}")

        except Exception as e:
            logger.error(f"[ERROR] Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # 步骤4：输出统计信息
        logger.info("\n[Step 4] Statistics...")
        total_distance = sum(float(t.get('distance', 0) or 0) for t in output_data)
        total_duration = sum(float(t.get('duration', 0) or 0) for t in output_data)
        total_customers = sum(len(t.get('customers', []) or []) for t in output_data)

        logger.info(f"Total delivery distance: {total_distance / 1000:.2f} km")
        logger.info(f"Total delivery duration: {total_duration} minutes")
        logger.info(f"Total customers served: {total_customers}")
        if output_data:
            logger.info(f"Avg customers per route: {total_customers / len(output_data):.2f}")

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation completed")
        logger.info("=" * 60)

    except TimeoutError:
        logger.error("评估超时")
        metrics['error_info'] = {"timeout": "process timeout"}

    except Exception as e:
        logger.error(f"评估异常: {e}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"详细错误信息:\n{tb}")
        metrics["error_info"] = {
            "exception": str(e),
            "traceback": tb
        }

    return metrics

# ===================== 以下为原 evaluator.py =====================
kv = sys.modules[__name__]

PLAN_FILE = "solution.json"


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,
# 跟刷分的混在一起分不出高低。分母为零(完美解)时给一个有限哨兵值,避免 inf 破坏 JSON。
PERFECT_Q = 1e6

def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def _recompute_distance(trips, data_dir) -> float:
    """按 input_data.json 的 locations 经纬度复算全部 trip 的总里程(米)。

    路线口径:SITE_002 → 仓库(装货) → customers 按给定顺序 → 仓库(backFlag=True 时回程)。
    坐标查不到的点跳过(不计入),这样缺坐标不会凭空产生里程、也不会白送 0。
    """
    import math
    with open(os.path.join(data_dir, "input_data.json"), encoding="utf-8") as f:
        inp = json.load(f)
    loc = {l["id"]: (float(l["x"]), float(l["y"])) for l in inp.get("locations", [])}
    back = bool(inp.get("backFlag", True))
    depot_lid = (inp.get("depots") or [{}])[0].get("locationId")
    depart_lid = (inp.get("departs") or [{}])[0].get("locationId")

    def hav(a, b):
        if a is None or b is None:
            return 0.0
        (lo1, la1), (lo2, la2) = a, b
        R = 6371000.0
        p1, p2 = math.radians(la1), math.radians(la2)
        dp = math.radians(la2 - la1)
        dl = math.radians(lo2 - lo1)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(max(0.0, h)))

    def pt(obj_or_id):
        if isinstance(obj_or_id, dict):
            lid = obj_or_id.get("locationId") or obj_or_id.get("id")
        else:
            lid = obj_or_id
        return loc.get(lid)

    total = 0.0
    for t in trips:
        seq = []
        dp = pt(t.get("departPoint")) or pt(depart_lid)
        dep = pt(t.get("depot")) or pt(depot_lid)
        if dp:
            seq.append(dp)
        if dep:
            seq.append(dep)
        for c in (t.get("customers") or []):
            q = pt(c)
            if q:
                seq.append(q)
        if back and dep:
            seq.append(dep)
        for i in range(len(seq) - 1):
            total += hav(seq[i], seq[i + 1])
    return float(total)


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(
        validity_score=0.0,
        quality_score=0.0,
        overall_score=0.0,
        reference_value=0.0,
    )
    try:
        reference_value, direction = load_baseline()
    except Exception as e:
        m["error_info"] = {"fatal": f"baseline 读取失败: {e}"}
        return m
    m["reference_value"] = reference_value

    input_path = os.path.join(data_dir, "input_data.json")
    if not os.path.exists(input_path):
        m["error_info"] = {"fatal": f"缺 input_data.json: {input_path}"}
        return m
    try:
        with open(input_path, encoding="utf-8") as fh:
            input_data = json.load(fh)
    except Exception as e:
        m["error_info"] = {"fatal": f"input_data 解析失败: {e}"}
        return m

    plan_path = os.path.join(submission_dir, PLAN_FILE)
    if not os.path.exists(plan_path):
        m["error_info"] = {"hard_violations": [f"缺 {PLAN_FILE}"]}
        return m
    try:
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        m["error_info"] = {"hard_violations": [f"{PLAN_FILE} 解析失败: {e}"]}
        return m

    output_data = plan.get("trips", plan) if isinstance(plan, dict) else plan
    if not isinstance(output_data, list) or not output_data:
        m["error_info"] = {"hard_violations": ["solution 里没有 trips / 为空"]}
        return m

# 维护说明：校验核心对**提交侧**字段使用裸下标，需由此处捕获异常。
    # (如 order['order_id']),选手少写或写错一个键名就抛 KeyError 穿出
    # compute_metrics,整个评估崩掉而不是判不合格。现统一兜住,异常一律按硬违规处理。
    try:
        is_valid, errors, warnings = kv.evaluate_solution(
            input_data=input_data, output_data=output_data
        )
    except Exception as e:
        m["error_info"] = {
            "hard_violations": [f"方案结构不合法,校验中断: {type(e).__name__}: {e}"]
        }
        return m

    # 修复：kunlun_validator 不执行 connectSize（单趟最多串几家），也不校验 departPoint/
    # depot 是否就是 input_data 里给定的出发点/油库（指到油库可抹掉出发点→油库的空驶里程）。
    _extra = []
    _conn = input_data.get("connectSize")
    _dep_ids = {d.get("locationId") for d in (input_data.get("departs") or [])} | \
               {d.get("id") for d in (input_data.get("departs") or [])}
    _depot_ids = {d.get("locationId") for d in (input_data.get("depots") or [])} | \
                 {d.get("id") for d in (input_data.get("depots") or [])}
    def _lid(o):
        return (o.get("locationId") or o.get("id")) if isinstance(o, dict) else o
    for ti, t in enumerate(output_data, 1):
        if not isinstance(t, dict):
            continue
        cs = t.get("customers") or []
        if isinstance(_conn, (int, float)) and _conn > 0 and len(cs) > int(_conn):
            _extra.append(f"trip#{ti} 串点 {len(cs)} 家 > connectSize {int(_conn)}")
        dp = t.get("departPoint")
        if dp is not None and _dep_ids and _lid(dp) not in _dep_ids:
            _extra.append(f"trip#{ti} departPoint={_lid(dp)} 不是给定出发点")
        dpo = t.get("depot")
        if dpo is not None and _depot_ids and _lid(dpo) not in _depot_ids:
            _extra.append(f"trip#{ti} depot={_lid(dpo)} 不是给定油库")
    if _extra:
        is_valid = False
        errors = list(errors) + _extra

    if not is_valid:
        m["error_info"] = {"hard_violations": errors[:12], "total": len(errors)}
        return m

    try:
        sc = kv.score(output_data)
    except Exception as e:
        m["error_info"] = {"hard_violations": [f"计分中断: {type(e).__name__}: {e}"]}
        return m
    m["error_info"] = {}
    m["validity_score"] = 1.0
    # 2026-08-24 修:里程不再采信选手自报。原实现 total_distance 直接累加每条 trip 的
    # "distance" 字段在校验核心中也不按坐标复算 —— 交一份
    # {"distance": 1} 就能让 quality 撞 5.0 封顶。现按 input_data.json 的 locations 经纬度
    # 沿 SITE_002 → 仓库 → 各客户(按 customers 顺序) → 仓库(backFlag=True) 复算球面距离。
    # 自报值只保留为诊断字段,并在与复算值差得离谱时给出提示。
    total_distance = _recompute_distance(output_data, data_dir)
    self_reported = float(sum(t.get("distance", 0) or 0 for t in output_data))
    # 目标: 总行驶里程越低越好。
    # 2026-08-23 修:原实现把 total_distance(米,量级 4.9e6)直接当 quality_score,
    # 且把 load_baseline() 读出来的 direction 丢在一边从不使用 —— 结果是
    #   ① 方向反了:里程越短 quality 越小,优化得越好分数越低,排名整个倒置;
    #   ② 量纲不可比:全库其余 case 的 quality 都是"相对 baseline 的倍数"(≈1.0),
    #      这里却是百万量级的原始米数。
    # 改为与其余 case 一致的归一化:lower_is_better → quality = baseline / 选手值。
    # 不封顶(2026-08-25 全库统一去掉 5.0 cap,理由见文件头 PERFECT_Q 处);
    # 分母为零的完美解取有限哨兵值 PERFECT_Q,避免 inf 破坏 JSON。
    if direction == "higher_is_better":
        quality = total_distance / reference_value if reference_value > 0 else 0.0
    else:
        quality = reference_value / total_distance if total_distance > 0 else PERFECT_Q
    m["quality_score"] = round(float(quality), 6)
    m["overall_score"] = round(float(quality), 6)
    m["total_distance"] = total_distance
    m["player_objective"] = total_distance
    m["self_reported_distance"] = self_reported
    if total_distance > 0 and abs(self_reported - total_distance) / total_distance > 0.5:
        m.setdefault("warnings", []).append(
            f"自报里程 {self_reported:.0f} m 与按坐标复算的 {total_distance:.0f} m "
            f"相差超过五成，计分一律以复算值为准")
    m["combined_score_raw"] = sc["combined_score"]
    m["warnings"] = warnings[:12]
    return m


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
