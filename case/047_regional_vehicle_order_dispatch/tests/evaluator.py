"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  schema: |
    {
      "assignment": {
        "0": [16, 33],          # 车辆索引(字符串或整数) -> 该车承运的订单索引列表
        "1": [65, 145],
        "2": [],                # 未使用的车辆给空列表
        ...
      }
    }
  notes: >
    区配运输的车辆-订单分配。只需给出"哪些订单交给哪辆车", 不需要配送顺序或路径。
    - 车辆索引对应 car_info_detail 数组下标(0..30), 订单索引对应 order_info_detail 下标(0..148)。
    - 每个订单必须恰好出现一次; 所有 31 辆车都要出现(不用的车给空列表)。
    若选手交的是 CSV(order_index, car_index 两列), 按 car_index 归组成上面的字典。
    若用了订单号 eoorOrderNo / 车牌号而非下标, 按 data 里数组顺序换算成下标。
    列表里的元素必须是整数下标, 不要写成 "order_16" 这类字符串。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
PLAN_FILE = "solution.json"
DATA_FILE = "regional_dispatch_data.json"

# 与业务侧 offline_running.py 一致的目标权重
PARAM_DICT = {
    "baseline_cost_limited_ratio": 30,
    "distance_2_surrounding_order": 2,
    "order_geo_dist_limited": 30,
    "obj_weight_diameter": 500,
    "obj_weight_node": 6000,
    "obj_weight_community": 1500,
    "obj_weight_baseline_cost": 500,
    "obj_weight_nodegroup": 2000,
    "obj_weight_price": 1,
    "order_angle_limited": 45,
}




# ===================== 内联自 tests/_lib/Utils/node_related.py（评估器自包含，原文件已删除） =====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/10/24
# @Author  : benchmark-team
# @Email  : contact@example.invalid
# @FileName: node_related.py
"""

import logging
import math
from shapely.geometry import Polygon


def get_node_graph(node_data, neighbor_limited=0.0075):
    """获取节点信息图（graph）

    Args:
        node_data (dict): 静态数据中的节点信息数据
        neighbor_limited (float, optional): 判定为邻居的多边形距离限制. Defaults to 0.0075.
        output (bool, optional): 输出控制. Defaults to False.

    Returns:
        list, list, dict: 图节点信息列表，图边信息列表，节点索引到多边形中心点的映射
    """
    node_dict = {x["nodeName"]: x["locationTuple"] for x in node_data}
    node_list = list(node_dict.keys())
    node_index2name = {i: node_list[i] for i in range(0, len(node_list))}
    graph_node_list = list(node_index2name.keys())
    graph_edge_list = []
    node_index2center = {}
    # 遍历所有节点对，判断是否相邻
    for i in range(0, len(node_list)):
        node_i_name = node_index2name[i]
        node_i_location_list = node_dict[node_i_name]
        for j in range(i + 1, len(node_list)):
            node_j_name = node_index2name[j]
            node_j_location_list = node_dict[node_j_name]
            # 判断两个多边形是否相邻或距离小于阈值
            poly1 = Polygon(node_i_location_list)
            poly2 = Polygon(node_j_location_list)
            try:
                if i not in node_index2center.keys():
                    node_index2center[i] = (poly1.centroid.x, poly1.centroid.y)
                if j not in node_index2center.keys():
                    node_index2center[j] = (poly2.centroid.x, poly2.centroid.y)
            except Exception as e:
                print(node_i_location_list)
                print(node_j_location_list)
                raise Exception(e)
            neighbor_flag = False
            if poly1.touches(poly2) or poly1.intersects(poly2):
                logging.debug("[GraphBuild] {} and {} is torched or intersected".format(node_i_name, node_j_name))
                neighbor_flag = True
            if not neighbor_flag:
                min_distance = poly1.distance(poly2)
                if min_distance < neighbor_limited:
                    logging.debug("[GraphBuild] {} and {} distance is {} below than {}".format(
                                  node_i_name, node_j_name, min_distance, neighbor_limited))
                    neighbor_flag = True
                else:
                    pass
            # 如果两个节点相邻，则添加一条边
            if neighbor_flag:
                graph_edge_list.append((i, j)) 
    return graph_node_list, graph_edge_list, node_index2center


def euler_distance_based_location(order_1_center, order_2_center):
    """计算两个服务区域中心点之间的欧式距离

    Args:
        order_1_center (tuple): 订单1的中心经纬度坐标
        order_2_center (tuple): 订单2的中心经纬度坐标

    Returns:
        float: 两者之间的欧氏距离
    """
    lon1, lat1 = order_1_center
    lon2, lat2 = order_2_center
    # 将十进制度数转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # 计算经纬度差
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    # 地球半径（单位：公里）
    r = 6371.0
    # 计算距离
    distance = c * r
    return distance


def manhattan_distance_based_location(order_1_center, order_2_center):
    """计算两个服务区域中心点之间的曼哈顿距离

    Args:
        order_1_center (tuple): 订单1的中心经纬度坐标
        order_2_center (tuple): 订单2的中心经纬度坐标

    Returns:
        float: 两者之间的曼哈顿距离
    """
    lon1, lat1 = order_1_center
    lon2, lat2 = order_2_center
    # 平均每度纬度相差的公里数（大约）
    km_per_degree_lat = 111.32
    # 平均每度经度相差的公里数（因纬度而异，这里使用在赤道的值作为近似）
    km_per_degree_lon = 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    # 计算经纬度差异
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    # 计算曼哈顿距离
    distance = km_per_degree_lon * dlon + km_per_degree_lat * dlat
    return distance


def calculate_bearing(order_1_center, order_2_center):
    """计算两个地点之间的方位角
    Args:
        order_1_center (tuple): 订单1的中心经纬度坐标
        order_2_center (tuple): 订单2的中心经纬度坐标

    Returns:
        float: 两点之间的角度数值
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [order_1_center[1], order_1_center[0],
                                                order_2_center[1], order_2_center[0]])
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# ===================== 内联自 tests/_lib/Utils/mutex_related.py（评估器自包含，原文件已删除） =====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/10/28
# @Author  : benchmark-team
# @Email  : contact@example.invalid
# @FileName: mutex_related.py
"""

import logging
import datetime
import networkx as nx


URBAN_AREA = ['海淀', '西城', "东城", "朝阳", "顺义", "石景山", "丰台"]
DISTANCE_MAGNIFICATION = 2.5


def get_order_car_mutex(car_info_detail, order_info_detail):
    """生成车辆订单互斥信息列表

    Args:
        car_info_detail (list): 车辆详细信息列表
        order_info_detail (list): 订单详细信息列表

    Returns:
        list: 包含(订单i, 车辆j)发生互斥的列表
    """
    order_car_mutex = []
    zone_num = len(car_info_detail[0]["carAvailableZone"])
    for o_index in range(0, len(order_info_detail)):
        order_info = order_info_detail[o_index]
        all_car_mutex_label = 0
        mutex_info_str = ""
        for c_index in range(0, len(car_info_detail)):
            car_info = car_info_detail[c_index]
            car_type = car_info["carType"]
            # 失败车型不产生互斥
            if car_info["carType"] == "CAR_TYPE_001":
                continue
            # 规则1：地库限制
            if car_info["car2Basement"] == 0 and order_info["orderAtBasement"] == 1:
                order_car_mutex.append((o_index, c_index))
                logging.debug("[OCMutex] Order {}[{}] is at basement but Car {}[{}] can't go there.".format(
                    o_index, order_info["orderID"], c_index, car_info["carType"]
                ))
                mutex_info_str += "{}-地库限制,".format(car_type)
                all_car_mutex_label += 1
                continue
            # 规则2: 高度限制
            if order_info["orderHeightLimited"] != -1 and \
               car_info["carHeight"] > order_info["orderHeightLimited"]:
                order_car_mutex.append((o_index, c_index))
                logging.debug("[OCMutex] Order {}[{}] have {} height limit but Car {}[{}] height {}.".format(
                    o_index, order_info["orderID"], order_info["orderHeightLimited"],
                    c_index, car_info["carType"], car_info["carHeight"]
                ))
                mutex_info_str += "{}-高度限制,".format(car_type)
                all_car_mutex_label += 1
                continue
            # 规则3：特殊件限制
            special_good_size_list = []
            for good_info in order_info["orderGoods"]:
                if good_info["isSpecialGoods"] == 1 and good_info["goodsMarketModel"] not in special_good_size_list:
                    special_good_size_list.append(good_info["goodsMarketModel"])
            if len(special_good_size_list) > 0 and order_info["orderAtBasement"] == 1:
                logging.warning("[OCMutex] Order {}[{}] has special goods but at basement, label -> 0".format(
                    o_index, order_info["orderID"]
                ))
                special_good_size_list = []
            is_mutex_label = False
            for special_good_size in special_good_size_list:
                if special_good_size not in car_info["carTransportSpecial"]:
                    order_car_mutex.append((o_index, c_index))
                    logging.debug("[OCMutex] Order {}[{}] have special goods but Car {}[{}] can't cover it.".format(
                        o_index, order_info["orderID"], c_index, car_info["carType"]
                    ))
                    all_car_mutex_label += 1
                    mutex_info_str += "{}-特殊件限制,".format(car_type)
                    is_mutex_label = True
                    break
            if is_mutex_label:
                continue
            # 规则4：尾号被限行
            # NOTE: 待实现，需要和司机信息联动
            # 规则5： 区域被限行
            if len(order_info["orderDependentZone"]) != zone_num:
                logging.error("[OCMutex] Zone Should be same, Order-{} has wrong number with {} not {}".format(
                    order_info["orderID"], len(order_info["orderDependentZone"]), zone_num
                ))
                raise Exception("[OCMutex] Zone Should be same, Order-{} has wrong number with {} not {}".format(
                    order_info["orderID"], len(order_info["orderDependentZone"]), zone_num
                ))
            is_mutex_label = False
            for z_index in range(0, zone_num):
                if order_info["orderDependentZone"][z_index] == 1 and car_info["carAvailableZone"][z_index] == 0:
                    order_car_mutex.append((o_index, c_index))
                    logging.debug("[OCMutex] Order {}[{}] is in zone {} but Car {}[{}] can't go there.".format(
                        o_index, order_info["orderID"], z_index, c_index, car_info["carType"]
                    ))
                    all_car_mutex_label += 1
                    mutex_info_str += "{}-区域限制,".format(car_type)
                    is_mutex_label = True
                    break
            if is_mutex_label:
                continue
    return order_car_mutex


def get_order_order_mutex(topology_graph, node_name2index,
                          order_info_detail, order_distance_dict,
                          order_span_node_limited=5,
                          max_distance_limited=40,
                          distance_2_surrounding_order=15):
    """生成订单订单互斥信息列表

    Args:
        topology_graph (networkx.Graph): 拓扑图结构
        node_name2index (dict): 节点名称到图中Node索引的映射字典
        order_info_detail (list): 订单详细信息列表
        order_distance_dict (dict): 订单间距离字典
        order_span_node_limited (int, optional): 订单间最大节点数距离. Defaults to 5.
        max_distance_limited (int, optional): 订单间最大距离限制. Defaults to 40.
        distance_2_surrounding_order (int, optional): 被算作仓库周边订单的距离限制. Defaults to 15.

    Returns:
        list: 包含(订单i1, 订单i2)发生互斥的列表
    """
    order_order_mutex = []
    mutex_count = {"rule_1": 0, "rule_2": 0}
    for o_index_1 in range(0, len(order_info_detail)):
        order_info_1 = order_info_detail[o_index_1]
        # 特殊情况：距离仓库15KM内的订单和其他订单不冲突
        if order_info_1["orderDistance2Storage"] <= distance_2_surrounding_order:
            logging.debug("[OOMutex] Order {}[{}] is near the storage area".format(o_index_1, order_info_1["orderID"]))
            continue
        for o_index_2 in range(o_index_1 + 1, len(order_info_detail)):
            order_info_2 = order_info_detail[o_index_2]
            # 特殊情况：距离仓库distance_2_surrounding_order内的订单和其他订单不冲突
            if order_info_2["orderDistance2Storage"] <= distance_2_surrounding_order:
                continue
            # 标准约束1：订单不能跨越order_span_node_limited个区块以上
            if order_span_node_limited > 0:
                if order_info_1["orderNode"] not in node_name2index.keys():
                    logging.warning("[OOMutex] Node {} not in graph!".format(order_info_1["orderNode"]))
                    continue
                if order_info_2["orderNode"] not in node_name2index.keys():
                    logging.warning("[OOMutex] Node {} not in graph!".format(order_info_2["orderNode"]))
                    continue
                distance = nx.shortest_path_length(topology_graph,
                                                source=node_name2index[order_info_1["orderNode"]],
                                                target=node_name2index[order_info_2["orderNode"]])
                if distance > order_span_node_limited:
                    order_order_mutex.append((o_index_1, o_index_2))
                    logging.debug("[OOMutex] Order {}[{}] and Order {}[{}] Node-Dist {} but limited {}".format(
                        o_index_1, order_info_1["orderID"], o_index_2, order_info_2["orderID"],
                        distance, order_span_node_limited
                    ))
                    mutex_count["rule_1"] += 1
            # 标准约束2： 订单不能跨越max_distance_limited公里以上
            distance = order_distance_dict[(o_index_1, o_index_2)]
            order_1_in_urban = False
            for area_name in URBAN_AREA:
                if area_name in order_info_1["recipientAddress"][:9]:
                    order_1_in_urban = True
                    break
            order_2_in_urban = False
            for area_name in URBAN_AREA:
                if area_name in order_info_2["recipientAddress"][:9]:
                    order_2_in_urban = True
                    break
            if order_1_in_urban == False and order_2_in_urban == False:
                mdl = max_distance_limited * DISTANCE_MAGNIFICATION
            else:
                mdl = max_distance_limited 
            if distance > mdl:
                order_order_mutex.append((o_index_1, o_index_2))
                logging.debug("[OOMutex] Order {}[{}] and Order {}[{}] Mean-Dist {} but limited {}".format(
                    o_index_1, order_info_1["orderID"], o_index_2, order_info_2["orderID"],
                    distance, mdl
                ))
                mutex_count["rule_2"] += 1
    logging.info("[OOMutex] Mutex count: {}".format(mutex_count))
    
    return order_order_mutex


def get_order2order_node_distance(topology_graph, node_name2index,
                                  order_info_detail,
                                  distance_2_surrounding_order=15):
    """计算订单与订单之间的服务区域距离"""
    order_order_node_distance = {}
    for o_index_1 in range(0, len(order_info_detail)):
        order_info_1 = order_info_detail[o_index_1]
        for o_index_2 in range(o_index_1 + 1, len(order_info_detail)):
            order_info_2 = order_info_detail[o_index_2]
            if order_info_2["orderDistance2Storage"] <= distance_2_surrounding_order or \
               order_info_1["orderDistance2Storage"] <= distance_2_surrounding_order:
                order_order_node_distance[(o_index_1, o_index_2)] = 0
                continue
            if order_info_1["orderNode"] not in node_name2index.keys():
                logging.warning("[OOMutex] Node {} not in graph!".format(order_info_1["orderNode"]))
                continue
            if order_info_2["orderNode"] not in node_name2index.keys():
                logging.warning("[OOMutex] Node {} not in graph!".format(order_info_2["orderNode"]))
                continue
            distance = nx.shortest_path_length(topology_graph,
                                               source=node_name2index[order_info_1["orderNode"]],
                                               target=node_name2index[order_info_2["orderNode"]])
            order_order_node_distance[(o_index_1, o_index_2)] = distance
    return order_order_node_distance


def get_order2order_angle_diff(static_data, order_info_detail, distance_2_surrounding_order=15):
    """计算订单与订单之间的角度差值"""
    order_order_angle_diff = {}
    location_coordinate = static_data["locationCoordinate"]
    for o_index_1 in range(0, len(order_info_detail)):
        order_info_1 = order_info_detail[o_index_1]
        for o_index_2 in range(o_index_1 + 1, len(order_info_detail)):
            order_info_2 = order_info_detail[o_index_2]
            if order_info_2["orderDistance2Storage"] <= distance_2_surrounding_order or \
               order_info_1["orderDistance2Storage"] <= distance_2_surrounding_order:
                order_order_angle_diff[(o_index_1, o_index_2)] = 0
                order_order_angle_diff[(o_index_2, o_index_1)] = 0
                continue
            order_center_1 = order_info_1["recipientAddLngLat"]
            order_center_2 = order_info_2["recipientAddLngLat"]
            order_angle_1 = calculate_bearing(location_coordinate, order_center_1)
            order_angle_2 = calculate_bearing(location_coordinate, order_center_2)
            angle_max, angle_min = max(order_angle_1, order_angle_2), min(order_angle_1, order_angle_2)
            angle_diff = min(angle_max - angle_min, 360 - angle_max + angle_min)
            order_order_angle_diff[(o_index_1, o_index_2)] = angle_diff
            order_order_angle_diff[(o_index_2, o_index_1)] = angle_diff
    return order_order_angle_diff

# ===================== 内联自 tests/_lib/Utils/constant_build.py（评估器自包含，原文件已删除） =====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2025/11/27
# @Author  : benchmark-team
# @Email  : contact@example.invalid
# @FileName: constant_build.py
"""

import time
import logging


class ConstantBuild:
    """构建计算用常量"""
    
    def __init__(self,
                 static_data,
                 car_info_detail,
                 order_info_detail,
                 graph_data_dict,
                 distance_2_surrounding_order):
        """初始化函数"""
        # 输入数据
        self.static_data = static_data
        self.car_info_detail = car_info_detail
        self.order_info_detail = order_info_detail
        self.G = graph_data_dict["G"]
        self.node_index2center = graph_data_dict["node_index2center"]
        self.node_name2index = graph_data_dict["node_name2index"]
        self.distance_2_surrounding_order = distance_2_surrounding_order
        # 安装时间数据
        self.good_install_time = {x["goodType"]: float(x["installWorkDuration"])
                                  for x in static_data["goodsCapabilityData"]}
        self.good_repair_time = {x["goodType"]: float(x["repairerWorkDuration"])
                                 for x in static_data["goodsCapabilityData"]}
        self.good_handle_time = {(x["floorCodeType"], x["floorRealityNum"]): x["avgWorkerNum"]
                                 for x in static_data["workRule"]}
    
    def _get_car_constant(self):
        """计算车辆相关常量"""
        car_info_detail = self.car_info_detail
        # 模型用车辆信息提取
        car_info_dict = {
            "car_num": len(car_info_detail),  # 车辆数量
            "car_id_list": [x["carID"] for x in car_info_detail],  # 车辆j的ID
            "car_baseline_cost": [x["carBaselineCost"] for x in car_info_detail],  # 车辆j的基准成本
            "car_standard_load": [x["carStandardLoad"] * 1000 for x in car_info_detail],  # 车辆j的标准载重
            "car_load_cz_lb": [x["loadRangeCiZhuan"][0] * 1000 for x in car_info_detail],  # 车辆j在装载瓷砖时的装载下限
            "car_load_cz_ub": [x["loadRangeCiZhuan"][1] * 1000 for x in car_info_detail],  # 车辆j在装载瓷砖时的装载上限
            "car_load_wy_lb": [x["loadRangeWeiYu"][0] * 1000 for x in car_info_detail],  # 车辆j在装载卫浴时的装载下限
            "car_load_wy_ub": [x["loadRangeWeiYu"][1] * 1000 for x in car_info_detail],  # 车辆j在装载卫浴时的装载上限
            "car_load_mix_lb": [x["loadRangeHunZhuang"][0] * 1000 for x in car_info_detail],  # 车辆j在装载混装时的装载下限
            "car_load_mix_ub": [x["loadRangeHunZhuang"][1] * 1000 for x in car_info_detail],  # 车辆j在装载混装时的装载上限
            "car_overload_limited": [x["carOverloadLimited"] * 1000 for x in car_info_detail],  # 车辆j的超重触发阈值
            "car_max_order": [x["carMaxOrder"] for x in car_info_detail],  # 车辆j的最大订单数
            "car_max_server_time": [x["carMaxWorkDuration"] * 60 for x in car_info_detail],  # 车辆j的最长服务时间
            "car_span_node_limited": [x["carSpanNodeLimited"] for x in car_info_detail]  # 车辆j的跨节点（服务区域）限制阈值
        }
        logging.info("[ConstantProcess-Car] Car Info Load Finish, #Car = {}".format(len(car_info_detail)))
        return car_info_dict
    
    def _get_order_constant(self):
        """计算订单相关常量"""
        order_info_detail = self.order_info_detail
        
        # 订单类型信息&订单总工作时间提取
        type_cz, type_wy, type_dz, order_worker_time_list, order_need_node_price = [], [], [], [], []
        for item in order_info_detail:
            is_cz, is_wy, is_dz = False, False, False
            order_install_time, order_repair_time, order_handle_time = 0, 0, 0 
            for goods in item["orderGoods"]:
                # 类型处理
                if goods["goodsClass"] == "瓷砖":
                    is_cz = True
                if goods["goodsClass"] == "卫浴":
                    is_wy = True
                if goods["goodsClass"] == "定制":
                    is_dz = True
                # 安装&维修时间处理
                if "安装" in item["orderType"]:
                    order_install_time += self.good_install_time[goods["goodsType"]] * goods["goodsQuantity"]
                if "维修" in item["orderType"]:
                    order_repair_time += self.good_repair_time[goods["goodsType"]] * goods["goodsQuantity"]
            type_cz.append(1 if is_cz else 0)
            type_wy.append(1 if is_wy else 0)
            type_dz.append(1 if is_dz else 0)
            # 搬运时间
            if "无运输" in item["orderType"]:
                order_handle_time = 0
                order_need_node_price.append(0)
            elif "退货" in item["orderType"]:
                order_handle_time = 15
                order_need_node_price.append(1)
            else:
                order_need_node_price.append(1)
                good_handle_key = (item["orderFloorCode"], item["orderFloorNum"])
                if good_handle_key in self.good_handle_time.keys():
                    use_good_handle_key = good_handle_key
                    order_handle_time = item["orderWeight"] / 1000 * self.good_handle_time[good_handle_key]
                else:
                    logging.debug("[ConstantProcess-Order] No GoodHandleTime Found for FloorKey-{}".format(
                        good_handle_key))
                    use_good_handle_key = (item["orderFloorCode"], "0")
                    order_handle_time = item["orderWeight"] / 1000 * \
                        self.good_handle_time[(item["orderFloorCode"], "0")]
                if "backOrderNum" in item.keys():
                    order_handle_time += item["backOrderNum"] * 15
                order_handle_time = round(order_handle_time, 4)
                # logging.info("[DEBUG] For O-{}, IT = {}, RT = {}, HT = {}({})".format(
                #     item["orderID"], order_install_time, order_repair_time, order_handle_time, good_handle_key
                # ))
            order_worker_time_list.append(order_install_time + order_repair_time + order_handle_time)
        
        # 订单是否可以甩单标记
        order_can_release = []
        for o_index_1 in range(0, len(order_info_detail)):
            order_info_1 = order_info_detail[o_index_1]
            # -- 外地地址场景
            if order_info_1["orderOutsideRegion"] not in [0, "0"]:
                order_can_release.append(1)
                logging.info("[ConstantProcess-Order] Order {} Can Release By OutsideRegion: {}, Address: {}".format(
                    order_info_1["orderID"], order_info_1["orderOutsideRegion"], order_info_1["recipientAddress"]
                ))
                continue
            # -- 计算最小距离
            min_distance = 1E10
            for o_index_2 in range(0, len(order_info_detail)):
                if o_index_1 == o_index_2:
                    continue
                order_1_center = order_info_detail[o_index_1]["recipientAddLngLat"]
                order_2_center = order_info_detail[o_index_2]["recipientAddLngLat"]
                distance = manhattan_distance_based_location(order_1_center, order_2_center)
                if distance < min_distance:
                    min_distance = distance
            if min_distance > 30:
                order_can_release.append(1)
                logging.info("[ConstantProcess-Order] Order {} Can Release By Distance: {:.4f}km".format(
                    order_info_1["orderID"], min_distance
                ))
            else:
                order_can_release.append(0)
        if len(order_info_detail) < 2:
            logging.warning("[ConstantProcess-Order] Order Num {} < 2, All Order Cannot Release".format(
                len(order_info_detail)))
            order_can_release = [1] * len(order_info_detail)
            
        # 计算每种车在每个订单的行驶时间
        order_travel_time = {}
        for c_index in range(0, len(self.car_info_detail)):
            temp_list = []
            for o_index in range(0, len(order_info_detail)):
                travel_distance = order_info_detail[o_index]["orderDistance2Storage"]
                travel_region = order_info_detail[o_index]["recipientDistrictCode"]
                car_speed = self.car_info_detail[c_index]["carAvgSpeed"][travel_region]
                travel_time = round(travel_distance / car_speed * 60, 2)
                temp_list.append(travel_time)
            order_travel_time[c_index] = temp_list
        
        # 每个订单相对基地的方位角度
        order_azimuth_angle = []
        location_coordinate = self.static_data["locationCoordinate"]
        for o_index in range(0, len(order_info_detail)):
            order_center = order_info_detail[o_index]["recipientAddLngLat"]
            order_azimuth_angle.append(calculate_bearing(location_coordinate, order_center))
        
        # 订单是否是仓库周边订单
        order_surrounding = []
        for o_index in range(0, len(order_info_detail)):
            order_distance = order_info_detail[o_index]["orderDistance2Storage"]
            order_surrounding.append(1 if order_distance <= self.distance_2_surrounding_order else 0)
            
        # 计算订单之间的曼哈顿距离
        order_distance_dict = {}
        for o_index_1 in range(0, len(order_info_detail)):
            for o_index_2 in range(o_index_1 + 1, len(order_info_detail)):
                angle_max = max(order_azimuth_angle[o_index_1], order_azimuth_angle[o_index_2])
                angle_min = min(order_azimuth_angle[o_index_1], order_azimuth_angle[o_index_2])
                angle_diff = min(angle_max - angle_min, (360 + angle_min) - angle_max)        
                order_1_distance2storage = order_info_detail[o_index_1]["orderDistance2Storage"]
                order_2_distance2storage = order_info_detail[o_index_2]["orderDistance2Storage"]
                # -- 如果其中一个是仓库周边订单，则距离认为是0
                if order_1_distance2storage <= self.distance_2_surrounding_order or \
                   order_2_distance2storage <= self.distance_2_surrounding_order:
                    order_distance_dict[(o_index_1, o_index_2)] = 0
                    order_distance_dict[(o_index_2, o_index_1)] = 0
                # -- 如果一个订单和另一个订单方位角差值小于正负5度，且二者仓库间距离差值小于20KM，则距离减半
                elif angle_diff <= 5:
                    order_1_center = order_info_detail[o_index_1]["recipientAddLngLat"]
                    order_2_center = order_info_detail[o_index_2]["recipientAddLngLat"]
                    distance = manhattan_distance_based_location(order_1_center, order_2_center)
                    if distance < 15:
                        order_distance_dict[(o_index_1, o_index_2)] = distance / 2
                        order_distance_dict[(o_index_2, o_index_1)] = distance / 2
                    else:
                        order_distance_dict[(o_index_1, o_index_2)] = distance
                    order_distance_dict[(o_index_2, o_index_1)] = distance
                # -- 其他情况正常计算曼哈顿距离
                else:
                    order_1_center = order_info_detail[o_index_1]["recipientAddLngLat"]
                    order_2_center = order_info_detail[o_index_2]["recipientAddLngLat"]
                    distance = manhattan_distance_based_location(order_1_center, order_2_center)
                    order_distance_dict[(o_index_1, o_index_2)] = distance
                    order_distance_dict[(o_index_2, o_index_1)] = distance
                    
        # 订单服务区域中心距离仓库的距离
        order_node_distance_2_storage = []
        for o_index in range(0, len(order_info_detail)):
            order_node = order_info_detail[o_index]["orderNode"]
            node_center = self.node_index2center[self.node_name2index[order_node]]
            node_distance = euler_distance_based_location(node_center, location_coordinate)
            order_node_distance_2_storage.append(round(node_distance, 2))
            
        # 订单信息整合
        order_info_dict = {
            "order_num": len(order_info_detail),  # 订单数量
            "order_id_list": [x["orderID"] for x in order_info_detail],  # 订单ID列表
            "order_weight": [round(x["orderWeight"], 2) for x in order_info_detail],  # 订单i的重量
            "type_cz": type_cz,  # 订单i是否包含瓷砖类型
            "type_wy": type_wy,  # 订单i是否包含卫浴类型
            "order_need_car_tail": [x["orderNeedCarTail"] for x in order_info_detail],  # 订单i是否需要车辆考虑尾号限行
            "order_distance_2_storage": [x["orderDistance2Storage"] for x in order_info_detail],  # 订单i与仓库的距离
            "order_node_distance_2_storage": order_node_distance_2_storage,  # 订单i的服务区域节点与仓库的距离
            "order_distance_dict": order_distance_dict,  # 订单距离字典，键(i1,i2)对应的值为订单i1与订单i2的距离
            "order_work_time": order_worker_time_list,  # 订单i需要的工作时间
            "order_travel_time": order_travel_time,  # 订单i需要的行驶时间
            "order_can_release": order_can_release,  # 订单i是否可以考虑被甩单
            "order_azimuth_angle": order_azimuth_angle,  # 订单i相对仓库的方位角
            "order_surrounding": order_surrounding,  # 订单i是否是仓库周边订单
            "order_need_node_price": order_need_node_price  # 订单i是否需要考虑节点价格
        }
        self.order_distance_dict = order_distance_dict
        logging.info("[ConstantProcess-Order] Order Info Load Finish, #Order = {}, TotalWeight = {:.2f}".format(
            len(order_info_detail), sum(order_info_dict["order_weight"])))
        return order_info_dict

    def get_node_comunity_constant(self):
        """计算服务区域Node&小区Community相关常量"""
        # 节点列表和节点分数
        used_node_list = [self.node_name2index[self.order_info_detail[i]["orderNode"]]
                          for i in range(0, len(self.order_info_detail))]
        used_node_list = list(set(used_node_list))
        node_score = {}
        for order_index in range(0, len(self.order_info_detail)):
            node_index = self.node_name2index[self.order_info_detail[order_index]["orderNode"]]
            if node_index not in node_score.keys():
                node_score[node_index] = 1
            else:
                node_score[node_index] += 1
        for key in node_score.keys():
            logging.debug("[ConstantProcess-Node] Node {} ({}) Score = {}".format(
                key, [k for k, v in self.node_name2index.items() if v == key], node_score[key]
            ))
            node_score[key] = node_score[key] * node_score[key]
            
        # 社区列表和社区分数
        community_list = list(set([tuple(x["recipientAddressSplit"]) for x in self.order_info_detail]))
        community_value2index = {community_list[index]: index for index in range(0, len(community_list))}
        community_score = {}
        for order_index in range(0, len(self.order_info_detail)):
             conmunity_index = community_value2index[tuple(self.order_info_detail[order_index]["recipientAddressSplit"])]
             if conmunity_index not in community_score.keys():
                 community_score[conmunity_index] = 1
             else:
                 community_score[conmunity_index] += 1
        for key in community_score.keys():
            logging.debug("[ConstantProcess-Node] Community {} ({}) Score = {}".format(
                key, [k for k, v in community_value2index.items() if v == key], community_score[key]
            ))
            community_score[key] = community_score[key] * community_score[key]
            
        # “大区”处理逻辑
        order_not_in_same_group = []
        node_name2group = {x["nodeName"]: x["nodeGroup"] for x in self.static_data["nodeData"]}
        for o_index_1 in range(0, len(self.order_info_detail)):
            for o_index_2 in range(o_index_1 + 1, len(self.order_info_detail)):
                order_node_1 = self.order_info_detail[o_index_1]["orderNode"]
                order_node_2 = self.order_info_detail[o_index_2]["orderNode"]
                group_list_1 = node_name2group[order_node_1]
                group_list_2 = node_name2group[order_node_2]
                if len(group_list_1) == 0 or len(group_list_2) == 0:
                    continue
                if len(set(group_list_1).intersection(set(group_list_2))) == 0:
                    order_not_in_same_group.append((o_index_1, o_index_2))
        logging.info("[ConstantProcess-Node] Not Same Group Orders Count = {}".format(len(order_not_in_same_group)))
        
        # 节点&社区信息整合
        node_info_dict = {
            "node_list": used_node_list,  # 服务区域（节点）列表
            "order2node": [(i, self.node_name2index[self.order_info_detail[i]["orderNode"]]) 
                            for i in range(0, len(self.order_info_detail))],  # 订单与服务区域的对应关系，内部的元素(i,n)表示订单i对应服务区域节点n
            "node_score": node_score,  # 服务区域n的分数，为包含订单数的平方
            "community_list": list(range(0, len(community_list))),  # 小区列表
            "order2community": [(i, community_value2index[tuple(self.order_info_detail[i]["recipientAddressSplit"])])
                                for i in range(0, len(self.order_info_detail))], # 订单与小区的对应关系，内部的元素(i,c)表示订单i对应小区c
            "community_score": community_score,  # 小区c的分数，为包含订单数的平方
            "order_not_in_same_group": order_not_in_same_group  # 不在同一“大区”内的订单对列表，内部元素(i1,i2)表示订单i1和订单i2不在同一“大区”内
        }
        logging.info("[ConstantProcess-Node] Node Community Info Load Finish, #Node = {}, #Community = {}".format(
            len(node_info_dict["node_list"]), len(node_info_dict["community_list"])))
        return node_info_dict
    
    def _get_mutex_constant(self):
        """计算互斥相关常量"""
        # 获取车辆C-订单O互斥信息和订单O-订单O互斥信息
        try:
            car_order_mutex = get_order_car_mutex(self.car_info_detail,
                                                  self.order_info_detail)
            logging.info("[ConstantProcess] CO Mutex Info Finish, #Tuple = {}".format(len(car_order_mutex)))
            order_order_mutex = get_order_order_mutex(self.G, self.node_name2index, self.order_info_detail,
                                                      self.order_distance_dict,
                                                      order_span_node_limited=-1,
                                                      max_distance_limited=1000,
                                                      distance_2_surrounding_order=self.distance_2_surrounding_order)
            logging.info("[ConstantProcess] OO Mutex Info Finish, #Tuple = {}".format(len(order_order_mutex)))
            order2order_node_distance = get_order2order_node_distance(
                self.G, self.node_name2index, self.order_info_detail,
                distance_2_surrounding_order=self.distance_2_surrounding_order
            )
            order2order_angle_diff = get_order2order_angle_diff(
                self.static_data, self.order_info_detail,
                distance_2_surrounding_order=self.distance_2_surrounding_order
            
            )
            mutex_info_dict = {
                "order_car_mutex": car_order_mutex,  # 车辆-订单互斥元组列表，内部元素(c,o)表示车辆c与订单o互斥
                "order_order_mutex": order_order_mutex,  # 订单-订单互斥元组列表，内部元素(o1,o2)表示订单o1与订单o2互斥
                "order2order_node_distance": order2order_node_distance,  # 订单对之间的服务区域节点距离字典，键(o1,o2)对应的值为订单o1与订单o2的服务区域节点距离
                "order2order_angle_diff": order2order_angle_diff  # 订单对之间的方位角差值字典，键(o1,o2)对应的值为订单o1与订单o2的方位角差值 
            }
        except Exception as e:
            logging.info("[ConstantProcess] Meet Order All Mutex!")
            raise Exception("Meet Order All Mutex, Error Msg: {}".format(repr(e)))
        
        logging.info("[ConstantProcess-Mutex] Mutex Info Load Finish")
        return mutex_info_dict

    def _get_price_constant(self):
        """计算价格相关常量"""
        # 获取价格信息
        car_start_price = {}
        for c_index in range(0, len(self.car_info_detail)):
            car_info = self.car_info_detail[c_index]
            car_type = car_info["carType"]
            node_price_dict = {x["nodeName"]: {y["carType"]: y["nodePrice"] for y in x["price"]}
                                for x in self.static_data["nodePriceData"]}
            for o_index in range(0, len(self.order_info_detail)):
                order_info = self.order_info_detail[o_index]
                if order_info["orderNode"] not in node_price_dict.keys():
                    logging.error("[ConstantProcess] Order Node [{}] not found in node_price".format
                                  (order_info["orderNode"]))
                    raise Exception("车型订单调度失败，订单的节点‘{}’"\
                        "没有在静态数据‘节点价格’中找到".format(order_info["orderNode"]))
                if car_type not in node_price_dict[order_info["orderNode"]].keys():
                    logging.error("[ConstantProcess] Car Type [{}] not found in node_price_data of Node [{}]".format(
                        car_type, order_info["orderNode"]))
                    raise Exception("车型订单调度失败，车型'{}'"\
                        "没有在静态数据‘节点价格’中找到".format(car_type))
                price = node_price_dict[order_info["orderNode"]][car_type]
                car_start_price[(o_index, c_index)] = price
        # 价格信息整合
        price_info_dict = {
            "car_start_price": car_start_price,  # 订单-车辆起步价字典，键(o,c)对应的值为订单o使用车辆c的起步价
            "car_node_price": [x["carNodePrice"] for x in self.car_info_detail],  # 车辆j的节点单价
            "car_overload_price": [x["carOverloadPrice"] / 1000 for x in self.car_info_detail],  # 车辆j的超重单价
            "car_price_punishment": [x["carPricePunishment"] for x in self.car_info_detail]  # 车辆j的使用惩罚价
        }
        
        logging.info("[ConstantProcess-Price] Price Info Load Finish")
        return price_info_dict

    def main_process(self):
        """常量计算主函数"""
        start_time = time.time()
        # 车辆相关常量
        car_info_dict = self._get_car_constant()
        # 订单相关常量
        order_info_dict = self._get_order_constant()
        # 节点&小区相关常量
        node_info_dict = self.get_node_comunity_constant()
        # 互斥相关常量
        mutex_info_dict = self._get_mutex_constant()
        # 价格相关常量
        price_info_dict = self._get_price_constant()
        # 常量整合
        constant_dict = {
            "car_info_dict": car_info_dict,
            "order_info_dict": order_info_dict,
            "node_info_dict": node_info_dict,
            "mutex_info_dict": mutex_info_dict,
            "price_info_dict": price_info_dict
        }
        end_time = time.time()
        logging.info("[ConstantProcess] Constant Build Finish, TimeCost = {:.2f}s".format(end_time - start_time))
        return constant_dict

# ===================== 内联自 tests/_lib/result_evaluate.py（评估器自包含，原文件已删除） =====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2025/11/27
# @Author  : benchmark-team
# @Email  : contact@example.invalid
# @FileName: result_evaluate.py
"""

import logging
import math
from typing import Dict, List, Tuple

import networkx as nx



def eval_dispatch_solution(solution,
                           order_info_dict,
                           car_info_detail,
                           static_data,
                           param_dict):
    """
    评估调度解的性能

    Args:
        solution: 解，字典形式 {car_idx: [order_idx1, order_idx2, ...]}
        order_info_dict: 订单详细信息列表
        car_info_detail: 车辆详细信息列表
        static_data: 静态数据
        param_dict: 参数字典

    Returns:
        dict: 评估结果，包含目标组成、成本组成、约束满足情况
    """
    logging.info("=" * 80)
    logging.info("[EvalDispatchSolution] Starting solution evaluation")
    logging.info("=" * 80)

    constant_dict = _build_constant_dict(order_info_dict, car_info_detail, static_data, param_dict)
    context = _prepare_eval_context(constant_dict, param_dict)
    normalized_solution = _normalize_solution(solution, context["car_num"])

    # 目标评估
    logging.info("-" * 80)
    logging.info("[EvalDispatchSolution] Objective Evaluation:")
    logging.info("-" * 80)
    obj_detail = _calculate_objective_detail(normalized_solution, context)
    _log_objective_detail(obj_detail)

    # 成本评估
    logging.info("-" * 80)
    logging.info("[EvalDispatchSolution] Cost Evaluation (by Car):")
    logging.info("-" * 80)
    cost_detail = _calculate_cost_detail(normalized_solution, context)
    _log_cost_detail(cost_detail)

    # 约束评估
    logging.info("-" * 80)
    logging.info("[EvalDispatchSolution] Constraint Evaluation:")
    logging.info("-" * 80)
    constraint_status, is_valid, violation_list = _evaluate_constraints(normalized_solution, context)
    _log_constraint_status(constraint_status)

    logging.info("=" * 80)
    logging.info("[EvalDispatchSolution] Evaluation completed")
    logging.info("=" * 80)

    return {
        "total_objective": obj_detail["total_objective"],
        "objective_detail": obj_detail,
        "cost_detail": cost_detail,
        "constraint_status": constraint_status,
        "is_valid": is_valid,
        "violation_count": len(violation_list),
        "violations": violation_list
    }


def _build_constant_dict(order_info_dict, car_info_detail, static_data, param_dict):
    """构建常量字典"""
    node_data = static_data["nodeData"]
    graph_node_list, graph_edge_list, node_index2center = get_node_graph(node_data, neighbor_limited=0.0075)
    node_dict = {x["nodeName"]: x["locationTuple"] for x in node_data}
    node_list = list(node_dict.keys())
    node_name2index = {node_list[i]: i for i in range(0, len(node_list))}
    G = nx.Graph()
    G.add_nodes_from(graph_node_list)
    for edge in graph_edge_list:
        u, v = edge[0], edge[1]
        u_center, v_center = node_index2center[u], node_index2center[v]
        distance = math.sqrt((u_center[0] - v_center[0]) ** 2 + (u_center[1] - v_center[1]) ** 2)
        G.add_edge(u, v, weight=distance)

    graph_data_dict = {
        "G": G,
        "node_index2center": node_index2center,
        "node_name2index": node_name2index,
        "isolated_node": []
    }

    constant_builder = ConstantBuild(
        static_data,
        car_info_detail,
        order_info_dict,
        graph_data_dict,
        param_dict.get("distance_2_surrounding_order", 0)
    )
    constant_dict = constant_builder.main_process()
    return constant_dict


def _prepare_eval_context(constant_dict, param_dict):
    """准备评估所需的上下文数据"""
    car_info = constant_dict["car_info_dict"]
    order_info = constant_dict["order_info_dict"]
    node_info = constant_dict["node_info_dict"]
    mutex_info = constant_dict["mutex_info_dict"]
    price_info = constant_dict["price_info_dict"]

    car_num = car_info["car_num"]
    order_num = order_info["order_num"]

    # 构建映射
    order_to_node = {order_idx: node_idx for order_idx, node_idx in node_info["order2node"]}
    order_to_community = {order_idx: comm_idx for order_idx, comm_idx in node_info["order2community"]}

    order_not_same_group_set = set()
    for o1, o2 in node_info["order_not_in_same_group"]:
        order_not_same_group_set.add((o1, o2))
        order_not_same_group_set.add((o2, o1))

    car_order_mutex_set = set(mutex_info["order_car_mutex"])

    order_order_mutex_set = set()
    for o1, o2 in mutex_info["order_order_mutex"]:
        order_order_mutex_set.add((o1, o2))
        order_order_mutex_set.add((o2, o1))

    weights = {
        "release": param_dict.get("obj_weight_release", 1e5),
        "price": param_dict.get("obj_weight_price", 1),
        "diameter": param_dict.get("obj_weight_diameter", 200),
        "node": param_dict.get("obj_weight_node", 100),
        "community": param_dict.get("obj_weight_community", 50),
        "nodegroup": param_dict.get("obj_weight_nodegroup", 50)
    }

    context = {
        "car_num": car_num,
        "order_num": order_num,
        "car_info": car_info,
        "order_info": order_info,
        "node_info": node_info,
        "mutex_info": mutex_info,
        "price_info": price_info,
        "order_to_node": order_to_node,
        "order_to_community": order_to_community,
        "order_not_same_group_set": order_not_same_group_set,
        "car_order_mutex_set": car_order_mutex_set,
        "order_order_mutex_set": order_order_mutex_set,
        "weights": weights,
        "order_angle_limited": param_dict.get("order_angle_limited", 45),
        "constraint_penalty_weight": param_dict.get("constraint_penalty_weight", 1e6)
    }
    return context


def _normalize_solution(solution, car_num):
    """标准化解的格式，确保每个车辆索引存在"""
    normalized = {car_idx: [] for car_idx in range(car_num)}
    if not isinstance(solution, dict):
        return normalized
    for key, orders in solution.items():
        try:
            car_idx = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= car_idx < car_num:
            normalized[car_idx] = list(orders)
    return normalized


def _calculate_objective_detail(solution, ctx):
    """计算目标组成"""
    order_info = ctx["order_info"]
    price_info = ctx["price_info"]
    car_info = ctx["car_info"]
    mutex_info = ctx["mutex_info"]
    weights = ctx["weights"]
    order_to_node = ctx["order_to_node"]
    order_to_community = ctx["order_to_community"]
    order_not_same_group_set = ctx["order_not_same_group_set"]

    order_weight = order_info["order_weight"]
    order_can_release = order_info["order_can_release"]
    order_distance_dict = order_info["order_distance_dict"]
    order_need_node_price = order_info["order_need_node_price"]
    order_distance_2_storage = order_info["order_distance_2_storage"]

    order_num = ctx["order_num"]

    # 目标1：甩单
    assigned_orders = set()
    for orders in solution.values():
        assigned_orders.update(orders)
    unassigned_orders = set(range(order_num)) - assigned_orders
    release_count = sum(1 for o in unassigned_orders if order_can_release[o] == 0)
    obj_release_raw = release_count
    obj_release_weighted = obj_release_raw * weights["release"]

    # 目标2：价格
    car_overload_limited = car_info["car_overload_limited"]
    car_overload_price = price_info["car_overload_price"]
    car_node_price = price_info["car_node_price"]
    car_start_price = price_info["car_start_price"]

    obj_price_raw = 0.0
    for car_idx, orders in solution.items():
        if not orders:
            continue
        total_weight = sum(order_weight[o] for o in orders)
        overload_threshold = car_overload_limited[car_idx]
        overload_fee = max(0, (total_weight - overload_threshold) * car_overload_price[car_idx])
        node_count = sum(order_need_node_price[o] for o in orders)
        node_fee = max(0, (node_count - 1)) * car_node_price[car_idx]
        max_distance_order = max(orders, key=lambda o: order_distance_2_storage[o])
        start_fee = car_start_price.get((max_distance_order, car_idx), 0)
        obj_price_raw += overload_fee + node_fee + start_fee
    obj_price_weighted = obj_price_raw * weights["price"]

    # 目标3：直径
    obj_diameter_raw = 0.0
    for orders in solution.values():
        if len(orders) <= 1:
            continue
        max_distance = 0.0
        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1:]:
                dist = order_distance_dict.get((o1, o2), order_distance_dict.get((o2, o1), 0))
                max_distance = max(max_distance, dist)
        obj_diameter_raw += max_distance
    obj_diameter_weighted = obj_diameter_raw * weights["diameter"]

    # 目标4：节点
    used_nodes = {order_to_node[o] for orders in solution.values() for o in orders if o in order_to_node}
    obj_node_raw = len(used_nodes)
    obj_node_weighted = obj_node_raw * weights["node"]

    # 目标5：小区
    used_communities = {order_to_community[o] for orders in solution.values() for o in orders if o in order_to_community}
    obj_community_raw = len(used_communities)
    obj_community_weighted = obj_community_raw * weights["community"]

    # 目标6：大区惩罚
    obj_nodegroup_raw = 0
    for orders in solution.values():
        if len(orders) <= 1:
            continue
        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1:]:
                if (o1, o2) in order_not_same_group_set:
                    obj_nodegroup_raw += 1
    obj_nodegroup_weighted = obj_nodegroup_raw * weights["nodegroup"]

    # 约束惩罚
    constraint_penalty_detail = _calculate_constraint_penalty(solution, ctx)

    total_obj = (obj_release_weighted + obj_price_weighted + obj_diameter_weighted +
                 obj_node_weighted + obj_community_weighted + obj_nodegroup_weighted +
                 constraint_penalty_detail["weighted_value"])

    return {
        "obj_release": {
            "raw_value": obj_release_raw,
            "weight": weights["release"],
            "weighted_value": obj_release_weighted
        },
        "obj_price": {
            "raw_value": obj_price_raw,
            "weight": weights["price"],
            "weighted_value": obj_price_weighted
        },
        "obj_diameter": {
            "raw_value": obj_diameter_raw,
            "weight": weights["diameter"],
            "weighted_value": obj_diameter_weighted
        },
        "obj_node": {
            "raw_value": obj_node_raw,
            "weight": weights["node"],
            "weighted_value": obj_node_weighted
        },
        "obj_community": {
            "raw_value": obj_community_raw,
            "weight": weights["community"],
            "weighted_value": obj_community_weighted
        },
        "obj_nodegroup": {
            "raw_value": obj_nodegroup_raw,
            "weight": weights["nodegroup"],
            "weighted_value": obj_nodegroup_weighted
        },
        "obj_constraint_penalty": constraint_penalty_detail,
        "total_objective": total_obj
    }


def _calculate_constraint_penalty(solution, ctx):
    """计算约束违反惩罚（与求解器保持一致）"""
    order_info = ctx["order_info"]
    car_info = ctx["car_info"]
    mutex_info = ctx["mutex_info"]

    order_weight = order_info["order_weight"]
    type_cz = order_info["type_cz"]
    type_wy = order_info["type_wy"]
    order_travel_time = order_info["order_travel_time"]
    order_work_time = order_info["order_work_time"]

    car_order_mutex_set = ctx["car_order_mutex_set"]
    order_order_mutex_set = ctx["order_order_mutex_set"]
    order2order_node_distance = mutex_info["order2order_node_distance"]
    order2order_angle_diff = mutex_info["order2order_angle_diff"]

    penalty_weight = ctx["constraint_penalty_weight"]
    order_angle_limited = ctx["order_angle_limited"]

    penalty_raw = 0.0
    penalty_weighted = 0.0

    # 约束1：订单唯一
    order_assigned = {}
    for car_idx, orders in solution.items():
        for order_idx in orders:
            if order_idx in order_assigned:
                penalty_raw += 1.0
                penalty_weighted += penalty_weight
            order_assigned[order_idx] = car_idx

    for car_idx, orders in solution.items():
        if not orders:
            continue

        total_weight = sum(order_weight[o] for o in orders)
        has_cz = any(type_cz[o] for o in orders)
        has_wy = any(type_wy[o] for o in orders)

        if has_cz and has_wy:
            load_lb = car_info["car_load_mix_lb"][car_idx]
            load_ub = car_info["car_load_mix_ub"][car_idx]
        elif has_cz:
            load_lb = car_info["car_load_cz_lb"][car_idx]
            load_ub = car_info["car_load_cz_ub"][car_idx]
        elif has_wy:
            load_lb = car_info["car_load_wy_lb"][car_idx]
            load_ub = car_info["car_load_wy_ub"][car_idx]
        else:
            load_lb = car_info["car_load_mix_lb"][car_idx]
            load_ub = car_info["car_load_mix_ub"][car_idx]

        if total_weight < load_lb:
            ratio = (load_lb - total_weight) / max(load_lb, 1)
            penalty_raw += ratio
            penalty_weighted += penalty_weight * ratio
        elif total_weight > load_ub:
            ratio = (total_weight - load_ub) / max(load_ub, 1)
            penalty_raw += ratio
            penalty_weighted += penalty_weight * ratio

        for order_idx in orders:
            if (order_idx, car_idx) in car_order_mutex_set:
                penalty_raw += 1.0
                penalty_weighted += penalty_weight

        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1:]:
                if (o1, o2) in order_order_mutex_set:
                    penalty_raw += 1.0
                    penalty_weighted += penalty_weight

                node_dist = order2order_node_distance.get((o1, o2),
                                                          order2order_node_distance.get((o2, o1), 0))
                car_span_node_limited = car_info["car_span_node_limited"][car_idx]
                if node_dist > car_span_node_limited:
                    ratio = (node_dist - car_span_node_limited) / max(car_span_node_limited, 1)
                    penalty_raw += ratio
                    penalty_weighted += penalty_weight * ratio

                angle_diff = order2order_angle_diff.get((o1, o2),
                                                        order2order_angle_diff.get((o2, o1), 0))
                if angle_diff > order_angle_limited:
                    ratio = (angle_diff - order_angle_limited) / max(order_angle_limited, 1)
                    penalty_raw += ratio
                    penalty_weighted += penalty_weight * ratio

        max_order = car_info["car_max_order"][car_idx]
        if len(orders) > max_order:
            ratio = (len(orders) - max_order) / max(max_order, 1)
            penalty_raw += ratio
            penalty_weighted += penalty_weight * ratio

        total_work_time = sum(order_work_time[o] for o in orders)
        max_travel_time = max([order_travel_time[car_idx][o] for o in orders], default=0)
        max_server_time = car_info["car_max_server_time"][car_idx]
        if total_work_time + max_travel_time > max_server_time:
            ratio = (total_work_time + max_travel_time - max_server_time) / max(max_server_time, 1)
            penalty_raw += ratio
            penalty_weighted += penalty_weight * ratio

    return {
        "raw_value": penalty_raw,
        "weight": penalty_weight,
        "weighted_value": penalty_weighted
    }


def _calculate_cost_detail(solution, ctx):
    """计算成本组成"""
    order_info = ctx["order_info"]
    car_info = ctx["car_info"]
    price_info = ctx["price_info"]

    order_weight = order_info["order_weight"]
    order_need_node_price = order_info["order_need_node_price"]
    order_distance_2_storage = order_info["order_distance_2_storage"]

    car_overload_limited = car_info["car_overload_limited"]
    car_overload_price = price_info["car_overload_price"]
    car_node_price = price_info["car_node_price"]
    car_start_price = price_info["car_start_price"]

    car_cost_detail = {}
    total_cost = 0.0

    for car_idx, orders in solution.items():
        if not orders:
            continue

        total_weight = sum(order_weight[o] for o in orders)
        overload_threshold = car_overload_limited[car_idx]
        overload_fee = max(0, (total_weight - overload_threshold) * car_overload_price[car_idx])

        node_count = sum(order_need_node_price[o] for o in orders)
        node_fee = max(0, (node_count - 1)) * car_node_price[car_idx]

        max_distance_order = max(orders, key=lambda o: order_distance_2_storage[o])
        start_fee = car_start_price.get((max_distance_order, car_idx), 0)

        car_total_cost = overload_fee + node_fee + start_fee
        total_cost += car_total_cost

        car_cost_detail[car_idx] = {
            "order_count": len(orders),
            "total_weight": total_weight,
            "overload_threshold": overload_threshold,
            "overload_fee": overload_fee,
            "node_count": node_count,
            "node_fee": node_fee,
            "start_fee": start_fee,
            "total_cost": car_total_cost
        }

    return {
        "total_cost": total_cost,
        "car_cost_detail": car_cost_detail
    }


def _evaluate_constraints(solution, ctx):
    """详细检查约束满足情况"""
    order_info = ctx["order_info"]
    car_info = ctx["car_info"]
    mutex_info = ctx["mutex_info"]

    order_weight = order_info["order_weight"]
    type_cz = order_info["type_cz"]
    type_wy = order_info["type_wy"]
    order_work_time = order_info["order_work_time"]
    order_travel_time = order_info["order_travel_time"]

    car_order_mutex_set = ctx["car_order_mutex_set"]
    order_order_mutex_set = ctx["order_order_mutex_set"]
    order2order_node_distance = mutex_info["order2order_node_distance"]
    order2order_angle_diff = mutex_info["order2order_angle_diff"]

    order_angle_limited = ctx["order_angle_limited"]

    constraint_status = {}
    violation_list = []

    # 约束1：订单唯一
    order_assigned = {}
    constraint1_violations = []
    for car_idx, orders in solution.items():
        for order_idx in orders:
            if order_idx in order_assigned:
                msg = f"订单{order_idx}被车辆{order_assigned[order_idx]}和车辆{car_idx}同时指派"
                constraint1_violations.append(msg)
                violation_list.append(msg)
            order_assigned[order_idx] = car_idx
    constraint_status["constraint1_order_uniqueness"] = {
        "satisfied": len(constraint1_violations) == 0,
        "violations": constraint1_violations
    }

    # 约束2-8
    constraint2_violations: List[str] = []
    constraint3_violations: List[str] = []
    constraint4_violations: List[str] = []
    constraint5_violations: List[str] = []
    constraint6_violations: List[str] = []
    constraint7_violations: List[str] = []
    constraint8_violations: List[str] = []

    for car_idx, orders in solution.items():
        if not orders:
            continue

        # 约束2：车辆满载
        total_weight = sum(order_weight[o] for o in orders)
        has_cz = any(type_cz[o] for o in orders)
        has_wy = any(type_wy[o] for o in orders)

        if has_cz and has_wy:
            load_lb = car_info["car_load_mix_lb"][car_idx]
            load_ub = car_info["car_load_mix_ub"][car_idx]
        elif has_cz:
            load_lb = car_info["car_load_cz_lb"][car_idx]
            load_ub = car_info["car_load_cz_ub"][car_idx]
        elif has_wy:
            load_lb = car_info["car_load_wy_lb"][car_idx]
            load_ub = car_info["car_load_wy_ub"][car_idx]
        else:
            load_lb = car_info["car_load_mix_lb"][car_idx]
            load_ub = car_info["car_load_mix_ub"][car_idx]

        if total_weight < load_lb:
            msg = f"车辆{car_idx}重量{total_weight:.2f}低于下限{load_lb:.2f}"
            constraint2_violations.append(msg)
            violation_list.append(msg)
        elif total_weight > load_ub:
            msg = f"车辆{car_idx}重量{total_weight:.2f}超过上限{load_ub:.2f}"
            constraint2_violations.append(msg)
            violation_list.append(msg)

        # 约束3：订单车辆互斥
        for order_idx in orders:
            if (order_idx, car_idx) in car_order_mutex_set:
                msg = f"订单{order_idx}与车辆{car_idx}互斥"
                constraint3_violations.append(msg)
                violation_list.append(msg)

        # 约束4：订单订单互斥
        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1:]:
                if (o1, o2) in order_order_mutex_set:
                    msg = f"车辆{car_idx}中订单{o1}与订单{o2}互斥"
                    constraint4_violations.append(msg)
                    violation_list.append(msg)

        # 约束5：跨节点限制
        car_span_node_limited = car_info["car_span_node_limited"][car_idx]
        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1:]:
                node_dist = order2order_node_distance.get((o1, o2),
                                                          order2order_node_distance.get((o2, o1), 0))
                if node_dist > car_span_node_limited:
                    msg = f"车辆{car_idx}中订单{o1}与订单{o2}节点距离{node_dist}超过限制{car_span_node_limited}"
                    constraint5_violations.append(msg)
                    violation_list.append(msg)

        # 约束6：订单角度限制
        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1:]:
                angle_diff = order2order_angle_diff.get((o1, o2),
                                                        order2order_angle_diff.get((o2, o1), 0))
                if angle_diff > order_angle_limited:
                    msg = f"车辆{car_idx}中订单{o1}与订单{o2}角度差{angle_diff:.2f}超过限制{order_angle_limited}"
                    constraint6_violations.append(msg)
                    violation_list.append(msg)

        # 约束7：订单数限制
        max_order = car_info["car_max_order"][car_idx]
        if len(orders) > max_order:
            msg = f"车辆{car_idx}订单数{len(orders)}超过最大限制{max_order}"
            constraint7_violations.append(msg)
            violation_list.append(msg)

        # 约束8：最长服务时间
        total_work_time = sum(order_work_time[o] for o in orders)
        max_travel_time = max([order_travel_time[car_idx][o] for o in orders], default=0)
        max_server_time = car_info["car_max_server_time"][car_idx]
        if total_work_time + max_travel_time > max_server_time:
            msg = (f"车辆{car_idx}总工作时间{total_work_time:.2f}+最大行驶时间{max_travel_time:.2f}超过"
                   f"限制{max_server_time:.2f}")
            constraint8_violations.append(msg)
            violation_list.append(msg)

    constraint_status["constraint2_load_range"] = {
        "satisfied": len(constraint2_violations) == 0,
        "violations": constraint2_violations
    }
    constraint_status["constraint3_order_car_mutex"] = {
        "satisfied": len(constraint3_violations) == 0,
        "violations": constraint3_violations
    }
    constraint_status["constraint4_order_order_mutex"] = {
        "satisfied": len(constraint4_violations) == 0,
        "violations": constraint4_violations
    }
    constraint_status["constraint5_span_node"] = {
        "satisfied": len(constraint5_violations) == 0,
        "violations": constraint5_violations
    }
    constraint_status["constraint6_angle_limit"] = {
        "satisfied": len(constraint6_violations) == 0,
        "violations": constraint6_violations
    }
    constraint_status["constraint7_max_order"] = {
        "satisfied": len(constraint7_violations) == 0,
        "violations": constraint7_violations
    }
    constraint_status["constraint8_max_service_time"] = {
        "satisfied": len(constraint8_violations) == 0,
        "violations": constraint8_violations
    }

    is_valid = all(status["satisfied"] for status in constraint_status.values())
    return constraint_status, is_valid, violation_list


def _log_objective_detail(obj_detail):
    """日志输出目标组成"""
    logging.info(f"  甩单目标 (Release):     原始值 = {obj_detail['obj_release']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_release']['weight']:.2e}, "
                 f"加权值 = {obj_detail['obj_release']['weighted_value']:.2f}")
    logging.info(f"  价格目标 (Price):        原始值 = {obj_detail['obj_price']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_price']['weight']:.2f}, "
                 f"加权值 = {obj_detail['obj_price']['weighted_value']:.2f}")
    logging.info(f"  直径目标 (Diameter):     原始值 = {obj_detail['obj_diameter']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_diameter']['weight']:.2f}, "
                 f"加权值 = {obj_detail['obj_diameter']['weighted_value']:.2f}")
    logging.info(f"  节点目标 (Node):         原始值 = {obj_detail['obj_node']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_node']['weight']:.2f}, "
                 f"加权值 = {obj_detail['obj_node']['weighted_value']:.2f}")
    logging.info(f"  小区目标 (Community):    原始值 = {obj_detail['obj_community']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_community']['weight']:.2f}, "
                 f"加权值 = {obj_detail['obj_community']['weighted_value']:.2f}")
    logging.info(f"  大区惩罚 (NodeGroup):    原始值 = {obj_detail['obj_nodegroup']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_nodegroup']['weight']:.2f}, "
                 f"加权值 = {obj_detail['obj_nodegroup']['weighted_value']:.2f}")
    logging.info(f"  约束惩罚 (Constraint):  原始值 = {obj_detail['obj_constraint_penalty']['raw_value']:.2f}, "
                 f"权重 = {obj_detail['obj_constraint_penalty']['weight']:.2e}, "
                 f"加权值 = {obj_detail['obj_constraint_penalty']['weighted_value']:.2f}")
    logging.info(f"  总目标值 (Total):        {obj_detail['total_objective']:.2f}")


def _log_cost_detail(cost_detail):
    """日志输出成本组成"""
    car_cost_detail = cost_detail["car_cost_detail"]
    for car_idx, detail in car_cost_detail.items():
        logging.info(f"  车辆 {car_idx}:")
        logging.info(f"    订单数: {detail['order_count']}")
        logging.info(f"    总重量: {detail['total_weight']:.2f} kg, 超载阈值: {detail['overload_threshold']:.2f} kg")
        logging.info(f"    超载费: {detail['overload_fee']:.2f}")
        logging.info(f"    节点数: {detail['node_count']}, 节点费: {detail['node_fee']:.2f}")
        logging.info(f"    起步费: {detail['start_fee']:.2f}")
        logging.info(f"    总成本: {detail['total_cost']:.2f}")
    logging.info(f"  所有车辆总成本: {cost_detail['total_cost']:.2f}")


def _log_constraint_status(constraint_status):
    """日志输出约束状态"""
    constraint_desc_map = {
        "constraint1_order_uniqueness": "订单唯一性",
        "constraint2_load_range": "车辆满载条件",
        "constraint3_order_car_mutex": "订单车辆互斥",
        "constraint4_order_order_mutex": "订单订单互斥",
        "constraint5_span_node": "跨节点限制",
        "constraint6_angle_limit": "订单角度限制",
        "constraint7_max_order": "订单数限制",
        "constraint8_max_service_time": "最长服务时间限制"
    }

    for name, status in constraint_status.items():
        desc = constraint_desc_map.get(name, name)
        if status["satisfied"]:
            logging.info(f"  {desc}: ✓ 满足")
        else:
            logging.warning(f"  {desc}: ✗ 违反 ({len(status['violations'])} 处)")
            for violation in status["violations"][:5]:
                logging.warning(f"    - {violation}")
            if len(status["violations"]) > 5:
                logging.warning(f"    ... 还有 {len(status['violations']) - 5} 处违反")

# ===================== 以下为原 evaluator.py =====================
def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "lower_is_better")


def _load_lib():
    """加载业务侧评估库(私有, 不随镜像下发给 agent)。"""
    pass  # 已内联: from result_evaluate import ...
    return eval_dispatch_solution


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        ref_value, direction = load_baseline()
        m["reference_value"] = ref_value

        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        with open(plan, encoding="utf-8") as f:
            sol = json.load(f)
        with open(os.path.join(data_dir, DATA_FILE), encoding="utf-8") as f:
            data = json.load(f)

        orders = data["order_info_detail"]
        cars = data["car_info_detail"]
        n_order, n_car = len(orders), len(cars)

        raw = sol.get("assignment")
        if not isinstance(raw, dict) or not raw:
            m["error_info"] = {"hard_violations": ["assignment 缺失或为空"]}
            return m

        # 归一成 {int 车辆下标: [int 订单下标]}
        assignment: Dict[int, list] = {}
        errors = []
        for k, v in raw.items():
            try:
                ci = int(k)
            except (TypeError, ValueError):
                errors.append(f"车辆键 {k!r} 不是整数下标")
                break
            if not 0 <= ci < n_car:
                errors.append(f"车辆下标 {ci} 越界(合法 0..{n_car - 1})")
                break
            if not isinstance(v, list):
                errors.append(f"车辆 {ci} 的订单列表不是数组")
                break
            lst = []
            for o in v:
                try:
                    oi = int(o)
                except (TypeError, ValueError):
                    errors.append(f"车辆 {ci} 含非整数订单 {o!r}")
                    break
                if not 0 <= oi < n_order:
                    errors.append(f"订单下标 {oi} 越界(合法 0..{n_order - 1})")
                    break
                lst.append(oi)
            if errors:
                break
            assignment[ci] = lst

        if errors:
            m["error_info"] = {"hard_violations": errors[:8]}
            return m

        # 覆盖唯一性(评估库也查, 这里先给清晰报错)
        flat = [o for v in assignment.values() for o in v]
        miss = sorted(set(range(n_order)) - set(flat))
        dup = sorted({o for o in flat if flat.count(o) > 1})
        if miss:
            errors.append(f"{len(miss)} 个订单未被分配, 例 {miss[:6]}")
        if dup:
            errors.append(f"{len(dup)} 个订单被分配给多辆车, 例 {dup[:6]}")
        if errors:
            m["error_info"] = {"hard_violations": errors[:8]}
            return m

        # 补齐未出现的车辆为空列表
        for ci in range(n_car):
            assignment.setdefault(ci, [])

        eval_fn = _load_lib()
        detail = eval_fn(assignment, orders, cars, data["static_data"], dict(PARAM_DICT))

        obj = float(detail["total_objective"])
        cs = detail.get("constraint_status", {})
        viols = detail.get("violations", []) or []
        is_valid = bool(detail.get("is_valid"))

        m["n_used_vehicles"] = sum(1 for v in assignment.values() if v)
        m["total_cost"] = round(float(detail["cost_detail"]["total_cost"]), 3)
        m["violation_count"] = int(detail.get("violation_count", len(viols)))
        m["player_objective"] = round(obj, 3)

        # 业务口径: 8 类约束的违反以 1e6 权重计入 total_objective(软惩罚),
        # 不直接判方案无效。消除违反是降低目标值的主要来源。
        # 仅"订单未全覆盖/重复分配"这类结构性问题在前面已判无效。
        m["is_feasible"] = is_valid
        if viols:
            m["error_info"] = {
                "constraint_violations": [str(v) for v in viols[:8]],
                "total": m["violation_count"],
                "unsatisfied_groups": [k for k, v in cs.items()
                                       if isinstance(v, dict) and not v.get("satisfied")],
                "_note": "违反已按 1e6/处 计入 player_objective，未判方案无效",
            }

        m["validity_score"] = 1.0
        if direction == "lower_is_better":
            quality = ref_value / obj if obj > 0 else 0.0
        else:
            quality = obj / ref_value if ref_value > 0 else 0.0
        m["quality_score"] = round((quality), 6)
        m["overall_score"] = m["quality_score"]
        return m

    except Exception as e:
        import traceback
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-900:]}
        return m


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
