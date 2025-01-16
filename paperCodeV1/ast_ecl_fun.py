#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:14:50 2023

 计算指定小天体指定时刻地球和太阳的黄道坐标

@author: superman
"""

import spiceypy as spice
import datetime as dt
import numpy as np
import math
from astropy.time import Time

# 返回太阳和地球相对observer的黄道坐标, 默认单位是AU
# 如果以太阳和航天器为参考，pos2需要修改
def ast_ecl_position(time, observer, target1='SUN', target2='EARTH', output_km=False):

    # 根据时间格式计算et_time
    if isinstance(time, str) and 'T' in time:
        et_time = spice.utc2et(time)
    elif isinstance(time, str) and time.startswith('JD'):
        et_time = spice.str2et(time)
    elif len(time) == 6:
        utc_time = dt.datetime(*time).strftime('%Y-%m-%dT%H:%M:%S')
        et_time = spice.utc2et(utc_time)

    # 计算地球和太阳的黄道坐标
    pos1, _ = spice.spkpos(target1, et_time, 'ECLIPJ2000', 'NONE', observer)
    pos2, _ = spice.spkpos(target2, et_time, 'ECLIPJ2000', 'NONE', observer)

    # 将位置坐标从km转换为au
    pos1_au = spice.convrt(pos1, 'KM', 'AU')
    pos2_au = spice.convrt(pos2, 'KM', 'AU')

    # output unit position in AU or km
    if output_km:
        return pos1, pos2
    else:
        return pos1_au, pos2_au


# 将UTC时间转化为儒略日
def utc_to_jd(utc_time):
    t = Time(utc_time, format='datetime')
    jd = t.jd
    return jd

# 将儒略日转化为UTC时间
def jd_to_utc(jd):
    t = Time(jd, format='jd')
    utc_time = t.datetime
    return utc_time

# 生成时间的数组
def generate_array(start_jd, tdur=1.0/24, dt=1.0/1440):
    array = []
    array.append(start_jd)
    jd = start_jd
    while jd <= start_jd+tdur:
        jd += dt
        array.append(jd)
    return array

def count_element(row):
    if isinstance(row, str):
        # 如果当前行是字符串，则将其按空格拆分，并计算元素个数
        count = len(row.split())
    elif isinstance(row, list):
        # 如果当前行是列表，则直接计算元素个数
        count = len(row)
    else:
        count = 1000
        # 如果当前行不是字符串也不是列表，则跳过
    return count

# exposure to cadence, n=bin size
def lc_bin(arr, n):
    """
    对一个二维数组（列表）arr进行处理，对连续的8列的行进行每n个元素求平均值操作。

    :param arr: 一个二维列表，其中的每个子列表代表一行，可能有8列或者不是8列。
    :param n: 用于计算平均值的连续元素数目。
    :return: 返回处理过的二维列表。
    """
    result = []
    i = 0
    n = int(n)
    while i < len(arr):
        row = arr[i]
        if len(row) != 8:  # 如果当前行不是8列，则原样添加到结果列表中，并继续处理下一行
            result.append(row)
            i += 1
        else:  # 如果当前行是8列，则开始处理连续的8列行
            temp = [[] for _ in range(8)]  # 初始化临时列表，用于存储各列的元素
            count = 0  # 计算连续的8列行的数量
            while i < len(arr) and len(arr[i]) == 8:
                for j in range(8):
                    temp[j].append(arr[i][j])  # 将当前行的元素添加到对应的临时列表中
                count += 1
                i += 1
                
                if count % n == 0:  # 如果连续的8列行数量达到n，则计算平均值并添加到结果列表中
                    avg_row = [np.average(col[:int(n)]) for col in temp]  # 计算各列的平均值
                    result.append(avg_row)
                    for col in temp:  # 清空已处理的元素
                        del col[:int(n)]

            # 如果剩余的连续8列行数量不足n，则计算平均值并添加到结果列表中
            if 0 < len(temp[0]) < n:
                avg_row = [sum(col) / len(col) for col in temp]  # 计算各列的平均值
                result.append(avg_row)

    i = 0
    while i < len(result):
        row = result[i]
        row_nele = count_element(row)
        if row_nele == 1 or row_nele == 8:
            i += 1
        elif row_nele == 2:
            j = i + 1
            while j < len(result) and count_element(result[j]) == 8:
                j += 1
            if j == i + 1:
                # 如果下面没有列数不为8的行，则第一列元素不变
                pass
            else:
                # 如果下面有列数不为8的行，则将第一列元素替换为下面所有列数为8的行之间的行数
                count = j - i - 1
                elements = result[i].split()
                elements[0] = count
                result[i] = ' '.join(str(e) for e in elements) + '\n'  # 将修改后的元素重新拼接为字符串，并加上结尾的'\n'
            i = j
        else:
            i += 1

    return result

def decompose_distance_along_pos(pos, d):
    # 计算向量pos的长度
    l = np.linalg.norm(pos)

    # 计算单位向量u
    u = pos / l

    # 计算距离d在pos方向上的分量
    d_pos = d * u

    # # 计算d在三个方向上的分量
    # d_x = d_pos[0]
    # d_y = d_pos[1]
    # d_z = d_pos[2]

    return d_pos


def decompose_distance(pos1, d, alpha):

    """
    计算d在未知坐标系下的三个分量。

    参数:
        pos1 (list or ndarray): 一个包含三个元素的列表或者NumPy数组，表示原点为起点的向量pos1在xyz三个方向上的分量。
        alpha (float): d和pos1的夹角，单位为弧度。
        d (float): 给定距离，表示圆上各点到原点的距离。

    返回:
        d_components (ndarray): 一个包含三个元素的NumPy数组，表示d在未知坐标系下的三个分量。
    """
    # 将角度值转换为弧度
    alpha = math.radians(alpha)
    
    # 将输入转换为NumPy数组
    pos1 = np.array(pos1)
    
    # 计算pos1的单位向量
    pos1_unit_vector = pos1 / np.linalg.norm(pos1)

    # 计算d在pos1法向量上的投影长度
    d_projection_length = d * np.cos(alpha)

    # 计算d在pos1法向量上的投影分量
    d_projection = pos1_unit_vector * d_projection_length

    # 计算d在与pos1垂直的平面上的长度
    d_perpendicular_length = d * np.sin(alpha)

    # 计算pos1的法向量, 
    pos1_normal_vector = np.cross(pos1_unit_vector, np.array([0, 0, 1]))

    # 如果法向量为零向量，说明pos1与z轴平行，此时选取任意一个垂直于pos1的向量作为法向量
    if np.linalg.norm(pos1_normal_vector) == 0:
        pos1_normal_vector = np.array([1, 0, 0])

    # 计算法向量的单位向量
    pos1_normal_unit_vector = pos1_normal_vector / np.linalg.norm(pos1_normal_vector)

    # 计算d在与pos1垂直的平面上的分量
    d_perpendicular = pos1_normal_unit_vector * d_perpendicular_length

    # 计算d在坐标系下的三个分量
    d_components = d_projection + d_perpendicular

    return d_components

def angle_between_vectors(a, b):
    dot_product = sum([a[i] * b[i] for i in range(len(a))])
    a_norm = math.sqrt(sum([a[i]**2 for i in range(len(a))]))
    b_norm = math.sqrt(sum([b[i]**2 for i in range(len(b))]))
    cos_theta = dot_product / (a_norm * b_norm)
    theta = math.acos(cos_theta)
    return math.degrees(theta)

# 显式地定义需要导出的函数和变量,以便使用from functions import *的方式导入模块
#__all__ = ["ast_ecl_position","utc_to_jd","jd_to_utc","generate_array"]