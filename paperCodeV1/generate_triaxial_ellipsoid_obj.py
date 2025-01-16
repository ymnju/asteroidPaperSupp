#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:38:29 2023

程序的主要功能是生成一个三轴椭球的顶点坐标数据，以及椭球的面数据，并将椭球旋转到指定的位置，
最后将椭球的顶点和面存成obj文件 和 Damit能识别的文件。

input parametrers:
    三轴椭球尺寸 (a, b, c) 
    自转方向 (lambda_deg,beta_deg) 
    椭球在u和v方向的分辨率 (u_resolution, u_resolution)
output files:
    obj 文件: ellipsoid_YYYYMMDD.obj
    Damit需要的输入文件: ellipsoid_YYYYMMDD

@author: superman
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from astropy.time import Time
import ast_ecl_fun as af
import subprocess
import math

# 用OCTANT TRIANGULATION方法生成三轴椭球的顶点坐标数据,
# 令abc与xyz分别重合, 三轴椭球在每个象限内被分割成N个水平行
def generate_vertices_oct(a, b, c, n):
    vertices = []
    vertices.append(np.array([0,0,c]))
    for i in range(n+1):
        phi = math.pi/2 - i*math.pi/(2*n)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for j in range(4*i):
            if i == 0:
                theta = 0
            else:
                theta = j*math.pi/(2*i)
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            x = a*cos_phi*cos_theta
            y = b*cos_phi*sin_theta
            z = c*sin_phi
            vertices.append(np.array([x,y,z]))

    vertices.append(np.array([0,0,-c]))
    for i in range(n):
        phi = -math.pi/2 + i*math.pi/(2*n)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        print(phi*57.2957795)
        for j in range(4*i):
            if i == 0:
                theta = 0
            else:
                theta = j*math.pi/(2*i)
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            x = a*cos_phi*cos_theta
            y = b*cos_phi*sin_theta
            z = c*sin_phi
            vertices.append(np.array([x,y,z]))
            print('  ',theta*57.2957795)
    return vertices

# 生成顶点对应的facets
def generate_facets_oct(n):
    facets = []
    for i in range(n):
        flag1 = flag2 = flag3 = flag4 = 1
        for j in range(4*i+1):
            if j<=i and flag1 != 0:
                a = facets_kij_oct(i,j)
                b = facets_kij_oct(i+1,j)
                c = facets_kij_oct(i+1,j+1)
                facets.append([a,b,c])
                if j+1 <= i :
                    d = facets_kij_oct(i,j+1)
                    facets.append([a,c,d])    
                else:
                    flag1 = 0
                    
            if i<=j<=2*i and flag2 != 0:
                a = facets_kij_oct(i,j)
                b = facets_kij_oct(i+1,j+1)
                c = facets_kij_oct(i+1,j+2)
                facets.append([a,b,c])
                if j+1 <= 2*i :
                    d = facets_kij_oct(i,j+1)
                    facets.append([a,c,d])    
                else:
                    flag2 = 0
                    
            if 2*i<=j<=3*i and flag3 != 0:
                a = facets_kij_oct(i,j)
                b = facets_kij_oct(i+1,j+2)
                c = facets_kij_oct(i+1,j+3)
                facets.append([a,b,c])
                if j+1 <= 3*i :
                    d = facets_kij_oct(i,j+1)
                    facets.append([a,c,d])    
                else:
                    flag3 = 0

            if 3*i<=j<=4*i and flag4 != 0:
                a = facets_kij_oct(i,j)
                b = facets_kij_oct(i+1,j+3)
                c = facets_kij_oct(i+1,j+4)
                facets.append([a,b,c])
                if j+1 <= 4*i :
                    d = facets_kij_oct(i,j+1)
                    facets.append([a,c,d])                       
                
    k0 = 2 * (n*n + n) + 1
    k1 = 2 * (n*n - n) + 1
    nv = 4*n*n + 2
    for i in range(n):
        flag1 = flag2 = flag3 = flag4 = 1
        for j in range(4*i+1):
            if j<=i and flag1 != 0:
                a = facets_kij_oct(i,j) + k0
                if a > nv : 
                    a = (a%nv) + k1
                b = (facets_kij_oct(i+1,j) + k0) 
                if b > nv : 
                    b = (b%nv) + k1
                c = (facets_kij_oct(i+1,j+1) + k0) 
                if c > nv : 
                    c = (c%nv) + k1
                facets.append([a,b,c])
                if j+1 <= i :
                    d = (facets_kij_oct(i,j+1) + k0) 
                    if d > nv : 
                        d = (d%nv) + k1
                    facets.append([a,c,d])    
                else:
                    flag1 = 0
                    
            if i<=j<=2*i and flag2 != 0:
                a = (facets_kij_oct(i,j) + k0) 
                if a > nv : 
                    a = (a%nv) + k1
                b = (facets_kij_oct(i+1,j+1) + k0) 
                if b > nv : 
                    b = (b%nv) + k1
                c = (facets_kij_oct(i+1,j+2) + k0) 
                if c > nv : 
                    c = (c%nv) + k1
                facets.append([a,b,c])
                if j+1 <= 2*i :
                    d = (facets_kij_oct(i,j+1) + k0) 
                    if d > nv : 
                        d = (d%nv) + k1
                    facets.append([a,c,d])    
                else:
                    flag2 = 0
                    
            if 2*i<=j<=3*i and flag3 != 0:
                a = (facets_kij_oct(i,j) + k0) 
                if a > nv : 
                    a = (a%nv) + k1
                b = (facets_kij_oct(i+1,j+2) + k0) 
                if b > nv : 
                    b = (b%nv) + k1
                c = (facets_kij_oct(i+1,j+3) + k0) 
                if c > nv : 
                    c = (c%nv) + k1
                facets.append([a,b,c])
                if j+1 <= 3*i :
                    d = (facets_kij_oct(i,j+1) + k0) 
                    if d > nv : 
                        d = (d%nv) + k1
                    facets.append([a,c,d])    
                else:
                    flag3 = 0

            if 3*i<=j<=4*i and flag4 != 0:
                a = (facets_kij_oct(i,j) + k0) 
                if a > nv : 
                    a = (a%nv) + k1
                b = (facets_kij_oct(i+1,j+3) + k0) 
                if b > nv : 
                    b = (b%nv) + k1
                c = (facets_kij_oct(i+1,j+4) + k0) 
                if c > nv : 
                    c = (c%nv) + k1
                facets.append([a,b,c])
                if j+1 <= 4*i :
                    d = (facets_kij_oct(i,j+1) + k0) 
                    if d > nv : 
                        d = (d%nv) + k1
                    facets.append([a,c,d])
                    
    return facets

def facets_kij_oct(i,j):
    if i == 0:
        return 1
    if j == 4*i :
        return 2 * (i*i - i + 1)
    else:
        return 2 * (i*i - i + 1) + j

def generate_triaxial_ellipsoid_oct(a, b, c, n):
    vertices = generate_vertices_oct(a, b, c, n)
    facets = generate_facets_oct(n)
    # 定义输出文件的名称
    model_filename = 'oct_ellipsoid'
    obj_filename = model_filename + '.obj'

    # 将椭球的顶点和面存成obj文件, 保存至obj_filename
    write_obj_file(obj_filename, vertices, facets)


# 生成三轴椭球的顶点坐标数据, 并令abc与xyz分别重合
def generate_ellipsoid_vertices(a, b, c, u_resolution=100, v_resolution=100):
    """
    生成三轴椭球的顶点坐标数据
    :param a: 椭球的x轴半径
    :param b: 椭球的y轴半径
    :param c: 椭球的z轴半径
    :param u_resolution: u方向上的分辨率
    :param v_resolution: v方向上的分辨率
    :return: 生成的三轴椭球的三维坐标数据
    """
    if not (a >= b >= c):
        raise ValueError("a must be greater than or equal to b, and b must be greater than or equal to c.")
    u = np.linspace(0, 2 * np.pi, u_resolution)  # 在[0, 2π]区间上等分u_resolution份，得到u方向的坐标
    v = np.linspace(0, np.pi, v_resolution)  # 在[0, π]区间上等分v_resolution份，得到v方向的坐标
    x = a * np.outer(np.cos(u), np.sin(v))  # 通过np.outer函数生成a方向上的坐标
    y = b * np.outer(np.sin(u), np.sin(v))  # 通过np.outer函数生成b方向上的坐标
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))  # 通过np.outer函数生成c方向上的坐标
    return x, y, z

# 将顶点坐标和面存成obj文件
def write_obj_file(filename, vertices, faces):
    """
    将顶点坐标和面存成obj文件
    :param filename: obj文件名
    :param vertices: 顶点坐标
    :param faces: 面数据
    """
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))  # 写入顶点坐标数据
        for face in faces:
            f.write('f')
            for vertex in face:
                f.write(' {}'.format(vertex))
            f.write('\n')

# 生成椭球的面数据
def generate_faces(u_resolution, v_resolution):
    """
    生成椭球的面数据
    :param u_resolution: u方向上的分辨率
    :param v_resolution: v方向上的分辨率
    :return: 生成的椭球面数据
    """
    faces = []
    for i in range(u_resolution - 1):
        for j in range(v_resolution - 1):
            a = i * v_resolution + j + 1
            b = i * v_resolution + (j + 1) % v_resolution + 1
            c = (i + 1) * v_resolution + (j + 1) % v_resolution + 1
            d = (i + 1) * v_resolution + j + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
#            faces.append([a + v_resolution, b + v_resolution, c + v_resolution])
#            faces.append([a + v_resolution, c + v_resolution, d + v_resolution])
    return faces

# 通过坐标系旋转, 实现向量变换, 分析过程见2023年7月31日工作记录
def rotate_vector(vector, lambda_deg, beta_deg):
    """
    通过坐标系旋转, 实现向量变换
    :param vector: 待旋转的向量，可以是单个向量，也可以是一个矩阵，每行表示一个向量
    :param lambda_deg: 绕z轴旋转的角度，单位为度
    :param beta_deg: 绕y轴旋转的角度，单位为度
    :return: 旋转后的向量或矩阵
    """
    # 将角度值转换为弧度值
    lambda_rad = np.radians(-lambda_deg)
    beta_rad = np.radians(beta_deg-90)

    # 先沿Y轴方向旋转90-beta，生成旋转矩阵
    R1_y = np.array([[np.cos(beta_rad), 0, -np.sin(beta_rad)],
                   [0, 1, 0],
                   [np.sin(beta_rad), 0, np.cos(beta_rad)]])

    # 再沿Z轴方向旋转lambda，生成旋转矩阵
    R2_z = np.array([[np.cos(lambda_rad), np.sin(lambda_rad), 0],
                   [-np.sin(lambda_rad), np.cos(lambda_rad), 0],
                   [0, 0, 1]])
    
    # 生成整体的旋转矩阵R
    R = np.dot(R2_z, R1_y)

    # 对向量进行旋转, 当vector是3行时,R乘在左边;
    # 当vector是N行3列时, 根据AB的转置等于B的转置乘以A的转置:
    rotated_vector = np.dot(vector, np.transpose(R))

    return rotated_vector

# 将obj文件转成Damit能识别的格式
def generate_model_file_from_obj(obj_filename, model_filename):
    """
    将obj文件转成Damit能识别的格式
    :param input_filename: 输入obj文件名
    :param output_filename: 输出文件名
    """
    # Read the OBJ file
    with open(obj_filename, 'r') as f:
        lines = f.readlines()

    # Extract vertices and faces from the OBJ file
    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertex = list(map(float, line.strip().split()[1:]))
            vertices.append(vertex)
        elif line.startswith('f '):
            face = list(map(int, line.strip().split()[1:]))
            faces.append(face)

    # Write the model file
    with open(model_filename, 'w') as f:
        # Write the number of vertices and faces
        f.write('{} {}\n'.format(len(vertices), len(faces)))

        # Write the vertex coordinates
        for vertex in vertices:
            f.write('{} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

        # Write the faces
        for face in faces:
            f.write('{}\n'.format(' '.join(map(str, face))))
    
def generate_lc_file_fixpos(pos1=[1.3, 0, 0], pos2=[1, 0, 0], filename="model_lcs_rel", tdur=10.0/24, dt=1.0/1440):
    """
    生成光变曲线文件.

    :param pos1: 1号天体的位置, 默认为 [1.3, 0, 0].
    :type pos1: list of float.

    :param pos2: 2号天体的位置, 默认为 [1, 0, 0].
    :type pos2: list of float.

    :param filename: 生成的光变曲线文件名, 默认为 "model_lcs_rel".
    :type filename: str.

    :param tdur: 观测时长, 默认为 10.0/24.
    :type tdur: float.

    :param dt: 观测时间间隔, 默认为 1.0/1440.
    :type dt: float.

    :return: 
    # data存储计算后的新数据
    # data 第一行是光变曲线数目, 第二行以后才是光变曲线数据
    # 每条光变曲线的第一行指定观测点数, 然后是光变曲线观测值
    # 观测值含8列, 分别是t,flux,pos1三列,pos2三列
    """
    # 生成观测时间数组
    t0 = Time("2025-01-01 0:0:0", format='iso')
    obs_tarr1 = np.arange(t0.jd, t0.jd+tdur, dt)

    # 生成数据
    data = []
    data.append("1\n")
    data.append(f"{len(obs_tarr1)} 0\n")

    for obs_tarr_i in obs_tarr1:
        new_line_data = [
            obs_tarr_i,
            1.0,
            pos1[0], pos1[1], pos1[2],
            pos2[0], pos2[1], pos2[2]
        ]
        data.append(new_line_data)

    # 将数据写入文件
    with open(filename, "w") as f:
        for row in data:
            if len(row) == 8:
                line = f"{row[0]:.6f}  {row[1]:.8e}  {row[2]:.8f} {row[3]:.8f} {row[4]:.8f}  {row[5]:.8e} {row[6]:.8e} {row[7]:.8e}\n"
            else:
                line = f"{row}"
            f.write(line)
      
# 创建out_par文件
def generate_par_file(lambda_deg=0,beta_deg=90):
    with open("out_par", "w") as f:
        line = f"{lambda_deg:.1f}  {beta_deg:.1f} 3 \n"
        f.write(line) 
        f.write('2460676.500000 0 \n')
        f.write('0.5 0.1 -0.5 \n')
        f.write('0.1 \n')     

# 给定abc、分辨率和自转方向, 生成三轴椭球顶点和面的坐标向量,
# 注意此时三轴椭球已经根据自转方向进行了旋转
def generate_rotEllipsoid(a=1,b=1,c=1,lambda_deg = 0, beta_deg = 90, u_resolution = 32, v_resolution = 32):

    # 生成三轴椭球的顶点坐标数据, 另a与x轴重合, c与z轴重合
    x, y, z = generate_ellipsoid_vertices(a, b, c, u_resolution, v_resolution)
    # 将椭球三个方向的数据粘合成矩阵, 每行表示一个顶点的三个坐标
    vertices_ori = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    # 将整个三轴椭球按给定的(lambda,beta)方向旋转
    vertices = rotate_vector(vertices_ori, lambda_deg, beta_deg)
    
    # 生成椭球的面数据, 每个面由三个顶点决定, 顶点按顺序编号, 对应vertices的顺序
    faces = generate_faces(u_resolution, v_resolution)
    
    return vertices,faces

def generate_rotEllipsoid_oct(a=1,b=1,c=1,lambda_deg = 0, beta_deg = 90, n = 8):

    # 生成三轴椭球的顶点坐标数据, 另a与x轴重合, c与z轴重合
    vertices_ori = generate_vertices_oct(a, b, c, n)

    
    # 将整个三轴椭球按给定的(lambda,beta)方向旋转
    vertices = rotate_vector(vertices_ori, lambda_deg, beta_deg)
    
    # 生成椭球的面数据, 每个面由三个顶点决定, 顶点按顺序编号, 对应vertices的顺序
    facets = generate_facets_oct(n)

    # 定义输出文件的名称
    model_filename = 'oct_ellipsoid'
    obj_filename = model_filename + '.obj'

    # 将椭球的顶点和面存成obj文件, 保存至obj_filename
    write_obj_file(obj_filename, vertices, facets)
    
    return vertices,faces


# 该程序只适用于绕z轴旋转的情况
def generate_pos2_from_pos1(pos1=[1,0,0], phaseAng=0, dpos2=1):
    """
    根据pos1和相位角phaseAng，计算pos2，使得pos1和pos2之间的夹角是phaseAng度，并且pos2向量的长度是dpos2.
    
    :param pos1: 1号天体的位置，默认为 [1,0,0].
    :type pos1: list of float.

    :param phaseAng: 天体之间的相位角(度数)，默认为 0.
    :type phaseAng: float.

    :return: pos2，使得pos1和pos2之间的夹角是phaseAng度.
    :rtype: list of float.
    """
    # 将角度转换为弧度
    phaseAng_rad = np.deg2rad(phaseAng)

    # 计算旋转矩阵
    rotMatrix = np.array([[np.cos(phaseAng_rad), np.sin(phaseAng_rad), 0],
                          [-np.sin(phaseAng_rad), np.cos(phaseAng_rad), 0],
                          [0, 0, 1]])

    # 计算pos2
    pos2 = np.dot(rotMatrix, pos1)    
    pos2 = pos2 / np.linalg.norm(pos2) * dpos2

    return pos2.tolist()

# 检验坐标旋转的程序是否正确
# vertices_ori=np.array([0,0,1]).reshape(1, 3)
# print(rotate_vector(vertices_ori, 40, 0))

os.chdir('/Users/superman/Downloads/')
os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/DamitBenu/test_ell/')

# 给定三轴椭球参数，包括三个轴abc, 自转方向(lambda,beta) 和 椭球在uv方向的分辨率
a = 3  # semi-axis length along x
b = 2  # semi-axis length along y
c = 2  # semi-axis length along z
lambda_deg = 90
beta_deg = 0 
u_resolution = 45  # number of vertices along the u direction
v_resolution = 30  # number of vertices along the v direction

# 其他可指定的参数
pos1=[1.3,0,0]
timeflag = f"l{lambda_deg}b{beta_deg}u{u_resolution}v{v_resolution}a{a}b{b}c{c}_{time.strftime('%y%m%d')}"

# 生成三轴椭球顶点和面的坐标向量
vertices,faces = generate_rotEllipsoid(a=a,b=b,c=c,lambda_deg = lambda_deg, beta_deg = beta_deg, u_resolution = u_resolution, v_resolution = v_resolution)

# 定义输出文件的名称
model_filename = 'ellipsoid_' + timeflag
obj_filename = model_filename + '.obj'

# 将椭球的顶点和面存成obj文件, 保存至obj_filename
write_obj_file(obj_filename, vertices, faces)
    
# 将obj文件再转成Damit能识别的格式, 保存至model_filename
generate_model_file_from_obj(obj_filename, model_filename)

# # 绘制三轴椭球
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ellipsoid')
plt.show()

generate_rotEllipsoid_oct(a=3,b=2,c=2,lambda_deg = 40, beta_deg = 40, n = 4)

# 0-180度范围内生成一系列phaseAng，步长30度
phaseAng_arr = range(0, 180, 30)

for phaseAng in phaseAng_arr:
    
    timeflag_ext = f"p{phaseAng}_{timeflag}"
    
    pos2 = generate_pos2_from_pos1(pos1=pos1, phaseAng=phaseAng, dpos2=1)
        
    # 生成模拟光变曲线需要的 时间-测站位置文件 model_lcs_rel
    generate_lc_file_fixpos(pos1=pos1,pos2=pos2)
    
    # 生成模拟光变曲线需要的 参数文件 out_par
    generate_par_file(lambda_deg,beta_deg)
    
    # 定义要执行的shell命令
    lc_filename = "lc_" + timeflag_ext
    
    # 使用 Python 的 f-string 模式来简化字符串拼接，将变量直接嵌入字符串中
    command = f"cat model_lcs_rel | lcgenerator -v out_par {model_filename} {lc_filename}"
    
    # 执行shell命令
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 输出结果
    print(result.stdout.decode())
    
    
    # gnuplot绘图命令
    fig_filename = "fig_" + timeflag_ext +'.png'
    
    gnuplot_script = f"""
    set term png
    set output '{fig_filename}'
    plot '{lc_filename}'
    """
    
    # 调用gnuplot程序
    # arial是mac自带字体, 没找到是因为字体路径设置有问题
    gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)
    gnuplot.communicate(gnuplot_script.encode())

