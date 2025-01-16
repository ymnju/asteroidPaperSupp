#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:34:06 2023

@author: superman
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# numbered NEO asteroids
NeoParFile = '/Users/superman/Documents/工作_科研/课题和论文/一作/2023.asteroid/data/NEOpars_num.csv'

# Damit asteroids
DamParFile = '/Users/superman/Documents/工作_科研/课题和论文/一作/2023.asteroid/data/DamitAstMod.csv'

# 绘制直方图
def plot_histogram(data, title, xlabel, nbins=50, alpha=0.7, xlim=None):
    plt.figure(figsize=(6, 6))
    plt.hist(data, bins=nbins, edgecolor='steelblue', alpha=alpha)
    #plt.title(title, fontsize=10)  # 调整标题文字大小
    plt.xlabel(xlabel, fontsize=14)  # 调整横轴标签文字大小
    plt.ylabel("Count", fontsize=14)  # 调整纵轴标签文字大小
    plt.xticks(fontsize=10)  # 调整横轴刻度文字大小
    plt.yticks(fontsize=10)  # 调整纵轴刻度文字大小
    if xlim:
        plt.xlim(xlim)
    else:
        plt.tight_layout()
    # plt.gca().spines['linewidth'] = 1.5  # 调整轴线粗细
    plt.gca().spines['bottom'].set_linewidth(1.5)  # 调整底部轴线粗细
    plt.gca().spines['left'].set_linewidth(1.5)  # 调整左侧轴线粗细
    plt.show()

"""
    ------------------------------------
    task 1 : 画所有Damit小天体的λ和β的直方图
    ------------------------------------
"""

# 读取CSV数据
df = pd.read_csv(DamParFile)

# 使用describe()统计列的中值、标准差等
stats = df.describe()
mids = stats.loc['50%']

# 创建子图            
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for ax, col in zip(axs.flat, df.columns[2:4]):
    # 设置直方图范围   
    ax.hist(df[col], bins=50, alpha=0.7, edgecolor='steelblue')
    # 绘制中值虚线
    ax.axvline(mids[col], color='r', linestyle='dashed', linewidth=1)
    # 在图中标注中值 
    ax.text(mids[col]*1.05, ax.get_ylim()[1] * 0.9, '{:0.2f}'.format(mids[col])) 
    ax.set_title(col)

# 绘制lambda直方图
bin_width = 20; bin_boundaries = np.arange(0, 360 + bin_width, bin_width)
plot_histogram(df["lambda"], title="Distribution of Lambda ", xlabel="Lambda (degree)", nbins=bin_boundaries, xlim=(0, 360))

# 绘制beta直方图
bin_width = 10; bin_boundaries = np.arange(-90, 90 + bin_width, bin_width)
plot_histogram(df["beta"], title="Distribution of Beta ", xlabel="Beta (degree)", nbins=bin_boundaries, xlim=(-90, 90))
 

"""
    ----------------------
    task 2 : 画NEO参数直方图
    ----------------------
"""

# 读取CSV数据
df = pd.read_csv(NeoParFile)

#df = df[df['a(AU)'] < 1.3]

# 使用describe()统计列的中值、标准差等
stats = df.describe()
mins = stats.loc['min']
means = stats.loc['mean']
mids = stats.loc['50%']
maxs = stats.loc['max']

# 计算每个列的范围
ranges = {col: (mins[col] , 2*stats.loc['75%'][col]) 
          for col in df.columns}

# 创建子图            
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

# 遍历每一列绘制直方图
for ax, col in zip(axs.flat, df.columns[1:]):
    # 设置直方图范围   
    n, bins, patches = ax.hist(df[col], bins=50, range=ranges[col], alpha=0.7)
    # 绘制中值虚线
    ax.axvline(mids[col], color='r', linestyle='dashed', linewidth=1)
    # 在图中标注中值 
    ax.text(mids[col]*1.05, ax.get_ylim()[1] * 0.9, '{:0.2f}'.format(mids[col])) 
    ax.set_title(col)

    # 获取峰值对应的横坐标
    peak_bin = np.argmax(n) 
    peak_x = (bins[peak_bin] + bins[peak_bin+1]) / 2

    # 在图中标注峰值
    ax.text(peak_x*1.05, ax.get_ylim()[1] * 0.2, '{:0.2f}'.format(peak_x))


plt.tight_layout()
plt.show()

# 绘制a的直方图
bin_width = 0.2; bin_boundaries = np.arange(0, 3 + bin_width, bin_width)
plot_histogram(df["a(AU)"], title="Distribution of a(AU) ", xlabel="Semi-major axis (AU)", nbins=bin_boundaries, xlim=(0, 3))

# 绘制rot_per(h)的直方图
bin_width = 0.5; bin_boundaries = np.arange(0, 20 + bin_width, bin_width)
plot_histogram(df["rot_per(h)"], title="Distribution of rotational period (hr) ", xlabel="Rotational period (hr)", nbins=bin_boundaries, xlim=(0, 20))

# 输出每列统计结果
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(stats.iloc[:, 1:].round(2))
    

"""
    --------------------------------------------------------------
    task 3 : 根据Tardioli et al.(2017)的拟合结果，画NEO的obliquity分布 
    --------------------------------------------------------------
"""

# 生成一些gamma值
gamma = np.linspace(0, np.pi, 1000)
cosgm = np.cos(gamma)
# 计算相应的p值
p = 1.12*cosgm**2 - 0.32*cosgm + 0.13

# 绘制gamma与p之间的关系图
plt.plot(gamma*57.3, p)
plt.xlabel('gamma')
plt.ylabel('p')
plt.title('p as a function of gamma')
plt.show()    

# 绘制cos gamma与p之间的关系图
plt.plot(cosgm, p)
plt.xlabel('cos(gamma)')
plt.ylabel('p')
plt.title('p as a function of gamma')
plt.show()   