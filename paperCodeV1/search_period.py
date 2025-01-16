#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:02:47 2023

分别用LS方法和PDM方法搜寻小行星自转周期

修改参数
wkdir: 工作路径，默认在“下载”文件夹
input_data: 光变曲线文件，共8列，前两列是时间和流量

以下函数里面的参数也可调整：
freqs = np.arange(max(fmin,0.1),min(fmax,10),fstep)
fine_freqs = np.arange(0.3*peak_frequency,10*peak_frequency,fstep)
freqs_PDM, theta_PDM = pdm(hours, col2, f_min=0.001, f_max=10, delf=1e-4)
fine_freqs_PDM, fine_theta_PDM = pdm(hours, col2, f_min=0.3*peak_frequency_PDM, f_max=3*peak_frequency_PDM, delf=1e-6,nbin=10)

@author: superman
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from pdmpy import pdm

def search_period_plot_gls(freqs,power,title=""):
    # 绘制频率谱图，横坐标为小时数
    fig, ax = plt.subplots(figsize=(16, 9))
    
    ax.plot(1./freqs, power,linewidth=4)
    ax.set_title(title,fontsize=32, fontweight='bold')
    plt.xlabel('Hours',fontsize=24, fontweight='bold')
    plt.ylabel('Power',fontsize=24, fontweight='bold')
    
    # 设置坐标轴刻度和字号
    ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=12)
    ax.spines['top'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    
    #plt.savefig("plot.png")
    plt.show()

# 根据周期画相位图
def plot_phaselc(hours, col2,peak_period,title="Fine Phased LC"):
    # 计算相位
    phase = hours / peak_period % 1

    # 绘制相位图
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(phase, col2,linewidth=4)
    ax.set_title(title,fontsize=32, fontweight='bold')
    plt.xlabel('Phase',fontsize=24, fontweight='bold')
    plt.ylabel('Relative flux',fontsize=24, fontweight='bold')
    
    # 设置坐标轴刻度和字号
    ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=12)
    ax.spines['top'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    
    #plt.savefig("plot.png")
    plt.show()
    
# 画图
def plot_lc(hours, col2,title="Whole LC"):

    # 绘制相位图
    fig, ax = plt.subplots(figsize=(40, 9))
    ax.scatter(hours/24.0, col2,linewidth=4)
    ax.set_title(title,fontsize=32, fontweight='bold')
    plt.xlabel('Time (days)',fontsize=24, fontweight='bold')
    plt.ylabel('Relative flux',fontsize=24, fontweight='bold')
    
    # 设置坐标轴刻度和字号
    ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=12)
    ax.spines['top'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    
    #plt.savefig("plot.png")
    plt.show()    
    
def search_period_V2(input_data,fmin_ud = 0.1,fmax_ud = 10,nbin = 10):
    
    with open(input_data, 'r') as f:
        lines = f.readlines()
    
    # 定义8个数组，用于存储8列数据
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []
    
    # 初始化一个空列表，用于记录被舍弃的行
    discarded_lines = []
    
    # 遍历每一行数据
    for line in lines:
        # 将每一行数据按照空格分割成多个元素
        elements = line.split()
        # 判断每行数据的列数是否为8
        if len(elements) == 8:
            # 如果第一列已经出现过，记录并跳过
            if float(elements[0]) in col1:
                discarded_lines.append(elements)
                continue
            # 如果列数为8，则将每列数据存储到对应的数组中
            col1.append(float(elements[0]))
            col2.append(float(elements[1]))
            col3.append(float(elements[2]))
            col4.append(float(elements[3]))
            col5.append(float(elements[4]))
            col6.append(float(elements[5]))
            col7.append(float(elements[6]))
            col8.append(float(elements[7]))
    
    # 输出每个数组的长度，以及第一个元素
    # =============================================================================
    # print(len(col1), col1[0])
    # print(len(col2), col2[0])
    # print(len(col3), col3[0])
    # print(len(col4), col4[0])
    # print(len(col5), col5[0])
    # print(len(col6), col6[0])
    # print(len(col7), col7[0])
    # print(len(col8), col8[0])
    # =============================================================================
    
    # 将8个数组按照 col1 从小到大排序
    col1, col2, col3, col4, col5, col6, col7, col8 = zip(*sorted(zip(col1, col2, col3, col4, col5, col6, col7, col8)))
    
    col1 = np.array(col1)
    col2 = np.array(col2)
    
    # 将JD时间转换为单位为天的时间
    days = col1 - col1[0]
    
    # 将时间转换为单位为小时的时间
    hours = days * 24
    
    # 计算时间间隔
    delta_time = np.abs(np.diff(hours))
    
    # 计算最大和最小频率
    fmax = 1 / (2 * delta_time.min())
    fmin = 1 / (2 * (hours[-1]-hours[0]))
    
    # 定义频率步长
    fstep = min(1e-5,1/(hours.max()-hours.min()))
    # 定义频率数组,最小频率小于100hr,最大频率大于0.1hr
    freqs = np.arange(max(fmin,fmin_ud),min(fmax,fmax_ud),fstep)
    
    # 使用generalized Lomb-Scargle方法计算频率谱
    # 在使用 LombScargle 类计算频率谱时，可以选择不加 autopower 来计算频率谱。在这种情况下，LombScargle 类会返回一个 LombScargle 对象，而不是频率和功率的数组。
    model = LombScargle(hours, col2)
    power = model.power(freqs)
    
    # 寻找峰值位置
    peak_index = np.argmax(power)
    peak_frequency = freqs[peak_index]
    peak_period = 1.0 / peak_frequency
    
    # 输出峰值位置和周期
    print("LS粗搜索 ...")
    #print("  峰值位置：", peak_index)
    print("  峰值频率：", peak_frequency)
    print("  峰值周期：", peak_period,"小时")
    print("  峰值周期*2：", peak_period*2,"小时")
    
    # 定义更精细的频率步长
    fstep = 1e-6 #min(1e-5,0.1/(hours.max()-hours.min()))
    # 定义更精细的频率范围
    #fine_freqs = np.linspace(0.2*peak_frequency, 10*peak_frequency, 1000000)
    fine_freqs = np.arange(max(0.3*peak_frequency,fmin,fmin_ud),min(10*peak_frequency,fmax,fmax_ud),fstep)
    
    # 计算更精细的频率谱
    fine_power = model.power(fine_freqs)
    
    # 寻找峰值位置
    fine_peak_index = np.argmax(fine_power)
    fine_peak_frequency = fine_freqs[fine_peak_index]
    fine_peak_period = 1.0 / fine_peak_frequency
    
    # 输出峰值位置和周期
    print("LS精细搜索 ...")
    #print("  峰值位置：", fine_peak_index)
    print("  峰值频率：", fine_peak_frequency)
    dp = (1.0/(fine_peak_frequency-fmin)-1.0/(fine_peak_frequency+fmin))
    print("  峰值周期：", fine_peak_period,"小时")
    print("       +/- {:.6f}".format(dp/2))
    print("  峰值周期*2：", fine_peak_period*2,"小时")
    print("         +/- {:.6f}".format(dp))
    
    print("LS方法建议采用周期范围（含一阶谐波）：")
    print("  ", fine_peak_period-1.5*dp,fine_peak_period+1.5*dp)
    print("  ", 2*fine_peak_period-3*dp,2*fine_peak_period+3*dp)
    
    # 绘制粗搜错频率谱图，横坐标为小时数
    search_period_plot_gls(freqs,power,title="LS Rough spectrum")
    # 绘制精细搜错频率谱图
    search_period_plot_gls(fine_freqs,fine_power,title="LS Fine spectrum")
    # LS周期的相位图
    plot_phaselc(hours, col2,fine_peak_period,title="LS Phased LC : P =" + f"{fine_peak_period:.5f}")
    # 2倍拟合周期的相位图
    plot_phaselc(hours, col2,2*fine_peak_period,title="LS Phased LC : P =" + f"{2*fine_peak_period:.5f}")
    
    
    # PDM 方法搜寻自转周期
    # 粗搜索, 周期范围0.1-1000小时, 步长0.0001
    freqs_PDM, theta_PDM = pdm(hours, col2, f_min=fmin_ud, f_max=fmax_ud, delf=1e-4)
    peak_frequency_PDM = freqs_PDM[np.argmin(theta_PDM)]
    
    # 精细搜索, 步长0.00001
    fine_freqs_PDM, fine_theta_PDM = pdm(hours, col2, f_min=max(0.3*peak_frequency_PDM,fmin_ud), f_max=min(3*peak_frequency_PDM,fmax_ud), delf=1e-6,nbin=nbin)
    
    #最佳PDM周期
    period_PDM = 1./fine_freqs_PDM[np.argmin(fine_theta_PDM)]
    
    # 作图：PDM结果，横坐标分别是周期和频率
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3),constrained_layout=True)
    ax1.plot(1/fine_freqs_PDM, fine_theta_PDM, 'k')
    ax1.axvline(period_PDM, color='red',alpha=0.3)
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Theta')
    
    ax2.plot(fine_freqs_PDM, fine_theta_PDM, 'k')
    ax2.axvline(1/period_PDM, color='red',alpha=0.3)
    for i in range(2,3):
        ax2.axvline(1/period_PDM/i, color='red', ls='--',alpha=0.3)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Theta')
    fig.suptitle("PDM diagram")
    # PDM周期的相位图
    plot_phaselc(hours, col2,period_PDM,title="PDM Phased LC : P =" + f"{period_PDM:.5f}")
    # 2倍拟合周期的相位图
    plot_phaselc(hours, col2,2*period_PDM,title="PDM Phased LC : P =" + f"{2*period_PDM:.5f}")
    
    plot_lc(hours, col2,title="Whole LC")
    
    print("PDM方法建议采用周期范围：")
    print("  ", period_PDM,"+/- 0.001")
    print("  ", 2*period_PDM,"+/- 0.001")
    
    times = 1 # 2 for 一般三轴椭球, 1 for 2016HO3
    range_ls = [times*fine_peak_period-3*dp,times*fine_peak_period+3*dp]    
    range_pdm = [times*period_PDM-0.001,times*period_PDM+0.001]
    range_pdm = [times*period_PDM-0.005,times*period_PDM+0.005]
    range_pdm = [period_PDM*(1-1.0/nbin),period_PDM*(1+1.0/nbin)]
    
    
    # 临时为2016HO3加的
    range_ls = [times*fine_peak_period-0.0001,times*fine_peak_period+0.0001]

    return range_ls,range_pdm
    

def example():
    wkdir = '/Users/superman/Downloads'
    # 频率范围user-defined参数,最大不超过（fmin_ud,fmax_ud）
    fmin_ud = 0.1
    fmax_ud = 10
    # PDM合并区间数
    nbin = 10
    
    # Define the number of files to generate, 该值等于并行进程数
    n = 1
    #input_file = "/Users/superman/science/测试vsc/DAMIT源程序0.20.1/input_period_scan"
    input_data = "/Users/superman/science/测试vsc/DAMIT源程序0.20.1/test_lcs_rel"
    input_data = "/Users/superman/Documents/资料/新分类/太阳系小天体/DamitBenu/test_lcs_rel"
    #input_data = "/Users/superman/Documents/资料/新分类/太阳系小天体/DamitTest248428/test_lcs_rel"
    #output_merge = "/Users/superman/science/测试vsc/DAMIT源程序0.20.1/out_periods"
    
    #pmin_user = 
    #pmax_user = 
    
    os.chdir(wkdir)
    search_period_V2(input_data,fmin_ud = 0.1,fmax_ud = 10,nbin = 10)
