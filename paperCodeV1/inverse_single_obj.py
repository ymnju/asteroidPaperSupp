#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:14:50 2023

 计算指定小天体指定时刻地球和太阳的黄道坐标

@author: superman
"""
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import ast_ecl_fun as af
import os
#import sys
from damit_funs import global_convexinv_parallel
from datetime import datetime, timedelta
from period_scan_parallel_V2 import period_scan_parallel_V2
from search_period_V2 import search_period_V2,plot_phaselc,plot_lc
import subprocess
import time as timesleep
import pandas as pd

#timesleep.sleep(600)

def get_interval_dates(start_date, time_interval, n):
    dates = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    for _ in range(n):
        dates.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))
        current_date += timedelta(days=time_interval)

    return dates

def get_lcdata(input_data):
    
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
    
    # 将8个数组按照 col1 从小到大排序
    col1, col2, col3, col4, col5, col6, col7, col8 = zip(*sorted(zip(col1, col2, col3, col4, col5, col6, col7, col8)))
    
    col1 = np.array(col1)
    col2 = np.array(col2)
    
    # 将JD时间转换为单位为天的时间
    days = col1 - col1[0]
    
    # 将时间转换为单位为小时的时间
    hours = days * 24
        
    return hours,col2

def astinv_1(i_lcs_file,flag):
    # 载入基本的kernels文件
    os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/cspice_wk/kernel/')
    spice.furnsh(['de440.bsp','naif0012.tls','codes_300ast_20100725.tf'])
    spice.furnsh(['20469219.bsp'])
    spice.furnsh(['traSat3AUw90_904612418.bsp','traSat1p3AUw90_907711300.bsp',
                  'traSat1AU_903073025.bsp','traSat1p01AUw180_907468185.bsp'])
    
    # 必要的常数值
    au_unit = 1.495978707e8
    
    # 初始条件
    # 指定观测目标 ID
    observer = '2016'       # 2016 HO3
    lcper = 0.5 #0.46155                 # 假设一个光变周期, 用以确定仿真的观测时长和cadence (hr)
    lam_list = []
    beta_list = []
    per_list = []
                
    # global_convexinv_parallel() 并行计算函数所需参数
    lam_step = 1    # 赤经步长
    beta_step = 1   # 赤纬步长
    n_per = 1       # 周期取值个数
    npt = 73 # 每个周期内采样点数
    
    
    # 精测自转周期时需要的输入参数
    input_file = "/Users/superman/Documents/资料/新分类/太阳系小天体/cspice_wk/"+observer+"/input_period_scan"
        
    # 进入数据读取和保存工作空间
    # spice.furnsh(observer+'.bsp') 
    os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/cspice_wk/'+observer+'/')
    
    directories = ['obs_lcs_rel', 'out_par_new', 'out_lcs_new', 'period_scan']
    for directory in directories:
        subprocess.run(['mkdir', directory], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
    file_log_name = "output_log.txt"
    file_outdat_name = "output_dat.txt"
    file_perlog_name = "output_perlog.txt"
    
    file_log = open(file_log_name, "w") ; file_log.close()
    
    file_perlog = open(file_perlog_name, "w") ; 
    file_perlog.write("period   rms     chi2  iter. dark area %\n")
    file_perlog.close()
    
    file_outdat = open(file_outdat_name, "w")
    file_outdat.write("per_search, per_std,   lambda,     beta,   period\n")
    file_outdat.close()
           
    period_scan_list = ["period   rms      chi2      iter. dark area %"]
    
    if 0: # 满足一定条件才进行周期粗搜索和精细搜索,否则直接赋值
        range_ls,range_pdm = search_period_V2(i_lcs_file,fmin_ud = 1.7,fmax_ud = 3,nbin=40)
        
        # 周期精测, 为提高效率只对一条光变曲线进行周期精测
        if 1:
            output_merge = "./period_scan/out_merge_periods"
            print("对PDM结果精确拟合, ", end='')
            i_output_string_pdm = period_scan_parallel_V2(12,input_file, i_lcs_file, output_merge, range_pdm[0],range_pdm[1])
            print("对LS结果精确拟合, ", end='')
            i_output_string_ls = period_scan_parallel_V2(12,input_file, i_lcs_file, output_merge, range_ls[0],range_ls[1])
            pdm_str_list = i_output_string_pdm.split()
            ls_str_list = i_output_string_ls.split()
            
            file_perlog = open(file_perlog_name, "a") 
            # 程序强行指定PMD周期对应i_per_std=1e-4, LS周期对应1e-3
            if float(pdm_str_list[1]) <= float(ls_str_list[1]):
                period_scan_list.append(i_output_string_pdm)
                per_search = float(pdm_str_list[0])
                i_per_std = 1e-4
                file_perlog.write(f"{i_output_string_pdm}\n")
            else:
                period_scan_list.append(i_output_string_ls)
                per_search = float(ls_str_list[0])
                i_per_std = 1e-3
                file_perlog.write(f"{i_output_string_ls}\n")
            file_perlog.close()
    else:
        per_search=lcper; i_per_std=1e-4
    # 使用os.system()执行终端命令，生成my_lcs_rel_flux文件
    #i_out_par_file = "./out_par_new/{}".format(ilc)
    #i_out_lcs_file = "./out_lcs_new/{}".format(ilc)
    #command = "cat {} | convexinv -v -o out_areas -p {} input_convexinv {} > /dev/null".format()
    #os.system(command)
    
    hours,col2 = get_lcdata(i_lcs_file)
    plot_phaselc(hours,col2,per_search,title="PDM Phased LC : P =" + f"{per_search:.5f}")
    plot_lc(hours,col2,title="Whole LC")
    
    colors = ['r','y','g','c','b','violet','black','navy','turquoise','magenta','r','y','g','c','b']
    #for i in range(5):
    #    plot_phaselc(hours[i*37:37*(i+1)],col2[i*37:37*(i+1)],per_search,title="PDM Phased LC : P =" + f"{per_search:.5f}",notshow=1)
    
    labels = []  # 用于存储图例标签
    
    for i in range(round(len(hours)/npt)):
        print('per_search=',per_search)
        fig, ax = plt.subplots(figsize=(16, 9))
        
        hours_i = hours[i*npt:npt*(i+1)]
        col2_i = col2[i*npt:npt*(i+1)]
        title="PDM Phased LC : P =" + f"{per_search:.5f}"
        phase = hours_i / per_search % 1
        
        # 绘制之前的数据点
        for j in range(i):
            hours_j = hours[j*npt:npt*(j+1)]
            col2_j = col2[j*npt:npt*(j+1)]
            phase_j = hours_j / per_search % 1
            ax.scatter(phase_j, col2_j, color=colors[j], linewidth=4)
        
        # 绘制当前数据点
        ax.scatter(phase, col2_i, color=colors[i], linewidth=4)
        
        # 添加图例标签
        labels.append(f"Data {i+1}")
        
        ax.set_title(title, fontsize=32, fontweight='bold')
        plt.xlabel('Phase', fontsize=24, fontweight='bold')
        plt.ylabel('Relative flux', fontsize=24, fontweight='bold')
        plt.ylim([0.85,1.1])
        
        # 设置坐标轴刻度和字号
        ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=12)
        ax.spines['top'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        
        # 添加图例
        ax.legend(labels, fontsize=20)
        
        #plt.savefig(f"plot_{i+1}.png")
        plt.show()
    
    print("!!per_search = ",per_search,"  i_per_std = ",i_per_std)
    
    per_std = i_per_std
    
    out_par_sortfile = 'out_par_allsort_'+flag+'_pfix_1deg.txt'
    i_result = global_convexinv_parallel(
        lam_step = lam_step,
        beta_step = beta_step,
        per_input = per_search,
        n_per = n_per,
        per_std = per_std,
        lambdas0 = 0,
        lambdas1 = 180,                              
        betas0 = -30,
        betas1 = 30,                    
        input_convexinv_dir='input_convexinv_new', # genarated input_convexinv files
        out_par_dir='out_par_new',
        out_lcs_dir='out_lcs_new', 
        in_lcs_file = i_lcs_file,
        input_convexinv_file = 'input_convexinv',
        out_par_sortfile = out_par_sortfile)
    
    # 将 ${out_par_sortfile} 保存到 ${out_par_dir}下 
    target_file = 'out_par_new.txt'
    subprocess.run(f"cp {out_par_sortfile} {target_file}", shell=True)
    
    # 定义保存数据的列表
    lam_list.append(float(i_result[0][1]))
    beta_list.append(float(i_result[0][2]))
    per_list.append(float(i_result[0][3]))
    print()
    i_result_arr = np.array([float(num) for num in i_result[0][1:4]])
    print(i_result_arr)
    
    file_outdat = open(file_outdat_name, "a")
    file_outdat.write(f"{per_search:>8.5f}, {i_per_std:>6.0e}, {lam_list[-1]:>8.2f}, {beta_list[-1]:>8.2f}, {per_list[-1]:>8.5f}\n") 
    file_outdat.close()
                
    # 卸载kernel文件
    spice.kclear()

def plot_map_2016(lambda_data, beta_data, title=None):
    
    # 设置经纬度的范围和网格大小
    lam_min = 0; lam_max = 361
    bet_min = -90; bet_max = 91
    lam_min = 0; lam_max = 121
    bet_min = -30; bet_max = 31    
    lambda_bins = np.arange(lam_min, lam_max, 10)  # 经度的范围和网格间隔
    beta_bins = np.arange(bet_min, bet_max, 10)    # 纬度的范围和网格间隔
    
    # 使用numpy的histogram2d函数计算每个网格中的点数
    counts, lambda_edges, beta_edges = np.histogram2d(beta_data, lambda_data, bins=(beta_bins, lambda_bins))
    
    # 将计数转换为概率
    probability = counts / np.sum(counts)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(6, 7))
    
    # 使用imshow绘制概率图，概率值用颜色表示
    c = ax.imshow(probability, extent=[lam_min, lam_max-1, bet_min, bet_max-1], origin='lower', aspect='auto', cmap='Blues')
    
    # 添加颜色条
    #plt.colorbar(c, label='Probability')
    plt.vlines(41.66,bet_min,bet_max-1,color='black')
    plt.hlines(3.15,lam_min,lam_max-1,color='black')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('Lambda (degree)',fontsize=20)
    ax.set_ylabel('Beta (degree)',fontsize=20)
    ax.set_title(title,fontsize=20)
    
    # 设置网格线
    ax.set_xticks(lambda_bins)
    ax.set_yticks(beta_bins)
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_facecolor('lightgrey')
    
    # 设置经度和纬度的tick间隔
    lambda_ticks = np.arange(lam_min, lam_max, 30)
    beta_ticks = np.arange(bet_min, bet_max, 10)
    
    # 设置横轴和纵轴的标签值，每30度一个标签
    ax.set_xticklabels(['{:.0f}°'.format(x) if x in lambda_ticks else '' for x in lambda_bins],fontsize=20)
    ax.set_yticklabels(['{:.0f}°'.format(y) if y in beta_ticks else '' for y in beta_bins],fontsize=20)
    
    # 显示图表
    plt.show()

# 读取文件的前100行数据的前四列
def read_file(file_path,nrows=100):
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2, 3], nrows=nrows)
    df.columns = ["chi2", "lambda", "beta", "per"]
    return df    
# 没改正光行差
#i_lcs_file = 'input_LCS_Bennu_longperiod_2350km.txt' #'Input_LCS_Bennu_correct_7.txt'             
#i_lcs_file = 'Input_LCS_Bennu_correct_5.txt'             

flags = ['4','5','6','7','8','9','10']
os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/cspice_wk/2016/')

if 1:
    for flag in flags:
        out_par_sortfile = 'out_par_allsort_'+flag+'_pfix_1deg.txt'
        df = read_file(out_par_sortfile,nrows=100)
        plot_map_2016(df["lambda"], df["beta"], title='')

if 0:
    for flag in flags:
        #timesleep.sleep(2500)  # 暂停1800秒
        i_lcs_file = 'Input_LCS_Bennu_correct_'+flag+'.txt'
        astinv_1(i_lcs_file,flag)
    
        
