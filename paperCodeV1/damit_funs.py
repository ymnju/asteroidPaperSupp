#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:30:32 2023

@author: superman
"""

import numpy as np
import subprocess
import sys
import os
import math
from tqdm import tqdm
import concurrent.futures

import glob
import matplotlib.pyplot as plt
from scipy import stats
import pdb

# return chi2, lam,beta,period, 3 phase function parameters, Lambert coefficient, dark facet
# all the *file keyword are in the base path
# all the *dir keyword are created to save necessary results
def global_convexinv_parallel(lam_step = 60,
                              beta_step = 30,
                              per_input = None,
                              n_per = 1,
                              per_std = 1e-5,
                              lambdas0 = 0,
                              lambdas1 = 360,                              
                              betas0 = -90,
                              betas1 = 90,
                              input_convexinv_dir='input_convexinv_new', # genarated input_convexinv files
                              out_par_dir='out_par_new',
                              out_lcs_dir='out_lcs_new', 
                              in_lcs_file = 'test_lcs_rel',
                              input_convexinv_file = 'input_convexinv',
                              out_par_sortfile = 'out_par_allsort.txt'):

    # 读取并修改input_convexinv文件
    os.makedirs(input_convexinv_dir, exist_ok=True)
    #os.system(f'rm {input_convexinv_dir}/*')
    os.makedirs(out_par_dir, exist_ok=True)
    os.makedirs(out_lcs_dir, exist_ok=True)
    
    # 读取原始文件内容
    with open(input_convexinv_file, 'r') as f:
        lines = f.readlines()
    
    period_hours = float(lines[2].split()[0])
    if per_input is not None:
        period_hours = per_input
    
    # 生成新的拟合参数值
    lambdas = np.arange(lambdas0, lambdas1, lam_step, dtype=int)
    
    # 加速 begin
    #lambdas1 = np.arange(40, 81, lam_step, dtype=int)
    #lambdas2 = np.arange(220, 261, lam_step, dtype=int)
    #lambdas = np.concatenate((lambdas1, lambdas2))
    # 加速 end
    
    betas = np.arange(betas0, betas1, beta_step, dtype=int)
    periods = np.around(np.random.normal(period_hours, per_std, size=n_per), decimals=5)
    
    # 保存所有拟合结果
    results = []
    
    # 将列表数据转换为字节流
#    with open(in_lcs_file, 'r') as f:
#        data_str = f.read()

#    in_lcs_bytes = data_str.encode()

    def process_parameters(l, b, p):
        new_lines = [
            f'{l:<4d}\t\t1\tinital lambda [deg] (0/1 - fixed/free)\n',
            f'{b:<4d}\t\t1\tinitial beta [deg] (0/1 - fixed/free)\n',
            f'{p:<7.5f}\t\t1\tinital period [hours] (0/1 - fixed/free)\n'
        ]
        new_content = ''.join(new_lines + lines[3:])
    
        in_par_file = f'./{input_convexinv_dir}/input_convexinv_{l}_{b}_{p}.txt'
        out_par_file = f'./{out_par_dir}/out_par_{l}_{b}_{p}.txt'
        out_lcs_file = f'./{out_lcs_dir}/out_lcs_{l}_{b}_{p}.txt'
        with open(in_par_file, 'w') as f:
            f.write(new_content)
    
        #cmd = f'cat {in_lcs_file} | convexinv -v -p {out_par_file} {in_par_file} {out_lcs_file}'
        #cmd = f'cat {in_lcs_file} | convexinv -v {in_par_file} {out_lcs_file}'
        cmd = f'cat {in_lcs_file} | convexinv_fast -v {in_par_file} {out_lcs_file}'
        result_cv = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    
        if result_cv.returncode != 0:
            return None
    
        output_lines = result_cv.stdout.decode('utf-8').split('\n')
        chi2_elements = None
        lambda_beta_period = None
        phase_function_params = None
        lambert_coefficient = None
        dark_facet_area = None
    
        for line in output_lines:
            if 'chi2' in line:
                chi2_elements = line.split()
            elif 'lambda, beta and period (hrs):' in line:
                lambda_beta_period = line.split(':')[1].split()
            elif 'phase function parameters:' in line:
                phase_function_params = line.split(':')[1].split()
            elif 'Lambert coefficient:' in line:
                lambert_coefficient = line.split(':')[1].split()
            elif 'plus a dark facet with area' in line:
                dark_facet_area = line.split('area')[1].split()
                break
    
        if chi2_elements is None or lambda_beta_period is None or phase_function_params is None or lambert_coefficient is None or dark_facet_area is None:
            print("Error parsing output in iteration ")
            return None
    
        return chi2_elements[2:3] + lambda_beta_period[0:] + phase_function_params[0:] + lambert_coefficient + dark_facet_area
    
    # debug 
    # aa=process_parameters(lambdas[0], betas[0], periods[0])
    
    # 使用线程池进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_results = [executor.submit(process_parameters, l, b, p) for l in lambdas for b in betas for p in periods]
        for future in tqdm(concurrent.futures.as_completed(future_results), total=len(future_results)):
            result = future.result()
            if result is not None:
                results.append(result)
    
    sorted_results = sorted(results, key=lambda x: float(x[0]))
    
    # 将排序后的结果写入文件a.txt中
    with open(out_par_sortfile, 'w') as f:
        for row in sorted_results:
            f.write('\t'.join(row) + '\n')
            
    return sorted_results

# 不同观测距离下小天体星等换算
#   2016-03-08 00:00:00  57455.00000000    23.00  0.19934423053942E+00  53.91015348 
def calculate_m(D_km):
    AU_to_km = 149597870.7  # 1 AU 约等于 149,597,870.7 公里
    D_au = D_km / AU_to_km  # 将距离 D (单位：公里) 转换为天文单位 AU
    d = 0.2  # 已知 d = 0.13 AU
    m = 23  # 已知 d = 0.13 AU 时，M = 22

    M = m + 5 * (math.log10(D_au / d))
    return M

# 文件夹下有很多数据文件，文件名为out_par_allsort{i}.txt，其中i是从0开始的自然数。
# 文件内容是9列数据，前8列是float型，最后一列数据含%。
# 请读取所有out_par_allsort{i}.txt文件的第一行，放到一个数组best_fit_arr中。
# 然后对best_fit_arr的第二列数据，计算3 sigma-clilp后的平均值，并用hist作出直方图。
# 对best_fit_arr的第三列数据，计算3 sigma-clilp后的平均值，并用hist作出直方图。
def plt_global_convexinv_parallel():

    os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/cspice_wk/0.01singleExp1Cad50NewDisChangePhaseAngle6LC')
    
    # 找到所有的文件名
    file_names = glob.glob('out_par_allsort*.txt')
    file_names.sort()
    
    # 使用 Python 内置的 pdb 模块来设置断点和调试程序
    #pdb.set_trace()

    # 读取所有文件的第一行数据
    best_fit_arr = np.zeros((len(file_names), 9))
    for i, file_name in enumerate(file_names):
        with open(file_name, 'r') as f:
            line = f.readline()
            # 使用 rstrip() 函数来去除 line 字符串末尾的换行符、制表符等
            best_fit_arr[i] = np.fromstring(line.rstrip('%\n'), sep='\t')

    
    
    # 对第二列数据进行 3 sigma-clilp 处理，计算平均值
    # 0.03 的含义是去掉数据的前 1.5% 和后 1.5%，也就是说，只使用数据的中间 97% 来计算平均值
    second_col = best_fit_arr[:, 1]
    # 筛选出大于200的元素，并将其减去180
    second_col[second_col > 300] -= 180
    second_col_mean = stats.trim_mean(second_col, 0.4)
    
    # 计算排除上下20%数据后的标准差
    second_col_sorted = sorted(second_col)
    lower_bound = int(len(second_col_sorted) * 0.15)
    upper_bound = int(len(second_col_sorted) * 0.85)
    second_col_trimmed = second_col_sorted[lower_bound:upper_bound]
    second_col_std_trimmed = stats.tstd(second_col_trimmed)

    second_col_clip = second_col[(second_col >= second_col_mean - 5 * second_col_std_trimmed) & (second_col <= second_col_mean + 5 * second_col_std_trimmed)]
    #second_col_clip = np.clip(second_col, second_col_mean - 3 * second_col_std_trimmed, second_col_mean + 3 * second_col_std_trimmed)
    second_col_clip_mean = np.mean(second_col_clip)
    
    # 对第三列数据进行 3 sigma-clilp 处理，计算平均值
    third_col = best_fit_arr[:, 2]
    third_col_mean = stats.trim_mean(third_col, 0.4)
    
    # 计算排除上下10%数据后的标准差
    third_col_sorted = sorted(third_col)
    lower_bound = int(len(third_col_sorted) * 0.15)
    upper_bound = int(len(third_col_sorted) * 0.85)
    third_col_trimmed = third_col_sorted[lower_bound:upper_bound]
    third_col_std_trimmed = stats.tstd(third_col_trimmed)

    third_col_clip = third_col[(third_col >= third_col_mean - 5 * third_col_std_trimmed) & (third_col <= third_col_mean + 5 * third_col_std_trimmed)]
    #third_col_clip = np.clip(third_col, third_col_mean - 3 * third_col_std_trimmed, third_col_mean + 3 * third_col_std_trimmed)
    third_col_clip_mean = np.mean(third_col_clip)

    print()
    print("lam  (sigma-clip): mean = {:.2f}" .format(second_col_clip_mean))
    print("                   sigma = {:.2f}" .format(second_col_std_trimmed))
    print("beta (sigma-clip): mean = {:.2f}" .format(third_col_clip_mean)) 
    print("                   sigma = {:.2f}" .format(third_col_std_trimmed))
    
    # 绘制直方图
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(second_col_clip, bins=15)
    ax[0].set_xlabel('lamda')
    ax[0].set_ylabel('Count')
    ax[0].set_title(f'Mean: {second_col_clip_mean:.2f}')
    ax[1].hist(third_col_clip, bins=15)
    ax[1].set_xlabel('beta')
    ax[1].set_ylabel('Count')
    ax[1].set_title(f'Mean: {third_col_clip_mean:.2f}')
    plt.show()    
#    pdb.set_trace()
        

#os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/DamitTest/')
#global_convexinv_parallel()
#sys.exit(0)

# os.chdir('/Users/superman/Documents/资料/新分类/太阳系小天体/cspice_wk/20469219_hj/')
# results = global_convexinv_parallel(
#             lam_step = 1,
#             beta_step = 1,
#             n_per = 1,
#             per_std = 1e-4,
#             input_convexinv_dir='input_convexinv_new', # genarated input_convexinv files
#             out_par_dir='out_par_new',
#             out_lcs_dir='out_lcs_new', 
#             in_lcs_file = 'B.txt',
#             input_convexinv_file = 'input_convexinv',
#             out_par_sortfile = 'out_par_allsort.txt')
# with open('out_par_allsort_A.txt', 'r') as f:
#     lines = f.readlines()[:10000]
# col1 = [float(row.split()[1]) for row in lines]
# col2 = [float(row.split()[2]) for row in lines]
# plt.hist(col1, bins=50)
# plt.xlim(30,80)
# plt.show()
# plt.hist(col2, bins=50)
# plt.xlim(30,80)
# plt.show()