#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:02:47 2023

@author: superman
"""
import os
import time
import pandas as pd  # 导入pandas库
from multiprocessing import Pool

# input file : input_period_scan 参数
# input data : test_lcs_rel 光变曲线
# period_scan_extend 默认输出读取的参数和求得的结果, 加“> /dev/null”使之不显示
def run_period_scan(input_file, input_data):
    os.system(f"cat {input_data} | period_scan_extend -v {input_file} out_{input_file} > /dev/null")    

def generate_input_files(n, input_file, input_data, p0, p1):
    # Read the input file to get the range of values to compute
    with open(input_file, "r") as f:
        line = f.readline().strip().split()
        a, b = map(float, line[:2])
        interval_coeff = line[-1]
        rest = " ".join(line[2:-1])  # Get the remaining parameters as a string
    if p0 is not None:
        a = float(p0)
    if p1 is not None:
        b = float(p1)    
    # Divide the range into equal-sized segments based on the number of files
    chunk_size = (b - a) / n
    ranges = [(a + i*chunk_size, a + (i+1)*chunk_size) for i in range(n)]
    ranges[-1] = (ranges[-1][0], b)  # Make sure the last range includes the end of the interval
    
    # Generate input files for each range
    for i, (start, end) in enumerate(ranges):
        filename = f"input_period_scan_{i+1}"
        with open(filename, "w") as f:
            f.write(f"{start:.5f} {end:.5f} {rest}\n")
            # Copy the remaining lines from the original input file
            with open(input_file, "r") as orig:
                next(orig)  # Skip the first line
                for line in orig:
                    f.write(line)
        #print(f"Generated {filename}")
        
def merge_output_files(n, output_merge):        
    # Open the output file for writing
    with open(output_merge, "w") as outfile:
        # Loop through all output file indices and append their contents to the output file
        for i in range(1, n+1):
            # Construct the output file name
            output_file = f"out_input_period_scan_{i}"
    
            # Open the output file and append its contents to the output file
            with open(output_file, "r") as infile:
                outfile.write(infile.read())   

def analyze_output_file(output_merge):
    # 读取文件并将其转换为DataFrame对象
    df = pd.read_table(output_merge, sep='\s+', header=None, engine='python')
    
    # 找到第三列最小值所在的行
    min_value = df[2].min()
    min_rows = df.loc[df[2] == min_value]
    
    # 如果有多行满足最小值条件，则选择第二列最小的一行作为返回值
    if len(min_rows) > 1:
        min_row = pd.DataFrame(df.iloc[min_rows[1].idxmin()]).T
    else:
        min_row = min_rows

    # 将最小值所在的行输出到终端
    output_string = min_row.to_string(index=None, header=None)
        
    print("卡方最小的一行：")
    print("period   rms      chi2      iter. dark area %")
    print(output_string)
    
    return output_string
    
  
# p0 and p1 are the range of fitted period    
def period_scan_parallel_V2(n, input_file, input_data, output_merge, p0=None, p1=None):
    generate_input_files(n, input_file, input_data, p0, p1)
        
    # Define the number of parallel processes
    num_processes = n

    # Get a list of input files to process
    input_files = [f"input_period_scan_{i}" for i in range(1, n+1)]

    # Create a process pool and submit each input file as a task
    # [input_data] * n这个表达式是将input_data这个元素重复复制n次，生成一个包含n个input_data元素的列表。
    # *n 的目的是为了确保每个input_files都对应相同的input_data参数
    # 使用下划线 _ 表示一个不需要使用的占位符变量。通过将返回值赋值给_，我们明确表示我们不需要使用该返回值
    with Pool(num_processes) as pool:             
        pool.starmap(run_period_scan, zip(input_files, [input_data] * n))
    
    merge_output_files(n, output_merge)
    output_string = analyze_output_file(output_merge)
    
    return output_string