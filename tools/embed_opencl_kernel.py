# tools/embed_opencl_kernel.py
#!/usr/bin/env python3
"""
将OpenCL kernel文件(.cl)转换为C++字符串字面量
用法: python embed_opencl_kernel.py input.cl output.h
"""

import sys
import os
import re

def cl_to_cpp_string(input_file, output_file):
    """将.cl文件转换为C++字符串"""
    
    # 读取.cl文件
    with open(input_file, 'r', encoding='utf-8') as f:
        cl_content = f.read()
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 生成C++头文件内容
    header_content = f"""// Auto-generated from {os.path.basename(input_file)}
#pragma once

#include <string>

namespace powerserve::opencl::embedded {{

const std::string {base_name}_cl_source = R"CLC(
{cl_content}
)CLC";

}} // namespace powerserve::opencl::embedded
"""
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header_content)
    
    print(f"Generated {output_file} from {input_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python embed_opencl_kernel.py <input.cl> <output.h>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    cl_to_cpp_string(input_file, output_file)

if __name__ == "__main__":
    main()