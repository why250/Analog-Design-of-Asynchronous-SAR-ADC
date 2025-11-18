# SAR ADC行为级模型

这个目录包含了理想SAR ADC（逐次逼近寄存器模数转换器）的Python行为级模型。

## 文件说明

- `sar_adc_ideal.py`: 理想SAR ADC行为级模型类
- `test_sar_adc.py`: 测试脚本，用于验证模型功能

## 快速开始

### 基本使用

```python
import numpy as np
from sar_adc_ideal import IdealSARADC

# 创建一个8位SAR ADC
adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)

# 转换一个模拟电压值
digital_code = adc.convert(0.5)
print(f"数字码: {digital_code}")

# 将数字码转换回电压
voltage = adc.digital_to_voltage(digital_code)
print(f"重构电压: {voltage} V")
```

### 转换电压数组

```python
# 转换多个电压值
voltages = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
codes = adc.convert_array(voltages)
print(f"数字码数组: {codes}")
```

### 查看转换过程

```python
# 获取转换历史记录
digital_code, history = adc.convert(0.5, return_history=True)

# 查看逐次逼近过程
for step in history:
    print(f"位 {step['bit']}: 测试码={step['test_code']}, "
          f"DAC电压={step['vdac']:.6f} V, "
          f"比较结果={'Vin>Vdac' if step['comparator_result'] else 'Vin<=Vdac'}")
```

## 类参数说明

### IdealSARADC类

#### 初始化参数

- `resolution` (int, 默认=8): ADC分辨率（位数）
- `vref_pos` (float, 默认=1.0): 正参考电压（V）
- `vref_neg` (float, 默认=0.0): 负参考电压（V）
- `vdd` (float, 默认=1.0): 电源电压（V），用于数字输出
- `vss` (float, 默认=0.0): 地电压（V），用于数字输出

#### 主要方法

- `convert(vin, return_history=False)`: 执行一次ADC转换
  - `vin`: 输入模拟电压
  - `return_history`: 是否返回转换历史记录
  - 返回: 数字码（整数形式）或 (数字码, 历史记录) 元组

- `convert_array(voltages, return_history=False)`: 转换电压数组
  - `voltages`: 输入电压数组（numpy数组）
  - `return_history`: 是否返回转换历史记录
  - 返回: 数字码数组或 (数字码数组, 历史记录列表) 元组

- `digital_to_voltage(digital_code)`: 将数字码转换为模拟电压
- `voltage_to_digital(voltage)`: 将模拟电压转换为数字码（理想量化）
- `get_info()`: 获取ADC参数信息
- `print_info()`: 打印ADC参数信息

## 运行测试

运行测试脚本来验证模型功能：

```bash
python test_sar_adc.py
```

测试包括：
1. 基本转换功能
2. 转换历史记录
3. 数组转换
4. 线性度测试
5. 不同分辨率测试
6. 电压范围限制测试
7. 传输特性曲线绘制

## 模型特性

### 理想特性

- **理想采样保持**: 无失真采样
- **理想比较器**: 无延迟、无噪声、无失调
- **理想DAC**: 无非线性、无失调
- **理想量化**: 无量化噪声（在行为级模型中）

### 可扩展性

模型设计为易于扩展，可以添加以下非理想因素：

1. **采样保持非理想性**
   - 采样噪声
   - 保持误差
   - 采样时间限制

2. **比较器非理想性**
   - 失调电压
   - 噪声
   - 延迟
   - 有限增益

3. **DAC非理想性**
   - 非线性
   - 失调
   - 增益误差
   - 单调性误差

4. **量化非理想性**
   - 量化噪声
   - 非线性误差

## 示例应用

### 示例1: 正弦波采样

```python
import numpy as np
import matplotlib.pyplot as plt
from sar_adc_ideal import IdealSARADC

# 创建ADC
adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)

# 生成正弦波
t = np.linspace(0, 1, 1000)
freq = 10  # Hz
vin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t)  # 0.1V - 0.9V

# 转换
codes = adc.convert_array(vin)

# 重构
vout = np.array([adc.digital_to_voltage(code) for code in codes])

# 绘制
plt.figure(figsize=(10, 6))
plt.plot(t, vin, 'b-', label='输入信号')
plt.plot(t, vout, 'r--', label='重构信号')
plt.xlabel('时间 (s)')
plt.ylabel('电压 (V)')
plt.title('SAR ADC采样示例')
plt.legend()
plt.grid(True)
plt.show()
```

### 示例2: 不同分辨率的比较

```python
from sar_adc_ideal import IdealSARADC

test_voltage = 0.333
resolutions = [4, 6, 8, 10, 12]

print(f"输入电压: {test_voltage:.6f} V\n")
print(f"{'分辨率':<8} {'数字码':<10} {'LSB(V)':<12} {'量化误差(V)':<15}")
print("-" * 50)

for res in resolutions:
    adc = IdealSARADC(resolution=res, vref_pos=1.0, vref_neg=0.0)
    code = adc.convert(test_voltage)
    vout = adc.digital_to_voltage(code)
    error = abs(test_voltage - vout)
    print(f"{res:<8} {code:<10} {adc.lsb:<12.9f} {error:<15.9f}")
```

## 依赖项

- numpy
- matplotlib (仅用于测试和绘图)

## 作者

Auto - 2024

## 许可证

本项目遵循原项目的许可证。

