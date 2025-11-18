# 使用python搭建sar adc行为级模型
_Exported on 11/7/2025 at 16:22:38 GMT+8 from Cursor (2.0.34)_

---

**User**

我想使用python进行sar adc行为级的建模,请用python搭建一个理想SAR ADC行为级模型.这个模型会封装在一个类（Class）中，使其易于使用和扩展。

---

**Cursor**


查看当前项目结构，然后创建理想 SAR ADC 的行为级模型类。



检查 `Python_Models` 目录，然后创建理想 SAR ADC 的行为级模型类。



创建理想 SAR ADC 的行为级模型类，包含采样保持、逐次逼近转换、理想比较器和理想 DAC。


```python
"""
理想SAR ADC行为级模型
====================

这个模块实现了一个理想的逐次逼近寄存器（SAR）模数转换器（ADC）的行为级模型。
该模型封装在一个类中，易于使用和扩展。

主要特性：
- 可配置的分辨率（位数）
- 可配置的输入电压范围
- 理想采样保持
- 理想比较器
- 理想DAC
- 逐次逼近转换算法

作者：Auto
日期：2024
"""

import numpy as np
from typing import Union, Optional


class IdealSARADC:
    """
    理想SAR ADC行为级模型类
    
    该类实现了一个理想的SAR ADC，包括采样保持、逐次逼近转换过程、
    理想比较器和理想DAC。
    
    参数
    ----------
    resolution : int, optional
        ADC分辨率（位数），默认值为8
    vref_pos : float, optional
        正参考电压（V），默认值为1.0
    vref_neg : float, optional
        负参考电压（V），默认值为0.0
    vdd : float, optional
        电源电压（V），用于数字输出，默认值为1.0
    vss : float, optional
        地电压（V），用于数字输出，默认值为0.0
    
    属性
    ----------
    resolution : int
        ADC分辨率（位数）
    vref_pos : float
        正参考电压
    vref_neg : float
        负参考电压
    vref_range : float
        参考电压范围（vref_pos - vref_neg）
    lsb : float
        最低有效位对应的电压值
    vdd : float
        电源电压
    vss : float
        地电压
    
    示例
    --------
    >>> # 创建一个8位SAR ADC
    >>> adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    >>> 
    >>> # 转换一个模拟电压值
    >>> digital_code = adc.convert(0.5)
    >>> print(f"数字码: {digital_code}")
    >>> 
    >>> # 转换多个电压值
    >>> voltages = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> codes = adc.convert_array(voltages)
    >>> print(f"数字码数组: {codes}")
    """
    
    def __init__(
        self,
        resolution: int = 8,
        vref_pos: float = 1.0,
        vref_neg: float = 0.0,
        vdd: float = 1.0,
        vss: float = 0.0
    ):
        """
        初始化理想SAR ADC
        
        参数
        ----------
        resolution : int, optional
            ADC分辨率（位数），默认值为8
        vref_pos : float, optional
            正参考电压（V），默认值为1.0
        vref_neg : float, optional
            负参考电压（V），默认值为0.0
        vdd : float, float, optional
            电源电压（V），用于数字输出，默认值为1.0
        vss : float, optional
            地电压（V），用于数字输出，默认值为0.0
        """
        # 参数验证
        if resolution <= 0:
            raise ValueError("分辨率必须大于0")
        if vref_pos <= vref_neg:
            raise ValueError("正参考电压必须大于负参考电压")
        
        # 存储参数
        self.resolution = resolution
        self.vref_pos = vref_pos
        self.vref_neg = vref_neg
        self.vref_range = vref_pos - vref_neg
        self.lsb = self.vref_range / (2 ** resolution)
        self.vdd = vdd
        self.vss = vss
        
        # 内部状态
        self._sampled_voltage = None
        self._conversion_history = []
    
    def _ideal_comparator(self, vin: float, vdac: float) -> bool:
        """
        理想比较器
        
        比较输入电压和DAC输出电压
        
        参数
        ----------
        vin : float
            输入电压
        vdac : float
            DAC输出电压
        
        返回
        ----------
        bool
            True表示vin > vdac，False表示vin <= vdac
        """
        return vin > vdac
    
    def _ideal_dac(self, digital_code: int) -> float:
        """
        理想DAC
        
        将数字码转换为模拟电压
        
        参数
        ----------
        digital_code : int
            数字码（整数形式）
        
        返回
        ----------
        float
            对应的模拟电压值
        """
        # 将数字码转换为电压
        # 公式：Vout = Vref_neg + (digital_code / 2^N) * (Vref_pos - Vref_neg)
        voltage = self.vref_neg + (digital_code / (2 ** self.resolution)) * self.vref_range
        return voltage
    
    def _sample_and_hold(self, vin: float) -> float:
        """
        理想采样保持
        
        采样输入电压并保持（理想情况下无失真）
        
        参数
        ----------
        vin : float
            输入电压
        
        返回
        ----------
        float
            采样保持后的电压值
        """
        # 理想采样保持：直接返回输入电压
        # 在实际模型中，可以添加采样噪声、保持误差等非理想因素
        return vin
    
    def _synchronous_conversion(self, vin: float) -> tuple[int, list]:
        """
        同步SAR转换过程
        
        执行逐次逼近转换算法
        
        参数
        ----------
        vin : float
            输入电压（采样后的值）
        
        返回
        ----------
        tuple[int, list]
            数字码（整数形式）和转换历史记录
        """
        # 限制输入电压范围
        vin_clamped = np.clip(vin, self.vref_neg, self.vref_pos)
        
        # 初始化数字码
        digital_code = 0
        conversion_history = []
        
        # 逐次逼近：从MSB到LSB
        for bit_pos in range(self.resolution - 1, -1, -1):
            # 设置当前位为1进行测试
            test_code = digital_code | (1 << bit_pos)
            
            # 通过DAC生成测试电压
            vdac = self._ideal_dac(test_code)
            
            # 比较器比较
            comp_result = self._ideal_comparator(vin_clamped, vdac)
            
            # 根据比较结果决定该位的值
            if comp_result:
                digital_code = test_code  # vin > vdac，保留该位为1
            # 否则该位保持为0（已经在digital_code中）
            
            # 记录转换历史
            conversion_history.append({
                'bit': bit_pos,
                'test_code': test_code,
                'vdac': vdac,
                'comparator_result': comp_result,
                'final_code': digital_code
            })
        
        return digital_code, conversion_history
    
    def convert(self, vin: float, return_history: bool = False) -> Union[int, tuple[int, list]]:
        """
        执行一次ADC转换
        
        参数
        ----------
        vin : float
            输入模拟电压
        return_history : bool, optional
            是否返回转换历史记录，默认值为False
        
        返回
        ----------
        int 或 tuple[int, list]
            如果return_history=False，返回数字码（整数形式）
            如果return_history=True，返回(digital_code, history)元组
        """
        # 采样保持
        sampled = self._sample_and_hold(vin)
        self._sampled_voltage = sampled
        
        # 执行转换
        digital_code, history = self._synchronous_conversion(sampled)
        self._conversion_history = history
        
        if return_history:
            return digital_code, history
        else:
            return digital_code
    
    def convert_array(self, voltages: np.ndarray, return_history: bool = False) -> Union[np.ndarray, tuple[np.ndarray, list]]:
        """
        转换电压数组
        
        参数
        ----------
        voltages : np.ndarray
            输入电压数组
        return_history : bool, optional
            是否返回转换历史记录，默认值为False
        
        返回
        ----------
        np.ndarray 或 tuple[np.ndarray, list]
            如果return_history=False，返回数字码数组
            如果return_history=True，返回(digital_codes, histories)元组
        """
        voltages = np.asarray(voltages)
        digital_codes = np.zeros_like(voltages, dtype=int)
        histories = []
        
        for i, vin in enumerate(voltages):
            if return_history:
                code, history = self.convert(vin, return_history=True)
                digital_codes[i] = code
                histories.append(history)
            else:
                digital_codes[i] = self.convert(vin)
        
        if return_history:
            return digital_codes, histories
        else:
            return digital_codes
    
    def digital_to_voltage(self, digital_code: int) -> float:
        """
        将数字码转换为对应的模拟电压值
        
        参数
        ----------
        digital_code : int
            数字码（整数形式）
        
        返回
        ----------
        float
            对应的模拟电压值
        """
        return self._ideal_dac(digital_code)
    
    def voltage_to_digital(self, voltage: float) -> int:
        """
        将模拟电压值转换为数字码（理想量化）
        
        参数
        ----------
        voltage : float
            模拟电压值
        
        返回
        ----------
        int
            对应的数字码
        """
        # 限制电压范围
        voltage_clamped = np.clip(voltage, self.vref_neg, self.vref_pos)
        
        # 理想量化：直接计算对应的数字码
        normalized = (voltage_clamped - self.vref_neg) / self.vref_range
        digital_code = int(np.round(normalized * (2 ** self.resolution - 1)))
        
        # 限制在有效范围内
        digital_code = np.clip(digital_code, 0, 2 ** self.resolution - 1)
        
        return digital_code
    
    def get_info(self) -> dict:
        """
        获取ADC参数信息
        
        返回
        ----------
        dict
            包含ADC参数的字典
        """
        return {
            'resolution': self.resolution,
            'vref_pos': self.vref_pos,
            'vref_neg': self.vref_neg,
            'vref_range': self.vref_range,
            'lsb': self.lsb,
            'max_code': 2 ** self.resolution - 1,
            'vdd': self.vdd,
            'vss': self.vss
        }
    
    def print_info(self):
        """
        打印ADC参数信息
        """
        info = self.get_info()
        print("=" * 50)
        print("理想SAR ADC参数信息")
        print("=" * 50)
        print(f"分辨率:           {info['resolution']} 位")
        print(f"正参考电压:       {info['vref_pos']:.6f} V")
        print(f"负参考电压:       {info['vref_neg']:.6f} V")
        print(f"参考电压范围:     {info['vref_range']:.6f} V")
        print(f"LSB:              {info['lsb']:.9f} V")
        print(f"最大数字码:       {info['max_code']}")
        print(f"电源电压:         {info['vdd']:.6f} V")
        print(f"地电压:           {info['vss']:.6f} V")
        print("=" * 50)


def example_usage():
    """
    使用示例
    """
    print("\n" + "=" * 60)
    print("理想SAR ADC行为级模型 - 使用示例")
    print("=" * 60 + "\n")
    
    # 创建8位SAR ADC
    print("1. 创建8位SAR ADC (Vref: 0V - 1V)")
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    adc.print_info()
    
    # 单次转换
    print("\n2. 单次转换示例")
    test_voltage = 0.5
    digital_code = adc.convert(test_voltage)
    reconstructed_voltage = adc.digital_to_voltage(digital_code)
    print(f"输入电压: {test_voltage:.6f} V")
    print(f"数字码:   {digital_code} (二进制: {bin(digital_code)})")
    print(f"重构电压: {reconstructed_voltage:.6f} V")
    print(f"量化误差: {abs(test_voltage - reconstructed_voltage):.9f} V")
    
    # 带历史记录的转换
    print("\n3. 带转换历史记录的转换")
    digital_code, history = adc.convert(test_voltage, return_history=True)
    print(f"输入电压: {test_voltage:.6f} V")
    print(f"最终数字码: {digital_code}")
    print("\n转换过程:")
    print(f"{'位':<4} {'测试码':<8} {'DAC电压(V)':<12} {'比较结果':<8} {'最终码':<8}")
    print("-" * 50)
    for step in history:
        print(f"{step['bit']:<4} {step['test_code']:<8} {step['vdac']:<12.6f} "
              f"{'Vin>Vdac' if step['comparator_result'] else 'Vin<=Vdac':<8} {step['final_code']:<8}")
    
    # 数组转换
    print("\n4. 数组转换示例")
    test_voltages = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    digital_codes = adc.convert_array(test_voltages)
    print(f"{'输入电压(V)':<15} {'数字码':<10} {'重构电压(V)':<15} {'量化误差(V)':<15}")
    print("-" * 60)
    for vin, code in zip(test_voltages, digital_codes):
        vout = adc.digital_to_voltage(code)
        error = abs(vin - vout)
        print(f"{vin:<15.6f} {code:<10} {vout:<15.6f} {error:<15.9f}")
    
    # 不同分辨率的比较
    print("\n5. 不同分辨率比较")
    test_voltage = 0.333
    print(f"输入电压: {test_voltage:.6f} V\n")
    print(f"{'分辨率':<8} {'数字码':<10} {'LSB(V)':<12} {'量化误差(V)':<15}")
    print("-" * 50)
    for res in [4, 6, 8, 10, 12]:
        adc_test = IdealSARADC(resolution=res, vref_pos=1.0, vref_neg=0.0)
        code = adc_test.convert(test_voltage)
        vout = adc_test.digital_to_voltage(code)
        error = abs(test_voltage - vout)
        print(f"{res:<8} {code:<10} {adc_test.lsb:<12.9f} {error:<15.9f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()

```


检查代码是否有问题，并创建一个简单的测试脚本：


```python
"""
SAR ADC行为级模型测试脚本
=======================

这个脚本用于测试理想SAR ADC行为级模型的功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from sar_adc_ideal import IdealSARADC


def test_basic_conversion():
    """测试基本转换功能"""
    print("=" * 60)
    print("测试1: 基本转换功能")
    print("=" * 60)
    
    # 创建8位SAR ADC
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    
    # 测试几个电压值
    test_voltages = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"\n{'输入电压(V)':<15} {'数字码':<10} {'重构电压(V)':<15} {'量化误差(V)':<15}")
    print("-" * 60)
    
    for vin in test_voltages:
        code = adc.convert(vin)
        vout = adc.digital_to_voltage(code)
        error = abs(vin - vout)
        print(f"{vin:<15.6f} {code:<10} {vout:<15.6f} {error:<15.9f}")
    
    print("\n测试通过！\n")


def test_conversion_history():
    """测试转换历史记录"""
    print("=" * 60)
    print("测试2: 转换历史记录")
    print("=" * 60)
    
    adc = IdealSARADC(resolution=4, vref_pos=1.0, vref_neg=0.0)
    vin = 0.625  # 应该得到 10 (二进制: 1010)
    
    code, history = adc.convert(vin, return_history=True)
    
    print(f"\n输入电压: {vin:.6f} V")
    print(f"最终数字码: {code} (二进制: {bin(code)})")
    print("\n转换过程:")
    print(f"{'位':<4} {'测试码':<8} {'DAC电压(V)':<12} {'比较结果':<12} {'最终码':<8}")
    print("-" * 50)
    
    for step in history:
        comp_str = "Vin > Vdac" if step['comparator_result'] else "Vin <= Vdac"
        print(f"{step['bit']:<4} {step['test_code']:<8} {step['vdac']:<12.6f} "
              f"{comp_str:<12} {step['final_code']:<8}")
    
    print("\n测试通过！\n")


def test_array_conversion():
    """测试数组转换"""
    print("=" * 60)
    print("测试3: 数组转换")
    print("=" * 60)
    
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    
    # 生成测试电压数组
    voltages = np.linspace(0, 1, 11)
    codes = adc.convert_array(voltages)
    
    print(f"\n转换了 {len(voltages)} 个电压值")
    print(f"数字码范围: {codes.min()} - {codes.max()}")
    print(f"预期范围: 0 - {2**8 - 1}")
    
    # 验证所有码都在有效范围内
    assert codes.min() >= 0, "数字码不能小于0"
    assert codes.max() <= 2**8 - 1, "数字码不能超过最大值"
    
    print("\n测试通过！\n")


def test_linearity():
    """测试线性度"""
    print("=" * 60)
    print("测试4: 线性度测试")
    print("=" * 60)
    
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    
    # 生成满量程的电压
    voltages = np.linspace(0, 1, 256)
    codes = adc.convert_array(voltages)
    
    # 计算INL和DNL（理想情况下应该为0）
    # 这里只做简单的验证
    expected_codes = np.round(voltages * 255).astype(int)
    
    inl = codes - expected_codes
    max_inl = np.max(np.abs(inl))
    
    print(f"\n最大INL: {max_inl} LSB")
    print(f"理想情况下INL应该为0")
    
    # 理想ADC的INL应该为0或非常接近0
    assert max_inl <= 1, f"INL过大: {max_inl}"
    
    print("\n测试通过！\n")


def test_different_resolutions():
    """测试不同分辨率"""
    print("=" * 60)
    print("测试5: 不同分辨率")
    print("=" * 60)
    
    test_voltage = 0.333
    resolutions = [4, 6, 8, 10, 12]
    
    print(f"\n输入电压: {test_voltage:.6f} V\n")
    print(f"{'分辨率':<8} {'数字码':<10} {'LSB(V)':<12} {'量化误差(V)':<15}")
    print("-" * 50)
    
    for res in resolutions:
        adc = IdealSARADC(resolution=res, vref_pos=1.0, vref_neg=0.0)
        code = adc.convert(test_voltage)
        vout = adc.digital_to_voltage(code)
        error = abs(test_voltage - vout)
        print(f"{res:<8} {code:<10} {adc.lsb:<12.9f} {error:<15.9f}")
    
    print("\n测试通过！\n")


def test_voltage_range():
    """测试电压范围限制"""
    print("=" * 60)
    print("测试6: 电压范围限制")
    print("=" * 60)
    
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    
    # 测试超出范围的电压
    test_voltages = [-0.1, 0.0, 0.5, 1.0, 1.1]
    
    print(f"\n{'输入电压(V)':<15} {'数字码':<10} {'重构电压(V)':<15}")
    print("-" * 50)
    
    for vin in test_voltages:
        code = adc.convert(vin)
        vout = adc.digital_to_voltage(code)
        print(f"{vin:<15.6f} {code:<10} {vout:<15.6f}")
    
    # 验证超出范围的电压被限制
    assert adc.convert(-0.1) == 0, "负电压应该被限制为0"
    assert adc.convert(1.1) == 255, "超出范围的电压应该被限制为最大值"
    
    print("\n测试通过！\n")


def plot_transfer_characteristic():
    """绘制传输特性曲线"""
    print("=" * 60)
    print("测试7: 绘制传输特性曲线")
    print("=" * 60)
    
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    
    # 生成输入电压
    voltages = np.linspace(0, 1, 1000)
    codes = adc.convert_array(voltages)
    
    # 绘制传输特性
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, codes, 'b-', linewidth=1.5, label='传输特性')
    plt.xlabel('输入电压 (V)', fontsize=12)
    plt.ylabel('数字码', fontsize=12)
    plt.title('理想SAR ADC传输特性 (8位)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    output_file = 'Python_Models/sar_adc_transfer_characteristic.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n传输特性曲线已保存到: {output_file}")
    
    plt.close()
    print("测试通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SAR ADC行为级模型 - 测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_basic_conversion()
        test_conversion_history()
        test_array_conversion()
        test_linearity()
        test_different_resolutions()
        test_voltage_range()
        plot_transfer_characteristic()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

```


创建 README 说明文档，并运行测试验证功能：


```markdown
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

```


运行测试验证功能：


