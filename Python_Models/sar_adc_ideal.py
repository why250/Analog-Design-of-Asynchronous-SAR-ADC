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
        vss: float = 0.0,
        *,
        sample_rate_hz: Optional[float] = None,
        sar_bit_period_s: Optional[float] = None,
        aperture_jitter_rms_s: float = 0.0,
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
        sample_rate_hz : float | None, optional
            采样频率（Hz）。若提供，将在历史记录中生成采样时间戳（理想，无抖动）
        sar_bit_period_s : float | None, optional
            SAR 每一位比较的理想时间间隔（秒）。若提供，将在历史记录中生成每位比较时间戳
        aperture_jitter_rms_s : float, optional
            采样孔径抖动RMS（秒）。若提供，将在采样时刻引入Δt并可结合dV/dt估计ΔV
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
        self.sample_rate_hz = sample_rate_hz
        self.sar_bit_period_s = sar_bit_period_s
        self.aperture_jitter_rms_s = float(aperture_jitter_rms_s)
        
        # 内部状态
        self._sampled_voltage = None
        self._conversion_history = []
        self._last_sample_time_s = None
        self._last_aperture_dt_s = 0.0
        self._aperture_dvdt = None
    
    def _compute_timing(self, t0: Optional[float]) -> tuple[Optional[float], Optional[list]]:
        """
        计算理想的采样时间与每位比较时间（若配置了时钟参数）
        
        参数
        ----------
        t0 : float | None
            本次转换起始参考时间（秒）。若为None且配置了sample_rate_hz，则按0计
        
        返回
        ----------
        (t_sample, t_bits) : (float|None, list[float]|None)
            采样时间与各位比较时间（从MSB到LSB）。若未配置相应参数，则返回None
        """
        # 采样时间
        if self.sample_rate_hz is not None and self.sample_rate_hz > 0:
            base_t = 0.0 if t0 is None else float(t0)
            dt = 0.0
            if self.aperture_jitter_rms_s > 0.0:
                dt = float(np.random.default_rng().normal(loc=0.0, scale=self.aperture_jitter_rms_s))
            t_sample = base_t + dt
            self._last_aperture_dt_s = dt
        else:
            t_sample = None
            self._last_aperture_dt_s = 0.0
        
        # 位时间序列（MSB->LSB）
        if self.sar_bit_period_s is not None and self.sar_bit_period_s > 0:
            if t_sample is None:
                base = 0.0 if t0 is None else float(t0)
            else:
                base = t_sample
            t_bits = [base + (i + 1) * self.sar_bit_period_s for i in range(self.resolution)]
        else:
            t_bits = None
        
        return t_sample, t_bits
    
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
    
    def _synchronous_conversion(self, vin: float, t_bits: Optional[list] = None, t_sample: Optional[float] = None) -> tuple[int, list]:
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
        for step_idx, bit_pos in enumerate(range(self.resolution - 1, -1, -1)):
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
                'final_code': digital_code,
                # 时间戳：若提供时钟配置，则记录本位比较时间与采样时间
                't_bit_s': None if t_bits is None else t_bits[step_idx],
                't_sample_s': t_sample
            })
        
        return digital_code, conversion_history
    
    def convert(self, vin: float, return_history: bool = False, t0: Optional[float] = None, dVdt: Optional[float] = None) -> Union[int, tuple[int, list]]:
        """
        执行一次ADC转换
        
        参数
        ----------
        vin : float
            输入模拟电压
        return_history : bool, optional
            是否返回转换历史记录，默认值为False
        t0 : float | None, optional
            转换起始参考时间（秒）。仅用于在历史中标注理想时间戳（本理想模型不改变数值结果）
        
        返回
        ----------
        int 或 tuple[int, list]
            如果return_history=False，返回数字码（整数形式）
            如果return_history=True，返回(digital_code, history)元组
        """
        # 计算理想时序（可选）
        t_sample, t_bits = self._compute_timing(t0)
        self._last_sample_time_s = t_sample
        self._aperture_dvdt = dVdt
        
        # 采样保持
        sampled = self._sample_and_hold(vin)
        # 采样孔径抖动电压校正（若提供dV/dt）
        if self.aperture_jitter_rms_s > 0.0 and self._aperture_dvdt is not None:
            sampled = sampled + float(self._aperture_dvdt) * float(self._last_aperture_dt_s)
        self._sampled_voltage = sampled
        
        # 执行转换
        digital_code, history = self._synchronous_conversion(sampled, t_bits=t_bits, t_sample=t_sample)
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
    # 打开时钟时间戳示例
    adc_time = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0, sample_rate_hz=1e6, sar_bit_period_s=5e-9)
    digital_code, history = adc_time.convert(test_voltage, return_history=True, t0=0.0)
    print(f"输入电压: {test_voltage:.6f} V")
    print(f"最终数字码: {digital_code}")
    print("\n转换过程:")
    print(f"{'位':<4} {'测试码':<8} {'DAC电压(V)':<12} {'比较结果':<8} {'最终码':<8} {'t_sample(s)':<14} {'t_bit(s)':<14}")
    print("-" * 50)
    for step in history:
        print(f"{step['bit']:<4} {step['test_code']:<8} {step['vdac']:<12.6f} "
              f"{'Vin>Vdac' if step['comparator_result'] else 'Vin<=Vdac':<8} {step['final_code']:<8} "
              f"{(step.get('t_sample_s') if step.get('t_sample_s') is not None else float('nan')):<14.3e} "
              f"{(step.get('t_bit_s') if step.get('t_bit_s') is not None else float('nan')):<14.3e}")
    
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

