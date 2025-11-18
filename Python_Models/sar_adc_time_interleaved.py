"""
时间交织SAR ADC行为级模型
========================

本模块提供 `TimeInterleavedSARADC` 类，基于 `IdealSARADC` 实现时间交织架构。
时间交织ADC使用多个子ADC通道，每个通道在不同时间采样，然后交错输出，
以提高整体采样率。

主要特性：
- 多个子ADC通道（通常为2的幂次）
- 时间交错采样
- 整体采样率 = 子ADC采样率 × 通道数
- 理想情况下无通道失配

作者：Auto
日期：2024
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from sar_adc_ideal import IdealSARADC


class TimeInterleavedSARADC:
    """
    时间交织SAR ADC行为级模型类
    
    该类使用多个 `IdealSARADC` 实例作为子通道，实现时间交织采样。
    每个通道按顺序在不同时间采样，然后交错输出数字码。
    
    参数
    ----------
    num_channels : int
        子ADC通道数量（通常为2的幂次，如2、4、8等）
    resolution : int, optional
        每个子ADC的分辨率（位数），默认值为8
    vref_pos : float, optional
        正参考电压（V），默认值为1.0
    vref_neg : float, optional
        负参考电压（V），默认值为0.0
    vdd : float, optional
        电源电压（V），用于数字输出，默认值为1.0
    vss : float, optional
        地电压（V），用于数字输出，默认值为0.0
    sub_adc_sample_rate_hz : float, optional
        每个子ADC的采样频率（Hz）。若提供，整体采样率 = sub_adc_sample_rate_hz × num_channels
    sub_adc_sar_bit_period_s : float, optional
        每个子ADC的SAR位周期（秒）
    aperture_jitter_rms_s : float, optional
        采样孔径抖动RMS（秒），默认值为0.0
    
    属性
    ----------
    num_channels : int
        子ADC通道数量
    resolution : int
        每个子ADC的分辨率
    overall_sample_rate_hz : float | None
        整体采样率（Hz）= sub_adc_sample_rate_hz × num_channels
    sub_adc_sample_rate_hz : float | None
        每个子ADC的采样频率
    channels : list[IdealSARADC]
        子ADC通道列表
    
    示例
    --------
    >>> # 创建4通道时间交织SAR ADC
    >>> adc = TimeInterleavedSARADC(
    ...     num_channels=4,
    ...     resolution=8,
    ...     sub_adc_sample_rate_hz=1e6  # 每个子ADC 1 MS/s
    ... )
    >>> # 整体采样率 = 4 MS/s
    >>> 
    >>> # 转换单个电压值（使用通道0）
    >>> code = adc.convert(0.5)
    >>> 
    >>> # 转换电压数组（自动交错使用各通道）
    >>> voltages = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> codes = adc.convert_array(voltages)
    """
    
    def __init__(
        self,
        num_channels: int,
        resolution: int = 8,
        vref_pos: float = 1.0,
        vref_neg: float = 0.0,
        vdd: float = 1.0,
        vss: float = 0.0,
        *,
        sub_adc_sample_rate_hz: Optional[float] = None,
        sub_adc_sar_bit_period_s: Optional[float] = None,
        aperture_jitter_rms_s: float = 0.0,
    ):
        """
        初始化时间交织SAR ADC
        
        参数
        ----------
        num_channels : int
            子ADC通道数量（通常为2的幂次）
        resolution : int, optional
            每个子ADC的分辨率（位数），默认值为8
        vref_pos : float, optional
            正参考电压（V），默认值为1.0
        vref_neg : float, optional
            负参考电压（V），默认值为0.0
        vdd : float, optional
            电源电压（V），用于数字输出，默认值为1.0
        vss : float, optional
            地电压（V），用于数字输出，默认值为0.0
        sub_adc_sample_rate_hz : float, optional
            每个子ADC的采样频率（Hz）
        sub_adc_sar_bit_period_s : float, optional
            每个子ADC的SAR位周期（秒）
        aperture_jitter_rms_s : float, optional
            采样孔径抖动RMS（秒），默认值为0.0
        """
        # 参数验证
        if num_channels <= 0:
            raise ValueError("通道数量必须大于0")
        if resolution <= 0:
            raise ValueError("分辨率必须大于0")
        if vref_pos <= vref_neg:
            raise ValueError("正参考电压必须大于负参考电压")
        
        # 存储参数
        self.num_channels = num_channels
        self.resolution = resolution
        self.vref_pos = vref_pos
        self.vref_neg = vref_neg
        self.vref_range = vref_pos - vref_neg
        self.lsb = self.vref_range / (2 ** resolution)
        self.vdd = vdd
        self.vss = vss
        self.sub_adc_sample_rate_hz = sub_adc_sample_rate_hz
        self.sub_adc_sar_bit_period_s = sub_adc_sar_bit_period_s
        self.aperture_jitter_rms_s = aperture_jitter_rms_s
        
        # 计算整体采样率
        if sub_adc_sample_rate_hz is not None:
            self.overall_sample_rate_hz = sub_adc_sample_rate_hz * num_channels
        else:
            self.overall_sample_rate_hz = None
        
        # 创建子ADC通道
        self.channels: list[IdealSARADC] = []
        for i in range(num_channels):
            channel = IdealSARADC(
                resolution=resolution,
                vref_pos=vref_pos,
                vref_neg=vref_neg,
                vdd=vdd,
                vss=vss,
                sample_rate_hz=sub_adc_sample_rate_hz,
                sar_bit_period_s=sub_adc_sar_bit_period_s,
                aperture_jitter_rms_s=aperture_jitter_rms_s,
            )
            self.channels.append(channel)
        
        # 当前通道索引（用于单次转换）
        self._current_channel_idx = 0
        
        # 转换历史（记录每个样本使用的通道）
        self._conversion_history = []
    
    def convert(self, vin: float, channel_idx: Optional[int] = None, t0: Optional[float] = None, dVdt: Optional[float] = None, return_history: bool = False) -> Union[int, tuple[int, dict]]:
        """
        执行一次ADC转换
        
        参数
        ----------
        vin : float
            输入模拟电压
        channel_idx : int, optional
            指定使用的通道索引。若为None，则使用当前通道并自动递增
        t0 : float, optional
            采样时刻（秒）。若提供，将用于时间戳记录
        dVdt : float, optional
            输入电压变化率（V/s），用于孔径抖动校正
        return_history : bool, optional
            是否返回转换历史记录，默认值为False
        
        返回
        ----------
        int 或 tuple[int, dict]
            如果return_history=False，返回数字码（整数形式）
            如果return_history=True，返回(digital_code, history)元组
        """
        # 选择通道
        if channel_idx is None:
            channel_idx = self._current_channel_idx
            self._current_channel_idx = (self._current_channel_idx + 1) % self.num_channels
        
        if channel_idx < 0 or channel_idx >= self.num_channels:
            raise ValueError(f"通道索引必须在 [0, {self.num_channels-1}] 范围内")
        
        # 计算该通道的采样时刻（时间交织）
        # 注意：如果t0是从convert_array传入的，它已经包含了正确的采样时刻
        # 只有在单次调用convert时，才需要添加通道相位偏移
        if t0 is not None and self.overall_sample_rate_hz is not None:
            # 对于convert_array调用：t0已经是正确的采样时刻，直接使用
            # 对于单次convert调用：需要添加通道相位偏移
            # 我们通过检查调用栈或使用更简单的方法：
            # 如果channel_idx是显式指定的（不是自动轮换的），说明是单次调用，需要加偏移
            # 但更简单的方法是：convert_array传入的t0已经是正确的，直接使用
            # 单次convert时，如果用户没有指定t0，或者t0是基准时刻，需要加偏移
            # 为了简化，我们假设：如果t0是整数倍的采样间隔，说明是从convert_array来的
            sample_interval = 1.0 / self.overall_sample_rate_hz
            t0_normalized = t0 / sample_interval
            is_integer_multiple = abs(t0_normalized - round(t0_normalized)) < 1e-6
            
            if is_integer_multiple:
                # t0已经是采样时刻（从convert_array传入），直接使用
                channel_t0 = t0
            else:
                # 单次调用，需要添加通道相位偏移
                channel_phase_offset = channel_idx / self.overall_sample_rate_hz
                channel_t0 = t0 + channel_phase_offset
        else:
            channel_t0 = t0
        
        # 使用选定的通道进行转换
        channel = self.channels[channel_idx]
        result = channel.convert(vin, return_history=return_history, t0=channel_t0, dVdt=dVdt)
        
        # 构建历史记录
        if return_history:
            if isinstance(result, tuple):
                code, sub_history = result
            else:
                code = result
                sub_history = None
            
            history = {
                'channel_idx': channel_idx,
                'digital_code': code,
                'sub_adc_history': sub_history,
                't0': t0,
                'channel_t0': channel_t0,
            }
            self._conversion_history.append(history)
            return code, history
        else:
            history = {
                'channel_idx': channel_idx,
                'digital_code': result,
                't0': t0,
            }
            self._conversion_history.append(history)
            return result
    
    def convert_array(self, voltages: np.ndarray, t0: Optional[float] = None, dVdt: Optional[float] = None, return_history: bool = False) -> Union[np.ndarray, tuple[np.ndarray, list]]:
        """
        转换电压数组（自动时间交织）
        
        参数
        ----------
        voltages : np.ndarray
            输入电压数组
        t0 : float, optional
            第一个样本的采样时刻（秒）
        dVdt : float | np.ndarray, optional
            输入电压变化率（V/s）。可以是标量或与voltages同长度的数组
        return_history : bool, optional
            是否返回转换历史记录，默认值为False
        
        返回
        ----------
        np.ndarray 或 tuple[np.ndarray, list]
            如果return_history=False，返回数字码数组
            如果return_history=True，返回(digital_codes, histories)元组
        """
        voltages = np.asarray(voltages)
        num_samples = len(voltages)
        digital_codes = np.zeros(num_samples, dtype=int)
        histories = []
        
        # 处理dVdt
        if dVdt is not None:
            dVdt_array = np.asarray(dVdt)
            if dVdt_array.shape != voltages.shape:
                if dVdt_array.size == 1:
                    dVdt_array = np.full_like(voltages, float(dVdt))
                else:
                    raise ValueError("dVdt必须与voltages同长度或为标量")
        else:
            dVdt_array = None
        
        # 计算采样时刻（如果提供t0和采样率）
        if t0 is not None and self.overall_sample_rate_hz is not None:
            sample_times = t0 + np.arange(num_samples) / self.overall_sample_rate_hz
        else:
            sample_times = None
        
        # 时间交织转换：每个样本使用不同的通道
        for i, vin in enumerate(voltages):
            channel_idx = i % self.num_channels
            
            # 获取采样时刻（已经是正确的时刻，不需要再加通道偏移）
            if sample_times is not None:
                sample_t = sample_times[i]
            else:
                sample_t = None
            
            # 获取dVdt
            if dVdt_array is not None:
                sample_dVdt = dVdt_array[i]
            else:
                sample_dVdt = None
            
            # 直接使用通道进行转换（sample_t已经是正确的采样时刻）
            channel = self.channels[channel_idx]
            if return_history:
                result = channel.convert(vin, return_history=True, t0=sample_t, dVdt=sample_dVdt)
                if isinstance(result, tuple):
                    code, sub_history = result
                else:
                    code = result
                    sub_history = None
                
                history = {
                    'channel_idx': channel_idx,
                    'digital_code': code,
                    'sub_adc_history': sub_history,
                    't0': sample_t,
                    'channel_t0': sample_t,  # 对于convert_array，两者相同
                }
                digital_codes[i] = code
                histories.append(history)
            else:
                code = channel.convert(vin, return_history=False, t0=sample_t, dVdt=sample_dVdt)
                digital_codes[i] = code
        
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
        # 使用第一个通道的DAC（所有通道相同）
        return self.channels[0].digital_to_voltage(digital_code)
    
    def get_info(self) -> dict:
        """
        获取ADC参数信息
        
        返回
        ----------
        dict
            包含ADC参数的字典
        """
        return {
            'model': 'TimeInterleavedSARADC',
            'num_channels': self.num_channels,
            'resolution': self.resolution,
            'vref_pos': self.vref_pos,
            'vref_neg': self.vref_neg,
            'vref_range': self.vref_range,
            'lsb': self.lsb,
            'max_code': 2 ** self.resolution - 1,
            'vdd': self.vdd,
            'vss': self.vss,
            'sub_adc_sample_rate_hz': self.sub_adc_sample_rate_hz,
            'overall_sample_rate_hz': self.overall_sample_rate_hz,
            'sub_adc_sar_bit_period_s': self.sub_adc_sar_bit_period_s,
            'aperture_jitter_rms_s': self.aperture_jitter_rms_s,
        }
    
    def print_info(self):
        """
        打印ADC参数信息
        """
        info = self.get_info()
        print("=" * 60)
        print("时间交织SAR ADC参数信息")
        print("=" * 60)
        print(f"通道数量:           {info['num_channels']}")
        print(f"分辨率:             {info['resolution']} 位/通道")
        print(f"正参考电压:         {info['vref_pos']:.6f} V")
        print(f"负参考电压:         {info['vref_neg']:.6f} V")
        print(f"参考电压范围:       {info['vref_range']:.6f} V")
        print(f"LSB:                {info['lsb']:.9f} V")
        print(f"最大数字码:         {info['max_code']}")
        if info['sub_adc_sample_rate_hz'] is not None:
            print(f"子ADC采样率:        {info['sub_adc_sample_rate_hz']/1e6:.3f} MS/s")
            print(f"整体采样率:         {info['overall_sample_rate_hz']/1e6:.3f} MS/s")
        if info['sub_adc_sar_bit_period_s'] is not None:
            print(f"SAR位周期:          {info['sub_adc_sar_bit_period_s']*1e9:.3f} ns")
        if info['aperture_jitter_rms_s'] > 0:
            print(f"孔径抖动RMS:        {info['aperture_jitter_rms_s']*1e15:.3f} fs")
        print("=" * 60)


def example_usage():
    """
    使用示例
    """
    print("\n" + "=" * 60)
    print("时间交织SAR ADC行为级模型 - 使用示例")
    print("=" * 60 + "\n")
    
    # 创建4通道时间交织SAR ADC
    print("1. 创建4通道时间交织SAR ADC")
    print("   每个子ADC: 8位, 1 MS/s")
    print("   整体采样率: 4 MS/s\n")
    adc = TimeInterleavedSARADC(
        num_channels=4,
        resolution=8,
        vref_pos=1.0,
        vref_neg=0.0,
        sub_adc_sample_rate_hz=1e6,
        sub_adc_sar_bit_period_s=5e-9,
    )
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
    
    # 数组转换（时间交织）
    print("\n3. 数组转换示例（自动时间交织）")
    test_voltages = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6])
    digital_codes, histories = adc.convert_array(test_voltages, return_history=True, t0=0.0)
    print(f"\n转换了 {len(test_voltages)} 个电压值")
    print(f"{'样本':<6} {'输入电压(V)':<15} {'数字码':<10} {'通道':<6} {'采样时刻(μs)':<15}")
    print("-" * 70)
    for i, (vin, code, hist) in enumerate(zip(test_voltages, digital_codes, histories)):
        t_us = hist.get('channel_t0', 0) * 1e6 if hist.get('channel_t0') is not None else 0
        print(f"{i:<6} {vin:<15.6f} {code:<10} {hist['channel_idx']:<6} {t_us:<15.6f}")
    
    # 通道使用统计
    print("\n4. 通道使用统计")
    channel_usage = {}
    for hist in histories:
        ch = hist['channel_idx']
        channel_usage[ch] = channel_usage.get(ch, 0) + 1
    for ch in sorted(channel_usage.keys()):
        print(f"通道 {ch}: {channel_usage[ch]} 次转换")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()

