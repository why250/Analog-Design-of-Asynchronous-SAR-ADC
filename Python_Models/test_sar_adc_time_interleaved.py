"""
TimeInterleaved SAR ADC 行为级模型测试脚本
=====================================

本脚本用于测试 `TimeInterleavedSARADC` 的主要功能点，涵盖：
- 基本参数与信息打印
- 通道轮询与单次转换
- 数组转换的交织采样验证
- 整体采样率与时间戳检查
- 正弦波量化示例
- 与单个 IdealSARADC 对比（验证交织提升采样率）
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sar_adc_ideal import IdealSARADC
from sar_adc_time_interleaved import TimeInterleavedSARADC


def test_basic_info():
    """测试基本信息打印"""
    print("=" * 60)
    print("测试 TI-1: 基本参数信息")
    print("=" * 60)

    adc = TimeInterleavedSARADC(
        num_channels=4,
        resolution=10,
        vref_pos=1.0,
        vref_neg=0.0,
        sub_adc_sample_rate_hz=2e6,  # 每个子ADC 2 MS/s
        sub_adc_sar_bit_period_s=4e-9,
        aperture_jitter_rms_s=100e-15,
    )
    adc.print_info()
    print("测试通过！\n")


def test_channel_rotation():
    """测试单次转换的通道轮询与历史记录"""
    print("=" * 60)
    print("测试 TI-2: 通道轮询与历史记录")
    print("=" * 60)

    adc = TimeInterleavedSARADC(
        num_channels=4,
        resolution=8,
        vref_pos=1.0,
        vref_neg=0.0,
        sub_adc_sample_rate_hz=1e6,
    )

    inputs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2]
    print(f"{'样本':<4} {'输入(V)':<10} {'数字码':<8} {'通道':<6} {'采样时间(μs)':<14}")
    print("-" * 60)

    for i, vin in enumerate(inputs):
        code, hist = adc.convert(vin, return_history=True, t0=i * 0.25e-6)
        t_us = hist.get('channel_t0', 0) * 1e6 if hist.get('channel_t0') is not None else 0
        print(f"{i:<4} {vin:<10.4f} {code:<8} {hist['channel_idx']:<6} {t_us:<14.6f}")

    print("\n内部轮询索引: ", adc._current_channel_idx)
    print("测试通过！\n")


def test_array_conversion_interleaving():
    """测试数组转换，并验证时间交织采样时刻"""
    print("=" * 60)
    print("测试 TI-3: 数组转换与采样时刻")
    print("=" * 60)

    num_channels = 4
    fs_sub = 1e6  # 每个子ADC 1 MS/s
    adc = TimeInterleavedSARADC(
        num_channels=num_channels,
        resolution=8,
        vref_pos=1.0,
        vref_neg=0.0,
        sub_adc_sample_rate_hz=fs_sub,
        sub_adc_sar_bit_period_s=5e-9,
    )

    voltages = np.linspace(0.1, 0.9, 12)
    codes, histories = adc.convert_array(voltages, return_history=True, t0=0.0)

    print(f"{'样本':<4} {'输入(V)':<10} {'数字码':<8} {'通道':<6} {'采样时间(μs)':<14}")
    print("-" * 60)
    for i, (vin, code, hist) in enumerate(zip(voltages, codes, histories)):
        t_us = hist['channel_t0'] * 1e6 if hist['channel_t0'] is not None else 0
        print(f"{i:<4} {vin:<10.4f} {code:<8} {hist['channel_idx']:<6} {t_us:<14.6f}")

    # 验证采样时刻差值
    times = np.array([hist['channel_t0'] for hist in histories])
    deltas = np.diff(times)
    if len(deltas) > 0:
        mean_delta = np.mean(deltas)
        expected = 1.0 / (fs_sub * num_channels)
        print(f"\n平均采样间隔: {mean_delta*1e6:.3f} μs (预期: {expected*1e6:.3f} μs)")
        assert np.allclose(mean_delta, expected, atol=expected*0.05), "采样间隔与预期不符"

    print("\n测试通过！\n")


def test_sine_wave_quantization():
    """时间交织ADC对正弦波的量化"""
    print("=" * 60)
    print("测试 TI-4: 正弦波量化")
    print("=" * 60)

    num_channels = 4
    fs_sub = 2e6  # 每个子ADC 2 MS/s
    adc = TimeInterleavedSARADC(
        num_channels=num_channels,
        resolution=10,
        vref_pos=1.0,
        vref_neg=0.0,
        sub_adc_sample_rate_hz=fs_sub,
    )

    fs_overall = fs_sub * num_channels
    t = np.arange(0, 8e-6, 1.0 / fs_overall)
    fin = 0.02 * fs_overall  # 约2% Nyquist
    vmid = 0.5
    amp = 0.4
    vin = vmid + amp * np.sin(2 * np.pi * fin * t)

    codes = adc.convert_array(vin)
    vq = np.array([adc.digital_to_voltage(c) for c in codes])

    plt.figure(figsize=(9, 4))
    plt.plot(t * 1e6, vin, 'k-', label='Input')
    plt.step(t * 1e6, vq, where='post', color='b', label='Quantized')
    plt.xlabel('Time (μs)')
    plt.ylabel('Voltage (V)')
    plt.title('Time-Interleaved SAR ADC Quantization (4-channel, Overall 8 MS/s)')
    plt.legend()
    plt.tight_layout()
    outp = Path(__file__).resolve().parent / "ti_sar_adc_sine_quantization.png"
    plt.savefig(outp, dpi=150)
    plt.close()
    print(f"正弦波量化图已保存到: {outp}")

    print("测试通过！\n")


def test_interleaving_vs_single_adc():
    """比较时间交织ADC与单个IdealSARADC在采样率上的差异"""
    print("=" * 60)
    print("测试 TI-5: 与单个理想ADC的采样率对比")
    print("=" * 60)

    num_channels = 4
    fs_sub = 1e6
    fs_overall = fs_sub * num_channels

    ti_adc = TimeInterleavedSARADC(
        num_channels=num_channels,
        resolution=8,
        vref_pos=1.0,
        vref_neg=0.0,
        sub_adc_sample_rate_hz=fs_sub,
    )

    ideal_adc = IdealSARADC(
        resolution=8,
        vref_pos=1.0,
        vref_neg=0.0,
        sample_rate_hz=fs_overall,
    )

    # 使用相同输入波形
    duration = 3e-6
    t = np.arange(0, duration, 1.0 / fs_overall)
    vin = 0.5 + 0.4 * np.sin(2 * np.pi * 0.25 * fs_overall * t)

    codes_ti = ti_adc.convert_array(vin, t0=0.0)
    codes_ideal = ideal_adc.convert_array(vin)

    diff = np.abs(codes_ti - codes_ideal)
    max_diff = diff.max()

    print(f"最大码差: {max_diff}")
    assert max_diff <= 1, "时间交织ADC与理想ADC输出差异过大"
    print("测试通过！\n")


def main():
    print("\n" + "=" * 60)
    print("时间交织 SAR ADC 行为级模型 - 测试套件")
    print("=" * 60 + "\n")

    try:
        test_basic_info()
        test_channel_rotation()
        test_array_conversion_interleaving()
        test_sine_wave_quantization()
        test_interleaving_vs_single_adc()

        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"\n测试失败: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
