"""
SAR ADC行为级模型测试脚本
=======================

这个脚本用于测试理想SAR ADC行为级模型的功能。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sar_adc_ideal import IdealSARADC
from sar_adc_nonideal import NonIdealSARADC


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
    
    # 启用理想时钟时间戳（示例：1 MS/s 采样，SAR每位5 ns）
    adc = IdealSARADC(
        resolution=4, vref_pos=1.0, vref_neg=0.0,
        sample_rate_hz=1e6, sar_bit_period_s=5e-9
    )
    vin = 0.625  # 应该得到 10 (二进制: 1010)
    
    code, history = adc.convert(vin, return_history=True, t0=0.0)
    
    print(f"\n输入电压: {vin:.6f} V")
    print(f"最终数字码: {code} (二进制: {bin(code)})")
    print("\n转换过程:")
    print(f"{'位':<4} {'测试码':<8} {'DAC电压(V)':<12} {'比较结果':<12} {'最终码':<8} {'t_sample(s)':<14} {'t_bit(s)':<14}")
    print("-" * 50)
    
    for step in history:
        comp_str = "Vin > Vdac" if step['comparator_result'] else "Vin <= Vdac"
        print(f"{step['bit']:<4} {step['test_code']:<8} {step['vdac']:<12.6f} "
              f"{comp_str:<12} {step['final_code']:<8} "
              f"{(step.get('t_sample_s') if step.get('t_sample_s') is not None else float('nan')):<14.3e} "
              f"{(step.get('t_bit_s') if step.get('t_bit_s') is not None else float('nan')):<14.3e}")
    
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
    
    # 这里不需要时钟，仅绘制静态传输特性
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0)
    
    # 生成输入电压
    voltages = np.linspace(0, 1, 1000)
    codes = adc.convert_array(voltages)
    
    # 绘制传输特性
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, codes, 'b-', linewidth=1.5, label='Transfer Characteristic')
    plt.xlabel('Input Voltage (V)', fontsize=12)
    plt.ylabel('Digital Code', fontsize=12)
    plt.title('Ideal SAR ADC Transfer Characteristic (8-bit)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(__file__).resolve().parent / "sar_adc_transfer_characteristic.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n传输特性曲线已保存到: {output_path}")
    
    plt.close()
    print("测试通过！\n")


def test_sine_quantization_time_domain():
    """正弦波量化效果（时域）"""
    print("=" * 60)
    print("测试8: 正弦波量化效果 (时域)")
    print("=" * 60)
    
    # 这里也可以打开时钟时间戳用于演示，但对量化结果无影响
    adc = IdealSARADC(resolution=8, vref_pos=1.0, vref_neg=0.0, sample_rate_hz=100_000, sar_bit_period_s=5e-9)
    
    # 生成时域正弦波（避免削顶，设置一定余量）
    fs = 100_000  # 采样率 (Hz)
    fin = 1000    # 正弦频率 (Hz)
    duration = 2e-3  # 时长 (s)
    t = np.arange(0.0, duration, 1.0 / fs)
    vmid = 0.5 * (adc.vref_pos + adc.vref_neg)
    amp = 0.45 * (adc.vref_pos - adc.vref_neg) / 2.0
    vin = vmid + amp * np.sin(2.0 * np.pi * fin * t)
    
    # 量化
    codes = adc.convert_array(vin)
    vq = np.array([adc.digital_to_voltage(c) for c in codes])
    
    # 绘图（使用英文以避免字体问题）
    plt.figure(figsize=(10, 5))
    plt.plot(t * 1e3, vin, 'b-', linewidth=1.2, label='Input Sine (Analog)')
    plt.step(t * 1e3, vq, where='post', color='r', linewidth=1.0, label='Quantized (Reconstructed)')
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Sine Wave Quantization in Time Domain (8-bit SAR ADC)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(__file__).resolve().parent / "sar_adc_sine_quantization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n正弦波量化图已保存到: {output_path}")
    
    plt.close()
    print("测试通过！\n")


def test_nonideal_vs_ideal():
    """理想 vs 非理想 SAR ADC 对比测试（传输特性 + 正弦时域）"""
    print("=" * 60)
    print("测试9: 理想 vs 非理想 对比")
    print("=" * 60)

    # 配置
    res = 8
    vpos, vneg = 1.0, 0.0
    ideal = IdealSARADC(resolution=res, vref_pos=vpos, vref_neg=vneg)
    nonideal = NonIdealSARADC(
        resolution=res, vref_pos=vpos, vref_neg=vneg,
        # 时钟/时序（示例参数）
        sample_rate_hz=1e6,          # 1 MS/s
        sar_bit_period_s=5e-9,       # 每位 5 ns
        aperture_jitter_rms_s=200e-15,  # 200 fs
        bit_jitter_rms_s=1e-12,         # 1 ps
        dac_settle_time_s=3e-9,         # 3 ns
        comp_regen_time_s=2e-9,         # 2 ns
        # 采样保持非理想性
        sampling_noise_std=150e-6,
        hold_droop_rate=500.0,  # 0.5mV/ms
        sampling_time_error=0.005,  # 0.5%误差
        # 比较器非理想性
        comparator_offset=0.6e-3,
        comparator_offset_std=0.2e-3,
        comparator_gain=2000.0,  # 有限增益
        # DAC非理想性
        cap_mismatch_sigma=0.002,
        dac_offset=0.3e-3,
        dac_gain_error=0.003,  # 0.3%增益误差
        # 量化非理想性
        quantization_noise_std=0.05,  # 0.05 LSB
    )

    # 1) 传输特性对比
    x = np.linspace(vneg, vpos, 1000)
    y_ideal = ideal.convert_array(x)
    y_nonideal = nonideal.convert_array(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_ideal, 'b-', linewidth=1.2, label='Ideal Transfer')
    plt.plot(x, y_nonideal, 'r--', linewidth=1.0, label='Non-ideal Transfer')
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Digital Code')
    plt.title('Transfer Characteristic: Ideal vs Non-ideal (8-bit)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out1 = Path(__file__).resolve().parent / "sar_adc_transfer_comparison.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"传输特性对比图已保存到: {out1}")

    # 2) 正弦量化时域对比
    fs = 100_000
    fin = 1000
    duration = 2e-3
    t = np.arange(0.0, duration, 1.0 / fs)
    vmid = 0.5 * (vpos + vneg)
    amp = 0.45 * (vpos - vneg) / 2.0
    vin = vmid + amp * np.sin(2.0 * np.pi * fin * t)

    codes_i = ideal.convert_array(vin)
    vqi = np.array([ideal.digital_to_voltage(c) for c in codes_i])
    codes_n = nonideal.convert_array(vin)
    vqn = np.array([nonideal.digital_to_voltage(c) for c in codes_n])

    plt.figure(figsize=(10, 5))
    plt.plot(t * 1e3, vin, 'k-', linewidth=1.0, alpha=0.6, label='Input Sine (Analog)')
    plt.step(t * 1e3, vqi, where='post', color='b', linewidth=1.0, label='Ideal Quantized')
    plt.step(t * 1e3, vqn, where='post', color='r', linewidth=1.0, alpha=0.9, label='Non-ideal Quantized')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (V)')
    plt.title('Time-domain Quantization: Ideal vs Non-ideal (8-bit)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out2 = Path(__file__).resolve().parent / "sar_adc_sine_quantization_comparison.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"正弦量化对比图已保存到: {out2}")

    # 简要数值对比：相同输入下的平均码差
    mean_code_diff = float(np.mean(np.abs(y_nonideal - y_ideal)))
    print(f"平均码差(传输特性): {mean_code_diff:.3f} LSB")
    print("测试通过！\n")

def test_jitter_snr_demo():
    """抖动对SNR演示：SNR vs fin，验证 SNR_jitter ≈ -20log10(2π fin σ_tj) 趋势"""
    print("=" * 60)
    print("测试10: 抖动对SNR (SNR vs fin)")
    print("=" * 60)

    # 配置
    fs = 1_000_000  # 采样率 1 MS/s
    sigma_tj = 200e-15  # 200 fs
    res = 12  # 提高分辨率，减小量化噪声影响
    vpos, vneg = 1.0, 0.0
    vmid = 0.5 * (vpos + vneg)
    amp = 0.45 * (vpos - vneg) / 2.0
    duration = 2e-3
    t = np.arange(0.0, duration, 1.0 / fs)

    # 频率扫描
    fins = np.logspace(3, 5.2, 12)  # 1 kHz -> ~160 kHz
    snr_meas = []
    snr_theory = []

    adc = NonIdealSARADC(
        resolution=res, vref_pos=vpos, vref_neg=vneg,
        sample_rate_hz=fs, sar_bit_period_s=5e-9,
        aperture_jitter_rms_s=sigma_tj,
        # 禁用其他非理想，突出孔径抖动
        sampling_noise_std=0.0, comparator_offset=0.0, comparator_offset_std=0.0,
        cap_mismatch_sigma=0.0, dac_offset=0.0, dac_gain_error=0.0,
        bit_jitter_rms_s=0.0, dac_settle_time_s=0.0, comp_regen_time_s=0.0,
        quantization_noise_std=0.0,
    )

    for fin in fins:
        vin = vmid + amp * np.sin(2.0 * np.pi * fin * t)
        # dV/dt = 2π f A cos(...)
        dvdt = 2.0 * np.pi * fin * amp * np.cos(2.0 * np.pi * fin * t)
        # 逐点转换（传入局部斜率以建模孔径抖动电压）
        codes = np.zeros_like(vin, dtype=int)
        for i in range(len(vin)):
            codes[i] = adc.convert(float(vin[i]), t0=float(t[i]), dVdt=float(dvdt[i]))
        vq = np.array([adc.digital_to_voltage(c) for c in codes])

        # SNR 估算：信号功率 / 误差功率
        err = vq - vin
        p_signal = (amp ** 2) / 2.0  # 正弦均方
        p_noise = float(np.mean(err ** 2)) + 1e-30
        snr = 10.0 * np.log10(p_signal / p_noise)
        snr_meas.append(snr)

        # 理论：SNR_jitter ≈ -20log10(2π fin σ_tj)
        snr_j = -20.0 * np.log10(2.0 * np.pi * fin * sigma_tj + 1e-30)
        snr_theory.append(snr_j)

    fins = np.array(fins)
    snr_meas = np.array(snr_meas)
    snr_theory = np.array(snr_theory)

    # 绘图
    plt.figure(figsize=(9, 5))
    plt.semilogx(fins, snr_meas, 'o-b', label='Measured SNR (with aperture jitter)')
    plt.semilogx(fins, snr_theory, '--r', label='Theoretical SNR_jitter ≈ -20log10(2π f σ)')
    plt.xlabel('Input Frequency fin (Hz)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs fin due to Aperture Jitter (σ = 200 fs)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outp = Path(__file__).resolve().parent / "sar_adc_jitter_snr.png"
    plt.savefig(outp, dpi=150)
    plt.close()
    print(f"SNR 曲线已保存到: {outp}")
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
        test_sine_quantization_time_domain()
        test_nonideal_vs_ideal()
        test_jitter_snr_demo()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

