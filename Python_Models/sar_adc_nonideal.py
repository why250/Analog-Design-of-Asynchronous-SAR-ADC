"""
非理想SAR ADC行为级模型
=======================

本模块提供 `NonIdealSARADC` 类，在 `IdealSARADC` 基础上引入多种非理想因素：

1. **采样保持非理想性**
   - 采样噪声（kT/C噪声）
   - 保持误差（电压衰减）
   - 采样时间限制

2. **比较器非理想性**
   - 失调电压（固定和随机）
   - 噪声
   - 延迟
   - 有限增益

3. **DAC非理想性**
   - 非线性（INL/DNL）
   - 失调
   - 增益误差
   - 单调性误差

4. **量化非理想性**
   - 量化噪声
   - 非线性误差

使用方式保持与 `IdealSARADC` 一致，便于切换。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sar_adc_ideal import IdealSARADC


class NonIdealSARADC(IdealSARADC):
    """
    非理想SAR ADC行为级模型类

    参数
    ----------
    resolution : int
        分辨率（位数）
    vref_pos, vref_neg : float
        参考电压正端/负端
    vdd, vss : float
        数字域电源与地（信息用途）
    
    # 采样保持非理想性
    sampling_noise_std : float, optional
        采样噪声标准差（单位：V），在采样保持阶段加到输入上，默认0
    hold_droop_rate : float, optional
        保持阶段电压衰减率（V/s），模拟电荷泄漏，默认0
    sampling_time_error : float, optional
        采样时间不足导致的误差（相对值，0-1），默认0
    
    # 比较器非理想性
    comparator_offset : float, optional
        比较器固定失调（单位：V），默认0
    comparator_offset_std : float, optional
        比较器失调的随机标准差（单位：V），每次比较都会独立采样一个偏移并叠加在固定失调上，默认0
    comparator_delay : float, optional
        比较器延迟（单位：s），默认0（行为级模型通常忽略）
    comparator_gain : float, optional
        比较器有限增益（无量纲），默认无穷大（理想）。有限增益会导致比较阈值模糊
    
    # DAC非理想性
    dac_inl_lut : np.ndarray | None, optional
        长度为 2^N 的INL查找表，单位为LSB。索引为数字码，值为该码的INL（可为正负），默认None
    cap_mismatch_sigma : float, optional
        单位电容相对标准差 sigma(ΔC/C)。若提供且未提供 `dac_inl_lut`，将基于二进制加权电容阵列生成一个静态DAC误差表，默认0
    dac_offset : float, optional
        DAC失调（单位：V），默认0
    dac_gain_error : float, optional
        DAC增益误差（相对值，例如0.01表示1%增益误差），默认0
    enforce_dac_monotonicity : bool, optional
        是否强制DAC单调性（修正非单调码），默认False
    
    # 量化非理想性
    quantization_noise_std : float, optional
        额外的量化噪声标准差（单位：LSB），在最终输出码上添加，默认0
    
    rng : np.random.Generator | None, optional
        随机数发生器；若为None，则内部创建一个默认生成器
    """

    def __init__(
        self,
        resolution: int = 8,
        vref_pos: float = 1.0,
        vref_neg: float = 0.0,
        vdd: float = 1.0,
        vss: float = 0.0,
        *,
        # 时钟与时序
        sample_rate_hz: Optional[float] = None,
        sar_bit_period_s: Optional[float] = None,
        aperture_jitter_rms_s: float = 0.0,
        bit_jitter_rms_s: float = 0.0,
        dac_settle_time_s: float = 0.0,
        comp_regen_time_s: float = 0.0,
        # 采样保持非理想性
        sampling_noise_std: float = 0.0,
        hold_droop_rate: float = 0.0,
        sampling_time_error: float = 0.0,
        # 比较器非理想性
        comparator_offset: float = 0.0,
        comparator_offset_std: float = 0.0,
        comparator_delay: float = 0.0,
        comparator_gain: float = float('inf'),
        # DAC非理想性
        dac_inl_lut: Optional[np.ndarray] = None,
        cap_mismatch_sigma: float = 0.0,
        dac_offset: float = 0.0,
        dac_gain_error: float = 0.0,
        enforce_dac_monotonicity: bool = False,
        # 量化非理想性
        quantization_noise_std: float = 0.0,
        # 其他
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(
            resolution=resolution,
            vref_pos=vref_pos,
            vref_neg=vref_neg,
            vdd=vdd,
            vss=vss,
        )

        # 采样保持非理想性
        self.sampling_noise_std = float(sampling_noise_std)
        self.hold_droop_rate = float(hold_droop_rate)
        self.sampling_time_error = float(sampling_time_error)
        
        # 比较器非理想性
        self.comparator_offset = float(comparator_offset)
        self.comparator_offset_std = float(comparator_offset_std)
        self.comparator_delay = float(comparator_delay)
        self.comparator_gain = float(comparator_gain) if comparator_gain != float('inf') else float('inf')
        
        # DAC非理想性
        self.dac_offset = float(dac_offset)
        self.dac_gain_error = float(dac_gain_error)
        self.enforce_dac_monotonicity = bool(enforce_dac_monotonicity)
        
        # 量化非理想性
        self.quantization_noise_std = float(quantization_noise_std)
        
        # 随机数生成器
        self._rng = rng if rng is not None else np.random.default_rng()
        
        # 时钟/时序参数
        self.sample_rate_hz = sample_rate_hz
        self.sar_bit_period_s = sar_bit_period_s
        self.aperture_jitter_rms_s = float(aperture_jitter_rms_s)
        self.bit_jitter_rms_s = float(bit_jitter_rms_s)
        self.dac_settle_time_s = float(dac_settle_time_s)
        self.comp_regen_time_s = float(comp_regen_time_s)
        
        # 采样时间（用于模拟采样时间限制）
        self._sampling_time = 0.0
        self._conversion_start_time = 0.0

        # 构建DAC误差表（单位：V），长度为2^N，对应每个码的静态电压误差
        self._dac_error_volts = self._build_dac_error_table(
            dac_inl_lut=dac_inl_lut,
            cap_mismatch_sigma=cap_mismatch_sigma,
        )
        
        # 如果启用单调性强制，修正DAC误差表
        if self.enforce_dac_monotonicity and self._dac_error_volts is not None:
            self._dac_error_volts = self._enforce_monotonicity(self._dac_error_volts)

    def _compute_timing(self, t0: Optional[float]) -> tuple[Optional[float], Optional[list]]:
        """
        计算带抖动的采样时间与每位比较时间（若配置时钟参数）
        """
        # 采样时间（带孔径抖动）
        if self.sample_rate_hz is not None and self.sample_rate_hz > 0:
            base = 0.0 if t0 is None else float(t0)
            jitter = 0.0
            if self.aperture_jitter_rms_s > 0.0:
                jitter = self._rng.normal(loc=0.0, scale=self.aperture_jitter_rms_s)
            t_sample = base + float(jitter)
            self._last_aperture_dt_s = float(jitter)
        else:
            t_sample = None
            self._last_aperture_dt_s = 0.0
        
        # 位比较时间（带每位抖动）
        if self.sar_bit_period_s is not None and self.sar_bit_period_s > 0:
            base = 0.0 if t0 is None else float(t0)
            if t_sample is not None:
                base = t_sample
            t_bits = []
            for i in range(self.resolution):
                t_nom = base + (i + 1) * self.sar_bit_period_s
                if self.bit_jitter_rms_s > 0.0:
                    t_nom += self._rng.normal(loc=0.0, scale=self.bit_jitter_rms_s)
                t_bits.append(t_nom)
        else:
            t_bits = None
        
        return t_sample, t_bits

    # ---------- 非理想组成部分 ----------
    def _sample_and_hold(self, vin: float) -> float:
        """
        采样保持，包含采样噪声、保持误差和采样时间限制
        """
        # 1. 采样噪声（kT/C噪声）
        if self.sampling_noise_std > 0.0:
            noise = self._rng.normal(loc=0.0, scale=self.sampling_noise_std)
        else:
            noise = 0.0
        
        sampled = vin + noise
        
        # 2. 采样时间限制（采样时间不足导致的误差）
        if self.sampling_time_error > 0.0:
            # 模拟采样时间不足：输入未完全建立
            error = self.sampling_time_error * (vin - sampled) * self._rng.uniform(0.0, 1.0)
            sampled += error
        
        # 3. 保持误差（电压衰减，在转换过程中累积）
        # 这里简化处理：在每次转换开始时应用一个固定的衰减
        if self.hold_droop_rate > 0.0:
            # 假设转换时间为常数，衰减 = rate * time
            # 简化：使用一个典型转换时间（例如1us）
            typical_conversion_time = 1e-6  # 1us
            droop = self.hold_droop_rate * typical_conversion_time
            sampled -= droop
        
        return sampled

    def _nonideal_comparator(self, vin: float, vdac: float) -> bool:
        """
        非理想比较器，包含失调、噪声、有限增益
        """
        # 1. 失调 = 固定偏置 + 随机偏置（若配置）
        if self.comparator_offset_std > 0.0:
            random_offset = self._rng.normal(loc=0.0, scale=self.comparator_offset_std)
        else:
            random_offset = 0.0
        total_offset = self.comparator_offset + random_offset
        
        # 2. 有限增益效应：比较器增益有限时，输出不是理想的阶跃函数
        # 简化模型：当输入差很小时，比较结果可能不确定
        input_diff = vin - vdac - total_offset
        
        if self.comparator_gain != float('inf') and abs(input_diff) < (self.vref_range / self.comparator_gain):
            # 在模糊区域内，比较结果可能随机
            # 使用sigmoid函数模拟有限增益的比较器
            threshold_voltage = self.vref_range / self.comparator_gain
            prob_high = 1.0 / (1.0 + np.exp(-input_diff / (threshold_voltage / 3.0)))
            return self._rng.random() < prob_high
        
        # 3. 理想比较（或增益足够大）
        return input_diff > 0.0

        # 核心定义（Logistic 函数）：σ(x) = 1 / (1 + e^(−x))；取值范围：(0, 1)，当 x→+∞ 时趋近 1，x→−∞ 时趋近 0
        # 将实数映射到概率区间 (0,1)，常用于概率建模/软阈值
        # 在本模型中的作用：用 σ(x) 将比较器的“输入差”映射为“输出为高”的概率；当输入差很小（有限增益下的模糊区），用平滑的概率过渡替代硬阈值
    def _vdac_nonideal(self, digital_code: int) -> float:
        """
        非理想DAC，包含非线性、失调、增益误差
        """
        # 1. 理想DAC输出
        ideal = super()._ideal_dac(digital_code)
        
        # 2. 非线性误差（INL/DNL）
        nonlin_error = 0.0
        if self._dac_error_volts is not None:
            nonlin_error = self._dac_error_volts[int(digital_code)]
        
        # 3. 增益误差
        gain_error = ideal * self.dac_gain_error
        
        # 4. 失调
        offset_error = self.dac_offset
        
        # 5. 总输出
        return ideal + nonlin_error + gain_error + offset_error
    
    def _enforce_monotonicity(self, error_table: np.ndarray) -> np.ndarray:
        """
        强制DAC单调性：确保随着数字码增加，DAC输出电压单调递增
        """
        num_codes = len(error_table)
        corrected = error_table.copy()
        
        # 计算每个码的理想电压
        ideal_voltages = np.array([
            super()._ideal_dac(code) for code in range(num_codes)
        ])
        
        # 计算每个码的实际电压（理想 + 误差）
        actual_voltages = ideal_voltages + corrected
        
        # 修正非单调性：如果某个码的电压小于前一个码，则调整误差使其等于前一个码
        for i in range(1, num_codes):
            if actual_voltages[i] < actual_voltages[i-1]:
                # 调整误差，使当前码的电压等于前一个码
                corrected[i] = corrected[i-1] + (ideal_voltages[i] - ideal_voltages[i-1])
                actual_voltages[i] = ideal_voltages[i] + corrected[i]
        
        return corrected

    def _build_dac_error_table(
        self,
        *,
        dac_inl_lut: Optional[np.ndarray],
        cap_mismatch_sigma: float,
    ) -> Optional[np.ndarray]:
        num_codes = 2 ** self.resolution

        # 1) 若提供INL查找表（单位：LSB），直接转换为电压误差
        if dac_inl_lut is not None:
            lut = np.asarray(dac_inl_lut, dtype=float)
            if lut.shape[0] != num_codes:
                raise ValueError("dac_inl_lut 长度应为 2^resolution")
            # 转换为电压误差：INL(code) [LSB] * LSB[V]
            return lut * self.lsb

        # 2) 否则若给出电容失配标准差，生成一次静态误差表（简化二进制加权阵列模型）
        if cap_mismatch_sigma > 0.0:
            # 生成每一位的等效电容：C_b = 2^b * C_unit * (1 + delta_b)
            # 其中 delta_b ~ N(0, sigma)
            bit_weights_nominal = 2.0 ** np.arange(self.resolution - 1, -1, -1)  # MSB -> LSB
            deltas = self._rng.normal(loc=0.0, scale=cap_mismatch_sigma, size=self.resolution)
            bit_caps = bit_weights_nominal * (1.0 + deltas)
            c_sum = np.sum(bit_caps)

            errors = np.zeros(num_codes, dtype=float)
            for code in range(num_codes):
                bits = [(code >> k) & 1 for k in range(self.resolution - 1, -1, -1)]
                weighted_sum = float(np.dot(bits, bit_caps))
                norm_nonideal = weighted_sum / c_sum
                norm_ideal = code / (num_codes)
                errors[code] = (norm_nonideal - norm_ideal) * self.vref_range
            # 固定端点（可选）：令0码与满码误差为0，去除整体漂移
            errors[0] = 0.0
            errors[-1] = 0.0
            return errors

        # 3) 否则无DAC误差
        return None

    # ---------- 覆盖转换流程：使用非理想比较器与DAC ----------
    def _synchronous_conversion(self, vin: float, *, t_sample: Optional[float], t_bits: Optional[list]) -> tuple[int, list]:
        vin_clamped = np.clip(vin, self.vref_neg, self.vref_pos)
        digital_code = 0
        conversion_history = []

        prev_t = t_sample
        for step_idx, bit_pos in enumerate(range(self.resolution - 1, -1, -1)):
            test_code = digital_code | (1 << bit_pos)
            vdac = self._vdac_nonideal(test_code)
            
            # 计算时间裕量并引入不足时的不确定性
            t_bit = None if t_bits is None else t_bits[step_idx]
            time_since_prev = None
            settle_ok = None
            settle_deficit = None
            comp_result = self._nonideal_comparator(vin_clamped, vdac)
            if t_bit is not None and prev_t is not None:
                time_since_prev = max(0.0, t_bit - prev_t)
                required = max(self.dac_settle_time_s, self.comp_regen_time_s)
                if required > 0.0:
                    settle_ok = time_since_prev >= required
                    if not settle_ok:
                        deficit = max(0.0, required - time_since_prev) / required
                        settle_deficit = float(deficit)
                        # 以 deficit 的比例增加误判概率（最多0.5）
                        flip_prob = min(0.5, 0.5 * deficit)
                        if self._rng.random() < flip_prob:
                            comp_result = not comp_result
                prev_t = t_bit
            else:
                prev_t = t_bit if t_bit is not None else prev_t
            
            if comp_result:
                digital_code = test_code
            conversion_history.append({
                'bit': bit_pos,
                'test_code': test_code,
                'vdac': vdac,
                'comparator_result': comp_result,
                'final_code': digital_code,
                't_sample_s': t_sample,
                't_bit_s': t_bit,
                'time_since_prev_s': time_since_prev,
                'settle_ok': settle_ok,
                'settle_deficit': settle_deficit,
            })

        # 量化噪声：在最终输出码上添加噪声
        if self.quantization_noise_std > 0.0:
            noise_lsb = self._rng.normal(loc=0.0, scale=self.quantization_noise_std)
            digital_code = int(np.round(digital_code + noise_lsb))
            digital_code = np.clip(digital_code, 0, 2 ** self.resolution - 1)

        return digital_code, conversion_history

    def convert(self, vin: float, return_history: bool = False, t0: Optional[float] = None, dVdt: Optional[float] = None) -> tuple[int, list] | int:
        """
        执行一次ADC转换（带可选时序/抖动建模与时间戳记录）
        """
        # 计算时序（含抖动）
        t_sample, t_bits = self._compute_timing(t0)
        
        # 采样保持（已含采样噪声/保持误差/时间误差等）
        sampled = self._sample_and_hold(vin)
        # 采样孔径抖动电压校正（若提供dV/dt）
        if self.aperture_jitter_rms_s > 0.0 and dVdt is not None:
            sampled = sampled + float(dVdt) * float(getattr(self, "_last_aperture_dt_s", 0.0))
        self._sampled_voltage = sampled
        
        # 逐次逼近（带时序/不足稳定性带来的随机性）
        digital_code, history = self._synchronous_conversion(sampled, t_sample=t_sample, t_bits=t_bits)
        self._conversion_history = history
        
        if return_history:
            return digital_code, history
        return digital_code

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            'model': 'NonIdealSARADC',
            # 时钟/时序
            'sample_rate_hz': self.sample_rate_hz,
            'sar_bit_period_s': self.sar_bit_period_s,
            'aperture_jitter_rms_s': self.aperture_jitter_rms_s,
            'bit_jitter_rms_s': self.bit_jitter_rms_s,
            'dac_settle_time_s': self.dac_settle_time_s,
            'comp_regen_time_s': self.comp_regen_time_s,
            # 采样保持非理想性
            'sampling_noise_std': self.sampling_noise_std,
            'hold_droop_rate': self.hold_droop_rate,
            'sampling_time_error': self.sampling_time_error,
            # 比较器非理想性
            'comparator_offset': self.comparator_offset,
            'comparator_offset_std': self.comparator_offset_std,
            'comparator_delay': self.comparator_delay,
            'comparator_gain': self.comparator_gain,
            # DAC非理想性
            'dac_offset': self.dac_offset,
            'dac_gain_error': self.dac_gain_error,
            'enforce_dac_monotonicity': self.enforce_dac_monotonicity,
            'has_dac_error_table': self._dac_error_volts is not None,
            # 量化非理想性
            'quantization_noise_std': self.quantization_noise_std,
        })
        return info


def _example_usage():
    print("\n" + "=" * 60)
    print("非理想SAR ADC行为级模型 - 使用示例")
    print("=" * 60 + "\n")

    # 示例0：含时钟与抖动的基础示例
    print("示例0: 含时钟与抖动（时间戳与稳定性标注）")
    adc0 = NonIdealSARADC(
        resolution=8, vref_pos=1.0, vref_neg=0.0,
        sample_rate_hz=1e6,          # 1 MS/s
        sar_bit_period_s=5e-9,       # 每位5ns
        aperture_jitter_rms_s=200e-15,  # 200 fs
        bit_jitter_rms_s=1e-12,         # 1 ps
        dac_settle_time_s=3e-9,         # 3 ns
        comp_regen_time_s=2e-9,         # 2 ns
    )
    code0, hist0 = adc0.convert(0.5, return_history=True, t0=0.0)
    print(f"  输入: 0.5V, 输出码: {code0}")
    print("  首几步时间与稳定性标注:")
    print(f"  {'bit':<4} {'t_sample(s)':<14} {'t_bit(s)':<14} {'settle_ok':<10} {'deficit':<10}")
    for step in hist0[:4]:
        print(f"  {step['bit']:<4} "
              f"{(step.get('t_sample_s') if step.get('t_sample_s') is not None else float('nan')):<14.3e} "
              f"{(step.get('t_bit_s') if step.get('t_bit_s') is not None else float('nan')):<14.3e} "
              f"{str(step.get('settle_ok')):<10} "
              f"{(step.get('settle_deficit') if step.get('settle_deficit') is not None else float('nan')):<10.3e}")

    # 示例1：采样保持非理想性
    print("示例1: 采样保持非理想性（噪声、保持误差、采样时间限制）")
    adc1 = NonIdealSARADC(
        resolution=8, vref_pos=1.0, vref_neg=0.0,
        sampling_noise_std=200e-6,
        hold_droop_rate=1e3,  # 1mV/ms
        sampling_time_error=0.01,  # 1%误差
    )
    code1 = adc1.convert(0.5)
    print(f"  输入: 0.5V, 输出码: {code1}")

    # 示例2：比较器非理想性
    print("\n示例2: 比较器非理想性（失调、噪声、有限增益）")
    adc2 = NonIdealSARADC(
        resolution=8, vref_pos=1.0, vref_neg=0.0,
        comparator_offset=0.5e-3,
        comparator_offset_std=0.2e-3,
        comparator_gain=1000.0,  # 有限增益
    )
    code2 = adc2.convert(0.5)
    print(f"  输入: 0.5V, 输出码: {code2}")

    # 示例3：DAC非理想性（INL查表）
    print("\n示例3: DAC非线性（INL查表）")
    lut = np.zeros(2 ** 8)
    lut[128:] = 0.3  # 半程后 +0.3 LSB 偏差
    adc3 = NonIdealSARADC(
        resolution=8, vref_pos=1.0, vref_neg=0.0,
        dac_inl_lut=lut,
        dac_offset=1e-3,  # DAC失调
        dac_gain_error=0.01,  # 1%增益误差
    )
    code3 = adc3.convert(0.5)
    print(f"  输入: 0.5V, 输出码: {code3}")

    # 示例4：DAC非理想性（电容失配 + 单调性强制）
    print("\n示例4: DAC非线性（电容失配 + 单调性强制）")
    adc4 = NonIdealSARADC(
        resolution=8, vref_pos=1.0, vref_neg=0.0,
        cap_mismatch_sigma=0.002,
        enforce_dac_monotonicity=True,
    )
    code4 = adc4.convert(0.5)
    print(f"  输入: 0.5V, 输出码: {code4}")

    # 示例5：综合非理想性
    print("\n示例5: 综合非理想性")
    adc5 = NonIdealSARADC(
        resolution=8, vref_pos=1.0, vref_neg=0.0,
        sampling_noise_std=150e-6,
        comparator_offset=0.6e-3,
        comparator_offset_std=0.2e-3,
        cap_mismatch_sigma=0.0015,
        dac_offset=0.5e-3,
        dac_gain_error=0.005,
        quantization_noise_std=0.1,  # 0.1 LSB量化噪声
    )
    code5 = adc5.convert(0.5)
    print(f"  输入: 0.5V, 输出码: {code5}")
    print(f"  参数信息: {adc5.get_info()}")


if __name__ == "__main__":
    _example_usage()


