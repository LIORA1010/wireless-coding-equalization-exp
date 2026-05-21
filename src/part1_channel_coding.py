"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)


def hamming74_encode(bits):
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # 将输入比特 reshape 为 (-1, 4)，每行是一个 4 位信息块
    blocks = bits.reshape(-1, 4)
    # 矩阵乘法 c = u * G，在 GF(2) 上取模 2
    encoded = (blocks @ HAMMING_G) % 2
    # 展平为一维数组返回
    return encoded.flatten()


def hamming74_syndrome(codewords):
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    # 计算伴随式 s = r * H^T mod 2
    syndromes = (codewords @ HAMMING_H.T) % 2
    return syndromes


def hamming74_decode(received):
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    # 复制一份，避免修改原始输入
    blocks = received.reshape(-1, 7).copy()

    # 计算每个码字的伴随式
    syndromes = hamming74_syndrome(blocks)

    # 对每个码字进行纠错
    for i, syndrome in enumerate(syndromes):
        # 若伴随式全为 0，无错误，跳过
        if not np.any(syndrome):
            continue
        # 将伴随式与 H 的每一列逐列比较，找到匹配的错误位置
        for col_idx in range(7):
            if np.array_equal(syndrome, HAMMING_H[:, col_idx]):
                # 翻转对应位置的比特
                blocks[i, col_idx] ^= 1
                break

    # 系统码的信息位是每个码字的前 4 位，展平返回
    return blocks[:, :4].flatten()


def convolutional_encode(bits):
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # 生成多项式（以比特向量表示）
    # g1 = [1, 1, 1]，g2 = [1, 0, 1]，约束长度 K=3，记忆长度 m=2
    g1 = np.array([1, 1, 1], dtype=int)
    g2 = np.array([1, 0, 1], dtype=int)

    # 在末尾追加 2 个 0（尾比特），使移位寄存器状态回到全零
    bits_with_tail = np.concatenate([bits, np.zeros(2, dtype=int)])
    n = len(bits_with_tail)

    # 移位寄存器，初始全零，长度为约束长度 K=3
    shift_reg = np.zeros(3, dtype=int)
    output = []

    for b in bits_with_tail:
        # 将新比特移入寄存器（寄存器从左到右：最新→最旧）
        shift_reg = np.roll(shift_reg, 1)
        shift_reg[0] = b

        # 计算两路输出（GF(2) 卷积）
        out1 = int(np.sum(shift_reg * g1) % 2)
        out2 = int(np.sum(shift_reg * g2) % 2)
        output.extend([out1, out2])

    return np.array(output, dtype=int)


def viterbi_decode_hard(received_bits):
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。
    使用汉明距离作为路径度量。
    """
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    # 生成多项式
    g1 = np.array([1, 1, 1], dtype=int)
    g2 = np.array([1, 0, 1], dtype=int)

    # 状态数：约束长度 K=3，记忆长度 m=2，共 2^2=4 个状态
    num_states = 4
    num_steps = len(received_bits) // 2

    # 初始化：路径度量（累积汉明距离），从全零状态出发
    INF = float('inf')
    pm = [INF] * num_states   # 当前步的路径度量
    pm[0] = 0                 # 初始状态 00，度量为 0
    survivors = [[0]] + [[] for _ in range(num_states - 1)]  # 幸存路径

    def get_output(state, input_bit):
        """根据当前状态和输入比特，计算两路输出比特。"""
        # 状态用 2 位表示：高位为 s1，低位为 s2
        s1 = (state >> 1) & 1
        s2 = state & 1
        reg = np.array([input_bit, s1, s2], dtype=int)
        out1 = int(np.sum(reg * g1) % 2)
        out2 = int(np.sum(reg * g2) % 2)
        return out1, out2

    def next_state(state, input_bit):
        """根据当前状态和输入比特，计算下一状态。"""
        s1 = (state >> 1) & 1
        # 新状态：input_bit 移入最高位，s1 成为次高位
        return ((input_bit << 1) | s1)

    # Viterbi 前向递推
    for step in range(num_steps):
        rx = received_bits[step * 2: step * 2 + 2]
        new_pm = [INF] * num_states
        new_survivors = [[] for _ in range(num_states)]

        for state in range(num_states):
            if pm[state] == INF:
                continue
            for input_bit in [0, 1]:
                out1, out2 = get_output(state, input_bit)
                # 汉明距离：与接收符号的差异
                hamming_dist = int(out1 != rx[0]) + int(out2 != rx[1])
                ns = next_state(state, input_bit)
                candidate = pm[state] + hamming_dist
                if candidate < new_pm[ns]:
                    new_pm[ns] = candidate
                    new_survivors[ns] = survivors[state] + [input_bit]

        pm = new_pm
        survivors = new_survivors

    # 回溯：选取终止状态 0（全零）的幸存路径
    # 若尾比特有效，终止状态应为 0；否则选路径度量最小的状态
    best_state = int(np.argmin(pm))
    decoded = np.array(survivors[best_state], dtype=int)

    # 去除末尾 2 个尾比特
    if len(decoded) >= 2:
        decoded = decoded[:-2]

    return decoded


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')


if __name__ == '__main__':
    run_coding_demo()