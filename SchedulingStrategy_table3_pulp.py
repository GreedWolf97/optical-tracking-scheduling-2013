"""复现论文低轨算例：先最大化得分，再最大化总观测弧长。

python SchedulingStrategy_table3_pulp.py
python SchedulingStrategy_table3_pulp.py --check-paper
python SchedulingStrategy_table3_pulp.py --verbose
"""

import argparse
import sys
from pathlib import Path

# 优先使用算例目录或其上级目录中的 .deps
example_directory = Path(__file__).resolve().parent
local_dependencies = next(
    (directory / '.deps' for directory in (example_directory, *example_directory.parents)
     if (directory / '.deps').is_dir()), None)
if local_dependencies is not None:
    sys.path.insert(0, str(local_dependencies))

import pulp

# -------------------- 1. 论文表 2：输入数据 --------------------
# (弧段号, 目标编号, 可见开始秒数, 可见结束秒数)
arcs = [
    (1, 27464, 12 * 3600 + 47 * 60 + 33, 12 * 3600 + 51 * 60 + 23),
    (2, 25260, 12 * 3600 + 58 * 60 + 46, 13 * 3600 + 2 * 60 + 36),
    (3, 36414, 13 * 3600 + 2 * 60 + 14, 13 * 3600 + 6 * 60 + 4),
    (4, 36413, 13 * 3600 + 2 * 60 + 25, 13 * 3600 + 6 * 60 + 15),
    (5, 36415, 13 * 3600 + 2 * 60 + 41, 13 * 3600 + 6 * 60 + 31),
    (6, 25892, 13 * 3600 + 2 * 60 + 44, 13 * 3600 + 6 * 60 + 34),
    (7, 37386, 13 * 3600 + 6 * 60 + 13, 13 * 3600 + 10 * 60 + 3),
    (8, 27534, 13 * 3600 + 6 * 60 + 45, 13 * 3600 + 9 * 60 + 37),
    (9, 26905, 13 * 3600 + 8 * 60 + 37, 13 * 3600 + 11 * 60 + 5),
    (10, 37210, 13 * 3600 + 9 * 60 + 35, 13 * 3600 + 13 * 60 + 25),
    (11, 27818, 13 * 3600 + 11 * 60 + 18, 13 * 3600 + 15 * 60 + 8),
    (12, 37874, 13 * 3600 + 12 * 60 + 50, 13 * 3600 + 14 * 60 + 32),
]

N_TA = len(arcs)
I = range(1, N_TA + 1)   # 弧段下标 1..N_TA
J = range(1, 9)          # 评分区间下标 1..8

# B[j]: 第 j 区间下界；B[j+1]: 下一区间下界（即本区间上界+1）
B = {1: 0, 2: 50, 3: 80, 4: 110, 5: 170,
     6: 230, 7: 250, 8: 300, 9: 86400}
M = {1: 0, 2: 4, 3: 8, 4: 16, 5: 18, 6: 20, 7: 22, 8: 25}

t_S = {i: arcs[i - 1][2] for i in I}
t_E = {i: arcs[i - 1][3] for i in I}
L = 86400       # 大 M，取一天秒数
TOL = 1e-5      # 浮点误差容差


# -------------------- 2. 时间转换与结果校验 --------------------
def sec_to_hms(sec):
    """秒数 -> HH:MM:SS[.sss]"""
    ms = int(round(sec * 1000))
    sec = ms // 1000
    text = f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
    if ms % 1000:
        text += f'.{ms % 1000:03d}'
    return text


def hms(value):
    """'HH:MM:SS' -> 秒数"""
    hh, mm, ss = map(int, value.split(':'))
    return hh * 3600 + mm * 60 + ss


# 论文表 3 的排程（仅用于 --check-paper 验证）
PAPER = {
    1: ('12:47:33', '12:51:23'),
    2: ('12:58:46', '13:02:36'),
    4: ('13:02:36', '13:02:41'),
    5: ('13:02:41', '13:04:31'),
    6: ('13:04:31', '13:06:21'),
    7: ('13:06:21', '13:08:11'),
    8: ('13:08:11', '13:09:01'),
    9: ('13:09:01', '13:11:05'),
    10: ('13:11:05', '13:12:55'),
    12: ('13:12:55', '13:13:18'),
    11: ('13:13:18', '13:15:08'),
}


def score_of(duration):
    """按实际弧长查表得分：B[j] <= duration <= B[j+1]-1 -> M[j]"""
    for j in J:
        if B[j] - TOL <= duration <= B[j + 1] - 1 + TOL:
            return M[j]
    raise ValueError(f'弧长不在合法区间：{duration}')


def validate(result, expected_score=None):
    """从实际时刻独立重算得分，并检查可见性、弧长、非重叠。"""
    if len(result) != N_TA or {r[0] for r in result} != {a[0] for a in arcs}:
        raise ValueError('结果必须恰好包含每个弧段一次（含零长度弧段）')

    windows = {a[0]: a for a in arcs}
    score = 0
    for arc_id, target, start, end, duration in result:
        _, expected_target, lo, hi = windows[arc_id]
        if target != expected_target or not (lo - TOL <= start <= end + TOL and end <= hi + TOL):
            raise ValueError(f'弧段 {arc_id} 超出可见时段或目标编号不符')
        if abs(duration - (end - start)) > TOL:
            raise ValueError(f'弧段 {arc_id} 弧长不符')
        score += score_of(duration)

    # 两两非重叠：对任意两段，至少一个先结束
    for i, left in enumerate(result):
        for right in result[i + 1:]:
            if not (left[3] <= right[2] + TOL or right[3] <= left[2] + TOL):
                raise ValueError(f'弧段 {left[0]} 与 {right[0]} 冲突')

    if expected_score is not None and score != expected_score:
        raise ValueError(f'实际得分 {score} != 期望得分 {expected_score}')
    return score, sum(r[4] for r in result)


# -------------------- 3. 按论文符号建立 PuLP 模型 --------------------
def build_model(T_M=None, fixed_times=None):
    """
    T_M=None : 第一阶段，max Σ_i Σ_j C_ij·M_j
    T_M=数值 : 第二阶段，max Σ_i (Y_E_i - Y_S_i)，s.t. Σ C_ij·M_j = T_M
    fixed_times : {i: (start, end)}，固定时刻用于验证论文表 3
    """
    model = pulp.LpProblem('space_tracking', pulp.LpMaximize)
    make_variable = model.add_variable

    # Y_S_i, Y_E_i ∈ [t_S_i, t_E_i]，连续变量
    Y_S = {i: make_variable(f'Y_S_{i}', lowBound=t_S[i], upBound=t_E[i],
                            cat=pulp.LpContinuous) for i in I}
    Y_E = {i: make_variable(f'Y_E_{i}', lowBound=t_S[i], upBound=t_E[i],
                            cat=pulp.LpContinuous) for i in I}

    # C_ij ∈ {0,1}：第 i 段弧长是否属于第 j 区间
    C = {(i, j): make_variable(f'C_{i}_{j}', cat=pulp.LpBinary)
         for i in I for j in J}

    # S_ij ∈ {0,1} (i≠j)：1 表示 i 在 j 之前，0 表示 i 在 j 之后
    S = {(i, j): make_variable(f'S_{i}_{j}', cat=pulp.LpBinary)
         for i in I for j in I if i != j}

    # 目标表达式
    total_score = pulp.lpSum(C[i, j] * M[j] for i in I for j in J)
    total_length = pulp.lpSum(Y_E[i] - Y_S[i] for i in I)

    if T_M is None:
        model += total_score, 'maximize_total_score'
    else:
        model += total_length, 'maximize_total_length'
        model += total_score == T_M, 'fix_T_M'

    # 每段约束
    for i in I:
        # (1) Σ_j C_ij = 1
        model += pulp.lpSum(C[i, j] for j in J) == 1, f'one_interval_{i}'

        # (2) Y_E_i - Y_S_i >= Σ_j C_ij·B_j
        model += (Y_E[i] - Y_S[i] >=
                  pulp.lpSum(C[i, j] * B[j] for j in J)), f'length_lower_{i}'

        # (3) Y_E_i - Y_S_i <= Σ_j C_ij·B_{j+1} - 1
        model += (Y_E[i] - Y_S[i] <=
                  pulp.lpSum(C[i, j] * B[j + 1] for j in J) - 1), f'length_upper_{i}'

        # (4) 可见窗口：Y_S_i >= t_S_i, Y_E_i <= t_E_i, Y_E_i >= Y_S_i
        #     （前两条与变量上下界重复，保留以对应论文）
        model += Y_S[i] >= t_S[i], f'visible_start_{i}'
        model += Y_E[i] <= t_E[i], f'visible_end_{i}'
        model += Y_E[i] >= Y_S[i], f'nonnegative_length_{i}'

    # 两两非重叠（大 M 放松）：S_ij + S_ji = 1
    #   Y_E_i - Y_S_j <= L·S_ji
    #   Y_E_j - Y_S_i <= L·S_ij
    for i in I:
        for j in I:
            if i < j:
                model += S[i, j] + S[j, i] == 1, f'order_pair_{i}_{j}'
                model += Y_E[i] - Y_S[j] <= L * S[j, i], f'no_overlap_{i}_{j}'
                model += Y_E[j] - Y_S[i] <= L * S[i, j], f'no_overlap_{j}_{i}'

    # 验证模式：固定论文表 3 时刻
    if fixed_times is not None:
        for i in I:
            start, end = fixed_times[i]
            model += Y_S[i] == start, f'paper_start_{i}'
            model += Y_E[i] == end, f'paper_end_{i}'

    return model, Y_S, Y_E, C, S


# -------------------- 4. 调用 CBC 并校验 --------------------
def solve(T_M=None, fixed_times=None, verbose=False):
    """返回 (总得分, 逐弧段排程)"""
    model, Y_S, Y_E, C, S = build_model(T_M, fixed_times)

    # 新版 PuLP：直接用 COIN_CMD 调 CBC
    solver = pulp.COIN_CMD(msg=verbose, gapRel=0, threads=1)
    if not solver.available():
        raise RuntimeError('找不到 CBC，请安装：python -m pip install "pulp[cbc]"')

    stats = model.solve(solver)
    if not (stats.has_solution and stats.status == pulp.LpSolveStatus.Optimal):
        raise RuntimeError(f'CBC 未证明最优：{stats.status}')

    if not model.valid(TOL):
        raise RuntimeError('求解结果未通过模型约束检查')

    # C、S 必须是 0/1
    for group in (C, S):
        for variable in group.values():
            value = variable.value()
            if value is None or min(abs(value), abs(value - 1)) > TOL:
                raise RuntimeError('求解结果含非法二元变量值')

    result = []
    for i in I:
        start = Y_S[i].value()
        end = Y_E[i].value()
        result.append((i, arcs[i - 1][1], start, end, end - start))

    # 独立按实际弧长重算得分，与模型 C_ij 得分对比
    score, _ = validate(result, T_M)
    binary_score = sum(M[j] * round(C[i, j].value()) for i in I for j in J)
    if binary_score != score:
        raise RuntimeError('模型分数与实际弧长评分不一致')
    return score, result


# -------------------- 5. 显示结果及验证论文方案 --------------------
def print_result(title, score, result, only_active=True):
    """打印排程；only_active=True 时隐藏零长度弧段。"""
    _, total = validate(result, score)
    active = sum(r[4] > 0 for r in result)
    print(f'\n{title}：得分 {score}，总弧长 {total} 秒，正长度弧段 {active} 个')
    print('弧段号 目标编号   开始时刻   结束时刻   弧长(s) 得分')
    for r in sorted(result, key=lambda x: (x[2], x[3], x[0])):
        if only_active and r[4] == 0:
            continue
        print(f'{r[0]:>4} {r[1]:>8} {sec_to_hms(r[2]):>10} '
              f'{sec_to_hms(r[3]):>10} {r[4]:8.3f} {score_of(r[4]):4d}')


def check_paper(optimum_score, optimum_total):
    """验证论文表 3 是否达到两阶段最优目标。"""
    fixed = {key: (hms(start), hms(end)) for key, (start, end) in PAPER.items()}
    # 表 3 省略弧段 3，给它一个合法的零长度位置
    fixed[3] = (hms('13:02:36'), hms('13:02:36'))
    score, result = solve(optimum_score, fixed_times=fixed)
    _, total = validate(result)
    if (score, total) != (optimum_score, optimum_total):
        raise RuntimeError('论文方案与两阶段最优目标不一致')
    print_result('论文表3（固定时刻的独立可行性验证）', score, result)
    print('验证通过：论文方案与自由优化方案均达到两阶段最优目标。')


# -------------------- 6. 程序入口 --------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-paper', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # 第一阶段：最大化得分
    score1, result1 = solve(verbose=args.verbose)
    print_result('第一步：最大化得分', score1, result1, only_active=False)

    # 第二阶段：固定最优得分，最大化总弧长
    score2, result2 = solve(score1, verbose=args.verbose)
    print_result('第二步：固定最优得分，最大化总弧长', score2, result2)

    if args.check_paper:
        check_paper(score2, sum(r[4] for r in result2))


if __name__ == '__main__':
    main()