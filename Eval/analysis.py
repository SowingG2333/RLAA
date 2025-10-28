import matplotlib.pyplot as plt
import numpy as np

num_iteration = [1, 3, 5]

utility_original = [1.0000, 1.0000, 1.0000]
utility_fgaa_ds = [0.9617, 0.9094, 0.8858]
utility_fgaa_lm = [0.8999, 0.7916, 0.6922] 
utility_fgaa_sft_lm = [0.9649, 0.9502, 0.9391]
utility_rlaa_lm = [0.9346, 0.8695, 0.7952]

privacy_original = [0.4452, 0.4452, 0.4452]
privacy_fgaa_ds = [0.3016, 0.2520, 0.2427]
privacy_fgaa_lm = [0.3041, 0.2384, 0.2150]
privacy_fgaa_sft_lm = [0.3471, 0.3305, 0.3057]
privacy_rlaa_lm = [0.3130, 0.2614, 0.2324]

# 绘制隐私-效用曲线
plt.figure(figsize=(10, 6))
plt.plot(privacy_original, utility_original, marker='o', label='Original Data', color='blue')
plt.plot(privacy_fgaa_ds, utility_fgaa_ds, marker='o', label='FGAA-DeepSeek-V3.2-Exp', color='orange')
plt.plot(privacy_fgaa_lm, utility_fgaa_lm, marker='o', label='FGAA-Llama3-8b', color='green')
plt.plot(privacy_fgaa_sft_lm, utility_fgaa_sft_lm, marker='o', label='FGAA-SFT-Llama3-8b', color='red')
plt.plot(privacy_rlaa_lm, utility_rlaa_lm, marker='o', label='RLAA-Llama3-8b', color='purple')

plt.title('Privacy-Utility Trade-off Curve')
plt.xlabel('Privacy (Lower is Better)')
plt.ylabel('Utility (Higher is Better)')
plt.legend()
plt.grid()
# 保存为pdf
plt.savefig('privacy_utility_curve.pdf')

# ----------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 定义 8 个 PII 属性轴
# attributes = ['LOC', 'INC', 'SEX', 'EDU', 'REL', 'AGE', 'OCC', 'POB']
# N = len(attributes)

# # 2. 从 Table 2 提取的数据
# data = {
#     'Original Text': [0.5206, 0.6198, 0.8016, 0.0743, 0.3223, 0.3057, 0.4214, 0.4958],
#     'FgAA on DeepSeek-V3.2-Exp': [0.0991, 0.6198, 0.6859, 0.0165, 0.1983, 0.2561, 0.0661, 0.0743],
#     'FgAA on Llama3-8b (Naive)': [0.0840, 0.5798, 0.6638, 0.0084, 0.1848, 0.2941, 0.0168, 0.0756],
#     'FgAA on Llama3-8b (SFT)': [0.2231, 0.5950, 0.7520, 0.0413, 0.2727, 0.3388, 0.2066, 0.2148],
#     'RLAA on Llama3-8b': [0.0744, 0.6033, 0.6942, 0.0165, 0.2314, 0.2645, 0.1074, 0.0992]
# }

# # 3. 计算雷达图的角度
# angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# angles += angles[:1] # 闭合多边形

# # 4. 定义颜色和样式
# colors = {
#     'Original Text': 'tab:blue',
#     'FgAA on DeepSeek-V3.2-Exp': 'tab:orange',
#     'FgAA on Llama3-8b (Naive)': 'tab:green',
#     'FgAA on Llama3-8b (SFT)': 'tab:red',
#     'RLAA on Llama3-8b': 'tab:purple' 
# }
# fill_alpha = 0.1
# line_width = 2.0 

# # 5. --- 创建正方形画布 ---
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# # 6. 循环绘制数据
# for method, values in data.items():
#     values_closed = values + values[:1] 
#     ax.plot(angles, values_closed, color=colors[method], 
#             linewidth=line_width, label=method)
#     ax.fill(angles, values_closed, color=colors[method], alpha=fill_alpha)

# # 7. 优化坐标轴
# ax.set_ylim(0, 0.9) 
# ax.set_yticks([0.2, 0.4, 0.6, 0.8])
# ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=10) 
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(attributes, size=12, fontweight='bold')

# # 8. 优化网格线
# ax.grid(axis='y', linestyle='-', alpha=0.4, color='gray') 
# ax.grid(axis='x', linestyle='-', alpha=0.4, color='gray') 

# # 9. --- 关键：将图例改为 3 列，使其更紧凑 ---
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), # 调整 Y 轴位置
#           ncol=3, fontsize=11) # 设为 3 列

# # 10. --- 调整画布边距，使其更紧凑 ---
# plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.95)

# # 11. 保存图像
# plt.savefig('pii_attribute_leakage_risk_comparison.pdf', bbox_inches='tight')

# -----------------------------------------------------------------------------