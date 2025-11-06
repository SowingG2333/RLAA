import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 加载数据 ---
csv_file_path = '/root/autodl-tmp/RLAA/Base/FgAA/sft/data/attacker/attacker_eval_loss.csv' 
df = pd.read_csv(csv_file_path)

# --- 2. 定义 X 轴和 Y 轴 ---
x_column = 'Step'
y_column = 'Value'

# --- 3. 开始绘图 ---
print("开始绘图...")
plt.figure(figsize=(6, 6)) # 设置图表大小

# 你可以在这里自定义所有样式：
plt.plot(
    df[x_column], 
    df[y_column],
    label='eval_loss',   # 图例标签
    color='#D95F02',        # 曲线颜色 (使用 'red', '#FF0000' 等)
    linestyle='-',       # 线条样式 ('-', '--', ':', '-.')
    linewidth=3          # 线条宽度
)

# --- 4. 美化图表 (添加标签、标题等) ---
plt.title('Eval Loss', fontsize=16)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()  # 显示图例
plt.grid(True, linestyle=':', alpha=0.7) # 添加网格线
plt.tight_layout() # 自动调整布局

# --- 5. 保存和显示图表 ---
plt.savefig("plot.svg")

print("图表已保存为 plot.svg")

# 在窗口中显示图表
plt.show()