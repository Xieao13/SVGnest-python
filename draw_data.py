import re
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import platform


# 读取日志文件
log_file_path = 'logs/ppo_20250928_013421/train_ppo.log'  # 修改为你的日志文件路径

system = platform.system()
if system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
elif system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 存储提取的数据
timestamps = []
utilizations = []
steps = []

# 读取并解析日志文件
with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 匹配包含 "Evaluation finished" 的行
        if 'Evaluation finished' in line:
            # 提取时间戳
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)

            # 提取 Average Utilization
            util_match = re.search(r'Average Utilization: ([\d.]+)', line)

            if timestamp_match and util_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                utilization = float(util_match.group(1))

                timestamps.append(timestamp)
                utilizations.append(utilization)

print(f"共提取到 {len(utilizations)} 个evaluation结果")
print("\n提取的数据：")
for ts, util in zip(timestamps, utilizations):
    print(f"{ts}: {util:.4f}")

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(utilizations) + 1), utilizations, marker='o',
         linewidth=2, markersize=8, color='#2E86AB')

plt.xlabel('Evaluation次数', fontsize=12)
plt.ylabel('Average Utilization', fontsize=12)
plt.title('模型训练过程中的Utilization变化', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# 添加数值标签
for i, (x, y) in enumerate(zip(range(1, len(utilizations) + 1), utilizations)):
    plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('utilization_plot.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存为: utilization_plot.png")
plt.show()

# 如果需要，也可以按时间轴绘制