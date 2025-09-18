import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Read CSV file
df = pd.read_csv('experiment.csv')
print(df.head())

# Calculate accuracy improvement and data reduction
df['accuracy_improvement'] = df['ShapeVVE'] - df['LSP']
df['data_reduction'] = 1 - df['Chosen/All Variables']

# Create a wider and flatter figure
fig, ax = plt.subplots(1, 1, figsize=(11,7))  # 增加高度以容纳标签

# Accuracy Improvement vs Data Reduction scatter plot
scatter = ax.scatter(df['data_reduction'], df['accuracy_improvement'],
                     color='black', s=120, alpha=0.8,
                     edgecolors='black', linewidth=1.5)

# 存储所有文本对象以便后续调整
text_objects = []

# Add data point labels above the points
for i, row in df.iterrows():
    x, y = row['data_reduction'], row['accuracy_improvement']

    # 基础偏移量
    text_y_offset = 0.03

    text = ax.annotate(row['Dataset'],
                       (x, y),
                       xytext=(0, text_y_offset),
                       textcoords='offset fontsize',
                       fontsize=11, fontweight='bold',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.9, edgecolor='gray', linewidth=0.5))
    text_objects.append(text)

ax.set_xlabel('Data Reduction Ratio', fontsize=20, fontweight='bold')
ax.set_ylabel('Accuracy Improvement', fontsize=20, fontweight='bold')
ax.set_title('Accuracy Improvement (ShapeVVE - LSP) vs Data Reduction Ratio', fontsize=20, fontweight='bold', pad=20)

# Customize grid and ticks
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=12)

# Set appropriate limits with extra space for labels
x_padding = 0.05
ax.set_xlim(df['data_reduction'].min() - x_padding, df['data_reduction'].max() + x_padding)
ax.set_ylim(df['accuracy_improvement'].min() , df['accuracy_improvement'].max())

# Add only the horizontal reference line
ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Baseline (No Improvement)')
ax.legend(loc='upper left', fontsize=20, framealpha=0.9)


# 手动调整重叠标签的方法
# 运行第一次后会看到哪些标签重叠，然后在这里添加调整
def adjust_overlapping_labels():
    """手动调整重叠的标签"""
    # 创建一个字典来映射数据集名称到调整偏移量 (x_offset, y_offset)
    manual_adjustments = {
        # 'Iris': (0.01, 0.04),  # 向右移动0.01，向上移动0.04
        # 'Wine': (-0.01, 0.05),  # 向左移动0.01，向上移动0.05
        # 'BreastCancer': (0.02, 0),  # 向右移动0.02，垂直位置不变
        # 'Diabetes': (0, 0.06),  # 只向上移动0.06
        'SelfRegulationSCP1':(0,-1.5),
        'Epilepsy':(0.2,-1.5),
        'BasicMotions':(0,1.6),
        'FingerMovements':(0,2),
        'FaceDetection':(0,-2),
        'PenDigits': (0, -1),
        'JapaneseVowels':(0,1),
        'DuckDuckGeeze':(0,-0.5),
        'Handwriting':(0,2),
        'ERing':(0,2),
        'UWaveGestureLibrary':(8,0),
        'MotorImagery':(0,1),
        'LSST':(0,1)
    }

    for text in text_objects:
        dataset_name = text.get_text()
        if dataset_name in manual_adjustments:
            x_offset, y_offset = manual_adjustments[dataset_name]
            # 获取当前文本位置并应用偏移
            current_pos = text.get_position()
            text.set_position((current_pos[0] + x_offset, current_pos[1] + y_offset))


# 应用手动调整
adjust_overlapping_labels()

# 添加统计信息
mean_improvement = df['accuracy_improvement'].mean()
mean_reduction = df['data_reduction'].mean()
plt.tight_layout(pad=3.0)

# 显示图形
# plt.show()

# 统计信息
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(f"Number of datasets: {len(df)}")
print(f"LSP average accuracy: {df['LSP'].mean():.4f}")
print(f"ShapeVVE average accuracy: {df['ShapeVVE'].mean():.4f}")
print(f"Average accuracy improvement: {df['accuracy_improvement'].mean():.4f}")
print(f"Average data reduction: {df['data_reduction'].mean():.4f}")
print(f"Datasets with improvement: {len(df[df['accuracy_improvement'] > 0])}")
print(f"Datasets with degradation: {len(df[df['accuracy_improvement'] < 0])}")

# 保存图形
plt.savefig('accuracy_improvement_flat.png', dpi=300, bbox_inches='tight', facecolor='white')