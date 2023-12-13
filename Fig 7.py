import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

# 读取 Excel 表格数据
file_path = 'E:\\论文\\论文\\新数据\\Fig. 7.xlsx'  # 替换为你的 Excel 文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称

data = pd.read_excel(file_path, sheet_name=sheet_name)

values1 = data['droplet_area_rate']
values2 = data['leaf_area_rate']

# 创建一个折线图
fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # 设置图表尺寸

# 生成递增的横坐标值从1开始
x_values = np.arange(0, len(values1))

# 生成更密集的横坐标值，用于平滑曲线绘制
x_new = np.linspace(x_values.min(), x_values.max(), 300)

# 使用 make_interp_spline 进行平滑曲线拟合
spline1 = make_interp_spline(x_values, values1, k=3)
spline2 = make_interp_spline(x_values, values2, k=3)

# 得到平滑后的曲线数据
y_smooth1 = spline1(x_new)
y_smooth2 = spline2(x_new)


plt.plot(x_new, y_smooth1, label='Randomly intercept leaf', linestyle='-', color='blue')
plt.plot(x_new, y_smooth2, label='Whole leaf', linestyle='-', color='orange')


# 添加图例和标签
plt.xlabel('Image Index')
plt.ylabel('Droplet deposition coverage rate(%)')


plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.legend()

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

fig.savefig('Fig. 7.tif',
            dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
