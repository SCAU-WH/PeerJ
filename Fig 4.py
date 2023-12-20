import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr

file_path = 'Fig. 4.xlsx'
sheet_name = 'Sheet1'

data = pd.read_excel(file_path, sheet_name=sheet_name)

values1 = data['area_1']
values2 = data['area_2']

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x_values = np.arange(0, len(values1))

x_new = np.linspace(x_values.min(), x_values.max(), 300)

spline1 = make_interp_spline(x_values, values1, k=3)
spline2 = make_interp_spline(x_values, values2, k=3)

y_smooth1 = spline1(x_new)
y_smooth2 = spline2(x_new)

plt.plot(x_new, y_smooth1, label='Pixel Enumeration Method', linestyle='-', color='blue')
plt.plot(x_new, y_smooth2, label='Region Divide Method ', linestyle='-', color='orange')

plt.xlabel('Image Index')
plt.ylabel('Area Utilization(%)')

plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.legend()
fig.savefig('Fig. 4.tif',
            dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
