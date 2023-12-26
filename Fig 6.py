import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

file_path = ' '
sheet_name = 'Sheet1'

data = pd.read_excel(file_path, sheet_name=sheet_name)

values1 = data['droplet_area_rate']
values2 = data['PS_area_rate']

plt.figure(1, figsize=(10, 6))
plt.subplot(1, 2, 1)

x_values = np.arange(0, len(values1))
x_new = np.linspace(x_values.min(), x_values.max(), 300)
spline1 = make_interp_spline(x_values, values1, k=3)
spline2 = make_interp_spline(x_values, values2, k=3)
y_smooth1 = spline1(x_new)
y_smooth2 = spline2(x_new)
plt.plot(x_new, y_smooth1, label='Algorithm in this paper', linestyle='-', color='blue')
plt.plot(x_new, y_smooth2, label='Photoshop', linestyle='-', color='orange')
plt.xlabel('Image Index')
plt.ylabel('Droplet deposition coverage rate(%)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

plt.figure(1, figsize=(10, 6))
plt.subplot(1, 2, 2)
coefficients = np.polyfit(values1, values2, 1)
linear_fit = np.polyval(coefficients, values1)
plt.scatter(values1, values2, label='Data points', color='blue')
plt.plot(values1, linear_fit, label='Linear Fit', color='orange')
plt.xlabel('Algorithm in this paper')
plt.ylabel('Photoshop')
plt.legend()
plt.tight_layout()

plt.savefig('Fig. 6.tif',
            dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
