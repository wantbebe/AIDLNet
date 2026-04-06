import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()

names = ['']


# plt.subplot(1, 1, 1)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data['metrics/mAP50(B)'] = data['metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan)
#     data['metrics/mAP50(B)'] = data['metrics/mAP50(B)'].fillna(data['metrics/mAP50(B)'].interpolate())
#     plt.plot(data['metrics/mAP50(B)'], label=i)
# plt.xlabel('epoch')
# plt.title('mAP_0.5')
# plt.legend()


plt.subplot(1, 1, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['metrics/mAP50-95(B)'] = data['metrics/mAP50-95(B)'].astype(np.float32).replace(np.inf, np.nan)
    data['metrics/mAP50-95(B)'] = data['metrics/mAP50-95(B)'].fillna(data['metrics/mAP50-95(B)'].interpolate())
    plt.plot(data['metrics/mAP50-95(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('mAP50-95.png')
