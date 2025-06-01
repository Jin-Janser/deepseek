import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('your_path.csv')

# 분포 적합
params = stats.t.fit(df['Unemployment_Rate'])
print(f"t-분포 모수: loc={params[0]:.1f}, scale={params[1]:.1f}, df={params[2]:.1f}")

# 신뢰구간/예측구간
ci = stats.t.interval(0.95, df=params[2], loc=params[0], scale=params[1])
pred_interval = (params[0] - 2.58*params[1], params[0] + 2.58*params[1])

# 시각화
plt.figure(figsize=(12,5))
plt.subplot(121)
sns.histplot(df['Unemployment_Rate'], kde=True)
plt.subplot(122)
stats.probplot(df['Unemployment_Rate'], dist=stats.t, sparams=params, plot=plt)
plt.tight_layout()
