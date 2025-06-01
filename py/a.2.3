# Python 데이터 전처리
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 데이터 로드
red = pd.read_csv('winequality-red.csv', sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')
red['type'] = 0; white['type'] = 1
df = pd.concat([red, white])

# 이상치 제거
Q1 = df.quantile(0.25); Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]

# 표준화
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('quality', axis=1))
y = df['quality'].values

import pymc3 as pm

with pm.Model() as model:
    # 사전분포: N(0,1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X.shape[1])
    alpha = pm.Normal('alpha', mu=y.mean(), sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    mu = alpha + pm.math.dot(X, beta)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    
    trace = pm.sample(2000, tune=2000, cores=4)
