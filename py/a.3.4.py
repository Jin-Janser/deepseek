import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import gmean
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

# 데이터 생성
np.random.seed(42)
n_samples = 5000

data = {
    'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                 n_samples, p=[0.4, 0.35, 0.2, 0.05]),
    'experience': np.clip(np.random.exponential(10, n_samples), 0, 40),
    'income': np.zeros(n_samples)
}

# 교육 수준별 소득 계수
edu_coeff = {'High School': 1, 'Bachelor': 1.5, 'Master': 2, 'PhD': 3}
data['income'] = 30000 * data['education'].map(edu_coeff) * \
                (1 + 0.02 * data['experience']) * \
                np.random.lognormal(0, 0.35, n_samples)

df = pd.DataFrame(data)
df['log_income'] = np.log(df['income'])

def optimize_bandwidth(data):
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(0.1, 1, 20)},
                        cv=5)
    grid.fit(data.values.reshape(-1, 1))
    return grid.best_params_['bandwidth']

# 그룹별 밀도 추정
groups = {'Gender': 'gender', 'Education': 'education'}
kde_results = {}

for group_type, col in groups.items():
    bandwidths = {}
    densities = {}
    
    for group in df[col].unique():
        subset = df[df[col] == group]['log_income']
        bw = optimize_bandwidth(subset)
        kde = KernelDensity(bandwidth=bw).fit(subset.values.reshape(-1, 1))
        
        x = np.linspace(subset.min(), subset.max(), 1000)
        log_dens = kde.score_samples(x.reshape(-1, 1))
        
        bandwidths[group] = bw
        densities[group] = np.exp(log_dens)
    
    kde_results[group_type] = {'bandwidths': bandwidths, 'densities': densities}

plt.figure(figsize=(12, 6))
for i, (group, values) in enumerate(kde_results['Gender']['densities'].items()):
    x = np.linspace(df['log_income'].min(), df['log_income'].max(), 1000)
    plt.plot(x, values, label=group, lw=2)
plt.title('성별 로그 소득 분포')
plt.xlabel('로그 소득')
plt.ylabel('밀도')
plt.legend()
plt.show()

def gini_coefficient(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) / (n * np.sum(x))) - (n + 1)/n

def atkinson_index(x, epsilon=0.5):
    if epsilon == 1:
        return 1 - gmean(x)/np.mean(x)
    else:
        return 1 - (np.mean(x**(1-epsilon))**(1/(1-epsilon)))/np.mean(x)

# 부트스트랩 신뢰구간
def bootstrap_ci(data, func, n_boot=1000):
    stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats.append(func(sample))
    return np.percentile(stats, [2.5, 97.5])

# 그룹별 분석
inequality = {}
for group in groups:
    results = {}
    for subset in df[group].unique():
        income_data = df[df[group] == subset]['income']
        results[subset] = {
            'Gini': (gini_coefficient(income_data), 
                    bootstrap_ci(income_data, gini_coefficient)),
            'Atkinson': (atkinson_index(income_data), 
                        bootstrap_ci(income_data, atkinson_index))
        }
    inequality[group] = results

def gini_coefficient(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) / (n * np.sum(x))) - (n + 1)/n

def atkinson_index(x, epsilon=0.5):
    if epsilon == 1:
        return 1 - gmean(x)/np.mean(x)
    else:
        return 1 - (np.mean(x**(1-epsilon))**(1/(1-epsilon)))/np.mean(x)

# 부트스트랩 신뢰구간
def bootstrap_ci(data, func, n_boot=1000):
    stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats.append(func(sample))
    return np.percentile(stats, [2.5, 97.5])

# 그룹별 분석
inequality = {}
for group in groups:
    results = {}
    for subset in df[group].unique():
        income_data = df[df[group] == subset]['income']
        results[subset] = {
            'Gini': (gini_coefficient(income_data), 
                    bootstrap_ci(income_data, gini_coefficient)),
            'Atkinson': (atkinson_index(income_data), 
                        bootstrap_ci(income_data, atkinson_index))
        }
    inequality[group] = results

# 회귀 계수 비교
coefs = pd.DataFrame({q: results[q]['params'] for q in results.keys()})
coefs.plot(kind='bar', figsize=(12,6))
plt.title('분위별 회귀 계수 비교')
plt.xticks(rotation=0)
plt.show()
