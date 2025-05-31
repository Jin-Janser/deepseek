# 데이터 로드
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
df = pd.read_csv('WHO_vaccine.csv')

# 등분산성 검정
from scipy.stats import levene
levene_result = levene(group_a, group_b)

# Welch's t-test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)

# ANOVA 및 Tukey HSD
from statsmodels.formula.api import ols
model = ols('Coverage_Rate ~ C(Vaccine_Type)', data=df).fit()
tukey = pairwise_tukeyhsd(df['Coverage_Rate'], df['Vaccine_Type'])
