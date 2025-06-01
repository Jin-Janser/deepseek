import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 데이터 로드
df = pd.read_csv('your_path.csv')

# 전처리
df['gender'] = df['gender'].map({'male':0, 'female':1})

# 기술통계
desc_stats = df[['math score','reading score','writing score']].describe()

# t-검정
male = df[df['gender']==0]
female = df[df['gender']==1]

t_results = {}
for subject in ['math score','reading score','writing score']:
    t_stat, p_val = stats.ttest_ind(male[subject], female[subject], equal_var=False)
    t_results[subject] = [t_stat, p_val]

# 시각화
plt.figure(figsize=(15,5))
for i, subject in enumerate(['math score','reading score','writing score'], 1):
    plt.subplot(1,3,i)
    sns.boxplot(x='gender', y=subject, data=df)
    plt.title(f'{subject} Distribution')
plt.show()
