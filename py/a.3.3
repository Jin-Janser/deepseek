import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# 데이터 로딩 (파일 경로 수정 필요)
df = pd.read_csv('train.csv')

# 결측치 처리
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 범주형 변수 인코딩
df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype('category')
df['Pclass'] = df['Pclass'].astype('category')

# 로그 변환
df['log_Fare'] = np.log1p(df['Fare'])

plt.figure(figsize=(10,6))
ax = plt.subplot(111)

for sex_val, label in [(0, 'Male'), (1, 'Female')]:
    kmf = KaplanMeierFitter()
    mask = df['Sex'] == sex_val
    kmf.fit(df['Age'][mask], df['Survived'][mask], label=label)
    kmf.plot_survival_function(ax=ax, ci_show=False)

plt.title('Kaplan–Meier Survival by Sex')
plt.xlabel('Age')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# 로그-순위 검정
results = logrank_test(
    df['Age'][df.Sex==0], df['Age'][df.Sex==1],
    event_observed_A=df['Survived'][df.Sex==0],
    event_observed_B=df['Survived'][df.Sex==1]
)
print(f"Log-rank test p-value: {results.p_value:.4f}")

plt.figure(figsize=(10,6))
ax = plt.subplot(111)

for pclass in [1, 2, 3]:
    kmf = KaplanMeierFitter()
    mask = df['Pclass'] == pclass
    kmf.fit(df['Age'][mask], df['Survived'][mask], label=f'Class {pclass}')
    kmf.plot_survival_function(ax=ax, ci_show=False)

plt.title('Kaplan–Meier Survival by Pclass')
plt.xlabel('Age')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# 데이터 준비
df_cox = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'log_Fare']].dropna()

cph = CoxPHFitter()
cph.fit(df_cox, duration_col='Age', event_col='Survived', show_progress=True)

# 결과표
cox_summary = cph.print_summary()

cph.check_assumptions(df_cox, p_value_threshold=0.05)

# 프로필 생성
new_passenger = pd.DataFrame({
    'Pclass': [1],
    'Sex': [1],
    'Age': [30],
    'SibSp': [0],
    'Parch': [0],
    'log_Fare': [np.log1p(30)]  # 1등석 평균 운임
})

# 생존 확률 예측
survival_prob = cph.predict_survival_function(new_passenger)

# 시각화
plt.figure(figsize=(10,6))
plt.plot(survival_prob.index, survival_prob.values.T)
plt.title('Survival Probability: 30yo Female, 1st Class')
plt.xlabel('Age')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()
