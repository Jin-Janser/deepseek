import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import shap

# 데이터 로드
df = pd.read_csv('train.csv').drop('Id', axis=1)

# 결측치 처리
num_vars = df.select_dtypes(include='number').columns.tolist()
cat_vars = df.select_dtypes(exclude='number').columns.tolist()

imputer = IterativeImputer(max_iter=10)
df[num_vars] = imputer.fit_transform(df[num_vars])
df[cat_vars] = df[cat_vars].fillna(df[cat_vars].mode().iloc[0])

# 파생변수 생성
df['RemodelAge'] = df['YearRemodAdd'] - df['YearBuilt']

# Target Encoding
te = TargetEncoder()
df[cat_vars] = te.fit_transform(df[cat_vars], df['SalePrice'])

# VIF 기반 변수 선택
X = sm.add_constant(df.drop('SalePrice', axis=1))
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
selected_vars = vif[vif < 5].index.tolist()

# 최종 모델
final_model = sm.OLS(df['SalePrice'], X[selected_vars]).fit()
print(final_model.summary())

# SHAP 값 계산
explainer = shap.LinearExplainer(final_model, X[selected_vars])
shap_values = explainer.shap_values(X[selected_vars])
shap.summary_plot(shap_values, X[selected_vars])
