import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. 데이터 로드 및 전처리
def load_and_preprocess_data(filepath):
    # 데이터 로드
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    
    # 결측치 처리 (선형 보간)
    df['Close'] = df['Close'].interpolate(method='linear')
    
    # 로그 수익률 계산
    df['Log_Return'] = np.log(df['Close']).diff().dropna()
    
    # 이상치 제거 (IQR 1.5 기준)
    Q1 = df['Log_Return'].quantile(0.25)
    Q3 = df['Log_Return'].quantile(0.75)
    IQR = Q3 - Q1
    filtered = df[(df['Log_Return'] >= Q1 - 1.5*IQR) & 
                (df['Log_Return'] <= Q3 + 1.5*IQR)]
    
    return filtered

# 2. STL 분해 및 시각화
def stl_decomposition(data, period=52):
    stl = STL(data, period=period, robust=True)
    result = stl.fit()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(4,1,1)
    plt.plot(result.trend)
    plt.title('Trend Component')
    
    plt.subplot(4,1,2)
    plt.plot(result.seasonal)
    plt.title('Seasonal Component')
    
    plt.subplot(4,1,3)
    plt.plot(result.resid)
    plt.title('Residual Component')
    
    plt.subplot(4,1,4)
    plt.plot(data, label='Original')
    plt.plot(result.trend + result.seasonal, label='Reconstructed')
    plt.legend()
    plt.title('Original vs Reconstructed')
    
    plt.tight_layout()
    plt.show()
    
    return result

# 3. ARIMA 모델링 및 예측
def arima_forecasting(data, steps=8):
    # ACF/PACF 분석
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    plot_acf(data, ax=ax1, lags=40)
    plot_pacf(data, ax=ax2, lags=40)
    plt.tight_layout()
    plt.show()
    
    # 최적 ARIMA 모델 선택
    model = auto_arima(data, seasonal=False, 
                      stepwise=True, information_criterion='bic',
                      trace=True, error_action='ignore',
                      suppress_warnings=True)
    
    print(f"Best ARIMA{model.order} - AIC:{model.aic()} BIC:{model.bic()}")
    
    # 8주 예측
    forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    
    # 시각화
    plt.figure(figsize=(12,6))
    plt.plot(data, label='Historical')
    plt.plot(forecast, label='Forecast', color='red')
    plt.fill_between(forecast.index, 
                    conf_int[:,0], conf_int[:,1],
                    color='pink', alpha=0.3)
    plt.title(f'ARIMA{model.order} Forecast')
    plt.legend()
    plt.show()
    
    return model, forecast

# 4. GARCH 분석
def garch_analysis(data, p=1, q=1):
    # ARCH-LM 검정
    model = arch_model(data, vol='GARCH', p=p, q=q, dist='StudentsT')
    result = model.fit(update_freq=5, disp='off')
    
    print(result.summary())
    print("\nARCH-LM Test p-value:", result.arch_lm_test().pvalue)
    
    # 조건부 변동성 시각화
    plt.figure(figsize=(12,6))
    plt.plot(data, label='Log Returns')
    plt.plot(result.conditional_volatility, 
            label='Conditional Volatility', color='red')
    plt.legend()
    plt.title('GARCH(1,1) Conditional Volatility')
    plt.show()
    
    return result

# 메인 실행
if __name__ == "__main__":
    # 데이터 준비
    dji = load_and_preprocess_data('DJI_weekly.csv')
    
    # STL 분해
    stl_result = stl_decomposition(dji['Log_Return'], period=52)
    
    # ARIMA 모델링 및 예측
    arima_model, forecast = arima_forecasting(dji['Log_Return'])
    
    # GARCH 분석
    garch_result = garch_analysis(dji['Log_Return'])
    
    # 최종 결과 출력
    print("\n최종 분석 결과 요약")
    print(f"1. 추세 강도: {stl_result.trend.std():.4f}")
    print(f"2. ARIMA 예측 범위 (95%): {forecast.min():.4f} ~ {forecast.max():.4f}")
    print(f"3. GARCH 지속성(α+β): {garch_result.params['alpha[1]'] + garch_result.params['beta[1]']:.4f}")
