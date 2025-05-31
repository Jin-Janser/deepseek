import numpy as np
import pandas as pd
import pymc3 as pm
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from pymc3.math import log, exp

# 가상 데이터 생성 (실제 SEER 데이터 구조 모방)
np.random.seed(42)
n_regions = 50
years = np.arange(2010, 2020)
age_groups = ['20-39', '40-59', '60-79', '80+']
races = ['White', 'Black', 'Hispanic', 'Asian']

data = []
for region in range(n_regions):
    lat = np.random.uniform(32, 48)
    lon = np.random.uniform(-125, -70)
    for year in years:
        for age in age_groups:
            for race in races:
                pop = np.random.poisson(5000) + 1000
                base_rate = 50 + 20*np.sin(lat*np.pi/180)
                age_effect = 1.2 if age == '80+' else 1.0
                race_effect = 1.3 if race == 'Black' else 1.0
                cases = np.random.poisson(base_rate * age_effect * race_effect * pop / 1e5)
                data.append([region, lat, lon, year, age, race, cases, pop])

df = pd.DataFrame(data, columns=['region','lat','lon','year','age','race','cases','pop'])

# 결측치 처리
df = df[(df['cases'] > 0) & (df['pop'] > 100)]

# 로그 변환
df['log_rate'] = np.log((df['cases']/df['pop'])*1e5 + 0.1)
df['region'] = df['region'].astype('category')

# 공간 가중치 행렬 생성
coords = df[['lon','lat']].drop_duplicates().values
W = DistanceBand(coords, threshold=100, binary=True).full()[0]

with pm.Model() as hierarchical_model:
    # 고정 효과
    age = pm.Normal('age', mu=0, sigma=1, shape=len(age_groups))
    race = pm.Normal('race', mu=0, sigma=1, shape=len(races))
    
    # 지역 랜덤 효과
    region_sd = pm.HalfNormal('region_sd', sigma=1)
    region_effect = pm.Normal('region_effect', mu=0, sigma=region_sd, 
                             shape=n_regions)
    
    # 모델 구조
    mu = (age[df.age.cat.codes] 
          + race[df.race.cat.codes] 
          + region_effect[df.region.cat.codes])
    rate = pm.Deterministic('rate', exp(mu))
    cases = pm.Poisson('cases', mu=rate * df['pop']/1e5, observed=df['cases'])
    
    # MCMC 샘플링
    trace = pm.sample(2000, tune=1000, target_accept=0.95)

with pm.Model() as spatial_model:
    # 공간 효과
    phi = pm.CAR('phi', mu=0, W=W, alpha=1)
    spatial_sd = pm.HalfNormal('spatial_sd', sigma=1)
    
    # 모델 구조
    mu = (age[df.age.cat.codes] 
          + race[df.race.cat.codes] 
          + phi[df.region.cat.codes]*spatial_sd)
    rate = pm.Deterministic('rate', exp(mu))
    cases = pm.Poisson('cases', mu=rate * df['pop']/1e5, observed=df['cases'])
    
    trace_spatial = pm.sample(2000, tune=1000)
model_compare = pm.compare({
    hierarchical_model: trace,
    spatial_model: trace_spatial
}, ic='WAIC')

print(model_compare)
