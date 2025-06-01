library(tidyverse)
# 데이터 처리
df <- read.csv("WHO_vaccine.csv") %>%
  filter(Region %in% c("A","B"))

# 선형회귀 추세 분석
lm_model <- lm(Coverage_Rate ~ Year, data=mcv_data)
summary(lm_model)

# ANOVA
aov_result <- aov(Coverage_Rate ~ Vaccine_Type, data=df)
TukeyHSD(aov_result)
