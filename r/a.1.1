library(tidyverse)
library(ggplot2)

# 데이터 로드
df <- read.csv('your_path.csv')

# 전처리
df$gender <- ifelse(df$gender == "male", 0, 1)

# 기술통계
desc_stats <- df %>%
  select(math.score, reading.score, writing.score) %>%
  summary()

# t-검정
t_test_results <- list(
  math = t.test(math.score ~ gender, data = df),
  reading = t.test(reading.score ~ gender, data = df),
  writing = t.test(writing.score ~ gender, data = df)
)

# 시각화
ggplot(df, aes(x=factor(gender), y=math.score)) + 
  geom_boxplot() +
  ggtitle("Math Score Distribution by Gender")
