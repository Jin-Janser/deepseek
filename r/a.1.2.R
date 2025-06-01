library(fitdistrplus)
library(ggplot2)

df <- read.csv("your_path.csv")
fit <- fitdist(df$Unemployment_Rate, "t", method="mle")

# 신뢰구간 계산
ci <- qt(c(0.025,0.975), df=fit$estimate[3])*fit$estimate[2] + fit$estimate[1]

# 시각화
ggplot(df, aes(sample=Unemployment_Rate)) + 
  geom_qq(distribution=qt, dparams=list(df=fit$estimate[3])) +
  geom_abline(slope=1, intercept=0)
