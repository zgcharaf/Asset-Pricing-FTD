library(quantmod)
library(tidyverse)
library(tidyquant)
library(dplyr)
library(readxl)
library(ggplot2)
library(aTSA)
library(urca)
library(strucchange)
library(vrtest)
library(vars)
library(forecast)
library(tsDyn)

#Data importation
USA_Data <- read.table("C:/Users/khali/Desktop/AP - HW3/AP3_USA_Import.txt")

#Declaring variables as time series 
SP500 <- ts(USA_Data$SP500, frequency=4,start=c(1985,1))
RGDP <- ts(USA_Data$RGDP, frequency=4,start=c(1985,1))
Yield10Y <- ts(USA_Data$Yield10Y, frequency=4,start=c(1985,1))

#Taking the log of our variables
LogSP500 <- log(SP500)
LogRGDP <- log(RGDP)

#Data overview
plot.ts(LogSP500)
plot.ts(LogRGDP)
plot.ts(Yield10Y)

#We prove the series are non-stationary
adf.test(LogSP500)
adf.test(LogRGDP)
adf.test(Yield10Y)

ur_LogSP500_trend <- ur.df(LogSP500, type =  "trend", selectlags =  "AIC")
summary(ur_LogSP500_trend)

ur_LogRGDP_trend <- ur.df(LogRGDP, type =  "trend", selectlags =  "AIC")
summary(ur_LogRGDP_trend)

ur_Yield_trend <- ur.df(Yield10Y, type =  "drift", selectlags =  "AIC")
summary(ur_Yield_trend)

#We prove the series are I(1)
Diff_SP500 <- diff(log(LogSP500))
Diff_SP500_TS <- as.ts(Diff_SP500)
plot.ts(Diff_SP500_TS)

Diff_RF <- diff(Yield10Y)
Diff_RF_TS <- as.ts(Diff_RF)
plot.ts(Diff_RF_TS)

Diff_RGDP <- diff(LogRGDP)
Diff_RGDP_TS <- as.ts(Diff_RGDP)
plot.ts(Diff_RGDP_TS)

adf.test(Diff_SP500_TS)
adf.test(Diff_RGDP_TS)
adf.test(Diff_RF_TS)

#We regress the log(S&P500) on log(RGDP) and the risk-free rate 
lm1 <- lm(LogSP500 ~ LogRGDP + Yield10Y)
summary(lm1)

#We extract the regression's residuals
resid <- resid(lm1)

#Unit root test 
ur_resid_none <- ur.df(resid, type =  "none", selectlags =  "AIC")
summary(ur_resid_none)
ndiffs(resid)

#Test for structural breaks 
Breaks <- breakpoints(LogSP500 ~ LogRGDP + Yield10Y)
summary(Breaks)
confint(Breaks)
breakpoints(Breaks)
coef(Breaks, breaks = 4)
plot(Breaks)

#We then run the second regression
lm2 <- lm(LogSP500 ~ LogRGDP + Yield10Y + breakfactor(Breaks, breaks = 4))
summary(lm2)
resid2 <- resid(lm2)

#Unit root test
ur_resid2_none <- ur.df(resid2, type =  "none", selectlags =  "AIC")
summary(ur_resid2_none)
ndiffs(resid2)

#Constitution of a single dataset for optimal lag selection
LogData <- cbind(LogSP500, LogRGDP, Yield10Y)
LogData_ts <- ts(data=LogData, start = c(1985,1), end = c(2021, 4), frequency = 4)
LogData2 <- cbind(LogSP500, LogRGDP, Yield10Y, breakfactor(Breaks, breaks = 4))

#Test for the cointegration rank 
#Lag Selection Criteria
lagselect <- VARselect(LogData_ts, lag.max = 10, type = "none")
lagselect$selection

#Trace test
Trace <- ca.jo(LogData2, type = "trace", ecdet = "const", K = 2)
summary(Trace)
#Two cointegrating relationship, as we reject critical value for <=1
# Max-eigenvalue test
Eigen <- ca.jo(LogData2, type = "eigen", ecdet = "const", K = 2)
summary(Eigen)
#Two cointegrating relationship, as we reject critical value for <=1

#Estimation of the corresponding error-correcting equation
VECM <- VECM(LogData, lag=8, r=2, include = "both", beta= NULL, estim="ML", LRinclude="const", exogen = breakfactor(Breaks, breaks = 4))
summary(VECM)

#Converting the VECM model to a VAR model
VAR <- vec2var(Trace, r=2)

#Forecasting the S&P500 Long-term equilibrium value for two periods ahead
forecastVAR <- predict(VAR, n.ahead = 12, ci = 0.95) 
fanchart(forecastVAR, names = 'logS&P500', main = 'Forecast of the logarithm of the S&P500 stock index 12 quarters ahead')
