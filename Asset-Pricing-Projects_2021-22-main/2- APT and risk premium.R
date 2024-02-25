library(tseries)
library(quantmod)
library(tidyquant)
library(xts)
library(urca)
library(xtable)
library(tsDyn)
library(fGarch)
library(forecast)
library(vrtest)
library(tidyverse)
library(dplyr)

#Importing data (source: FRED)
setwd("C:/Users/khali/Desktop/AP 2")
data <-read.table("Data_AP2_ToImport.txt", fill=TRUE, header=TRUE, dec=".")

#Declaring variables as time series (all variables are in index, apart from yields)
SP500 <- ts(data$SP500, frequency=12,start=c(2011,10))
NYield10Y <- ts(data$NYield10Y, frequency=12,start=c(2011,10))
IP <- ts(data$Industrial_Production, frequency=12,start=c(2011,10))
RYield10Y <- ts(data$RYield10Y, frequency=12,start=c(2011,10))
Volatility <- ts(data$VIX, frequency=12,start=c(2011,10))
BBBYield <- ts(data$BBB_Yield, frequency=12,start=c(2011,10))
FRED_data <- ts(data,start=c(2011,10),frequency=12)

#Plotting the time series
plot.ts(SP500,plot.type="single")
plot.ts(NYield10Y,plot.type="single")
plot.ts(IP,plot.type="single")
plot.ts(RYield10Y,plot.type="single")
plot.ts(Volatility,plot.type="single")
plot.ts(BBBYield,plot.type="single")
plot.ts(FRED_data,plot.type="multiple")

#Estimating the S&P500 risk premium
# Step 1: S&P500 returns (monthly)
SP500_num <- as.numeric(SP500)
Diff_SP500 <- SP500_num - lag(SP500_num)
Returns_SP500 <- (Diff_SP500 / SP500)*100

plot(Returns_SP500, type="l")
head(Returns_SP500)
tail(Returns_SP500)

# Step 2: subtracting 10Y treasury yield (we assume a long term investment horizon)
ER_SP500 <- Returns_SP500 - NYield10Y
plot(ER_SP500, type="l")
head(ER_SP500)
tail(ER_SP500)

# Estimating the multi-beta relationship underpinning S&P500 risk premium
#Step 1: constructing the inflation risk factor
Inflation_Risk <- NYield10Y-RYield10Y
plot(Inflation_Risk, type="l")

#Step 2: constructing the credit spread factor
Credit_Spread <- BBBYield-NYield10Y
plot(Credit_Spread, type="l")

#Step 3: converting index variables in m/m growth rates
IP_num <- as.numeric(IP)
Diff_IP <- IP_num - lag(IP_num)
IP_mm <- (Diff_IP / IP_num)*100
plot(IP_mm, type="l")

Volatility_num <- as.numeric(Volatility)
Diff_Volatility <- Volatility_num - lag(Volatility_num)
Volatility_mm <- (Diff_Volatility / Volatility_num)*100
plot(Volatility_mm, type="l")

#Step 4: Running a 5-factor model regression to explain S&P500 returns
model1 <- lm(Returns_SP500~IP_mm+Inflation_Risk+RYield10Y+Volatility_mm+Credit_Spread)
summary(model1)
