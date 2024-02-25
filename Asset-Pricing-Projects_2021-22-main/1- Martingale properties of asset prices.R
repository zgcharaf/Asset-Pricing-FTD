# loading packages
library(urca)
library(tseries)
library(xtable)
library(tsDyn)
library(fGarch)
library(forecast)
library(quantmod)
library(vrtest)

# Importing the data
getSymbols("^GSPC",from="2005-01-01", to="2021-09-10", src="yahoo")
getSymbols("^IRX",from="2005-01-01", to="2021-09-10", src="yahoo")


# Data fitting
SPSUB<- GSPC$GSPC.Close[1:1764]
SPCOV<- GSPC$GSPC.Close[3543:4201]

TBSUB <- IRX$IRX.Close[1:1764]
TBCOV <- IRX$IRX.Close[3543:4201]

# Declaring the variables as time series 
SUBTS <- ts(SPSUB$GSPC.Close, frequency=260,start=c(2005,01,03))
plot.ts(SUBTS)
COVIDTS <- ts(SPCOV$GSPC.Close, frequency=260,start=c(2019,01,30))
plot.ts(COVIDTS)

TBSUBTS <- ts(TBSUB,frequency=260,start=c(2005,01,03))
plot.ts(TBSUBTS)
TBCOVTS <- ts(TBCOV,frequency=260,start=c(2019,01,30))
plot.ts(TBCOVTS)

# Log transformation of the time series
LOGSUB <- log(SUBTS)
LOGCOV <- log(COVIDTS)
plot.ts(LOGSUB)
plot.ts(LOGCOV)

# Testing whether series are martingales (by testing if they are pure random walks)
# Step 1: testing whether the process has a unit root
LOGSUB_ADF <- ur.df(LOGSUB, type="none", selectlags="AIC")
summary(LOGSUB_ADF)

LOGCOV_ADF <- ur.df(LOGCOV, type="none", selectlags="AIC")
summary(LOGCOV_ADF)

# Step 2: examining whether the residuals verify the properties of a weak RW
RW1_LOGSUB <- arma(LOGSUB, order=c(1,0), include.intercept=FALSE)
summary(RW1_LOGSUB)
checkresiduals(RW1_LOGSUB)

Residuals_SUB1 <- residuals(RW1_LOGSUB)
Box.test(Residuals_SUB1,lag=20,type="Lj")

RW1_LOGCOV <- arma(LOGCOV, order=c(1,0), include.intercept=FALSE)
summary(RW1_LOGCOV)
checkresiduals(RW1_LOGCOV)

Residuals_COV1 <- residuals(RW1_LOGCOV)
Box.test(Residuals_COV1,lag=20,type="Lj")

# Specifying a strong RW with drift for the subprime crisis period
RW2_SUB <- arma(LOGSUB, order=c(1,0), include.intercept=TRUE)
summary(RW2_SUB)
checkresiduals(RW2_SUB)

Residuals_SUB2 <- residuals(RW2_SUB)
Box.test(Residuals_SUB2,lag=20,type="Lj")

# Specifying a strong RW with drift for the subprime crisis period
RW2_COV <- arma(LOGCOV, order=c(1,0), include.intercept=TRUE)
summary(RW2_COV)
checkresiduals(RW2_COV)

Residuals_COV2 <- residuals(RW2_COV)
Box.test(Residuals_COV2,lag=20,type="Lj")

# Creating the S&P500 excess returns variable
n <- nrow(LOGSUB)
Returns_LOGSUB <- (LOGSUB[2:n, 1] - LOGSUB[1:(n - 1), 1])
plot(Returns_LOGSUB, type="l")

n2 <- nrow(LOGCOV)
Returns_LOGCOV <- (LOGCOV[2:n2, 1] - LOGCOV[1:(n2 - 1), 1])
plot(Returns_LOGCOV, type="l")

n <- nrow(SUBTS)
Returns_SUB <- ((SUBTS[2:n, 1] - SUBTS[1:(n - 1), 1])/SUBTS[1:(n - 1), 1])
plot(Returns_SUB, type="l")

n2 <- nrow(COVIDTS)
Returns_COV <- ((COVIDTS[2:n2, 1] - COVIDTS[1:(n2 - 1), 1])/COVIDTS[1:(n2 - 1), 1])
plot(Returns_COV, type="l")

ExcessReturns_SUB <- Returns_SUB - TBSUB
ExcessReturns_SUB <- na.remove(ExcessReturns_SUB)
ExcessReturns_SUBTS <- ts(ExcessReturns_SUB, frequency=260,start=c(2005,01,03)) 
plot.ts(ExcessReturns_SUBTS)

ExcessReturns_COV <- Returns_COV - TBCOV
ExcessReturns_COV <- na.remove(ExcessReturns_COV)
ExcessReturns_COVTS <- ts(ExcessReturns_COV,frequency=260,start=c(2019,01,30))
plot.ts(ExcessReturns_COVTS)

# Testing whether excess returns series are martingales (by testing if they are pure random walks)
# Step 1: testing whether the process has a unit root
LOGERSUB_ADF <- ur.df(ExcessReturns_SUBTS, type="none", selectlags="AIC")
summary(LOGERSUB_ADF)

LOGERCOV_ADF <- ur.df(ExcessReturns_COVTS, type="none", selectlags="AIC")
summary(LOGERCOV_ADF)

# Step 2: examining whether the residuals verify the properties of a weak RW
RW1_LOGERSUB <- arma(ExcessReturns_SUBTS, order=c(1,0), include.intercept=FALSE)
summary(RW1_LOGERSUB)
checkresiduals(RW1_LOGERSUB)

Residuals_SUBER1 <- residuals(RW1_LOGERSUB)
Box.test(Residuals_SUBER1,lag=20,type="Lj")

RW1_LOGERCOV <- arma(ExcessReturns_COVTS, order=c(1,0), include.intercept=FALSE)
summary(RW1_LOGERCOV)
checkresiduals(RW1_LOGERCOV)

Residuals_COVER1 <- residuals(RW1_LOGERCOV)
Box.test(Residuals_COVER1,lag=20,type="Lj")

# Specifying a strong RW with drift for the subprime crisis period
RW2_SUBER <- arma(ExcessReturns_SUBTS, order=c(1,0), include.intercept=TRUE)
summary(RW2_SUBER)
checkresiduals(RW2_SUBER)

Residuals_SUBER2 <- residuals(RW2_SUBER)
Box.test(Residuals_SUBER2,lag=20,type="Lj")

# Specifying a strong RW with drift for the covid-19 crisis period
RW2_COVER <- arma(ExcessReturns_COVTS, order=c(1,0), include.intercept=TRUE)
summary(RW2_COVER)
checkresiduals(RW2_COVER)

Residuals_COVER2 <- residuals(RW2_COVER)
Box.test(Residuals_COVER2,lag=20,type="Lj")

 