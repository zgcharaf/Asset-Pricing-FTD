library(quantmod)
library(tidyverse)
library(tidyquant)
library(dplyr)
library(readxl)
library(ggplot2)
library(aTSA)
library(forecast)
library(tsDyn)
library(YieldCurve)
library(ycinterextra)
library(xts)
library(stats)
library(graphics)


#Import data
data("ECBYieldCurve")

#TOPIC 1: YIELD CURVE INTERPOLATION

#Plot yield curve data
mat<-c(3/12, 0.5, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)
for( i in c(1,125,250,375,450,655) ){
  plot(mat, ECBYieldCurve[i,], type="o", xlab="Maturities structure in years", ylab="Interest rates values")
  title(main=paste("Spot ECB yield curve observed at",time(ECBYieldCurve[i], sep=" ") ))
  grid()
}
EmpMeanYield <- colMeans(ECBYieldCurve)
#Estimation of the Nelson-Siegel parameters of the yield curve model
NSParameters <- Nelson.Siegel(rate=ECBYieldCurve,maturity=mat)
EmpMNSParameters <- colMeans(NSParameters)

#Plotting the fitted yield curve 

y <- NSrates(NSParameters, mat)
plot(mat,ECBYieldCurve[1,],main="Fitting Nelson-Siegel yield curve",
     xlab=c("Pillars in months"), type="o")
lines(mat,y[1,], col=2)
legend("topleft",legend=c("observed yield curve","fitted yield curve"),
       col=c(1,2),lty=1)
grid()

#Estimation of the Svensson parameters of the yield curve model
SVParameters <- Svensson(rate=ECBYieldCurve, maturity=mat)

#Plotting the fitted yield curve
w <- Srates(SVParameters, mat)
plot(mat,ECBYieldCurve[1,],main="Fitting Svensson yield curve",
     xlab=c("Pillars in months"), type="o", ylim=c(3.4,4.2))
lines(mat,w[1,], col=2)
legend("bottomright",legend=c("observed yield curve","fitted yield curve"),
       col=c(1,2),lty=1)
grid()

#Attempts to forecast the yield curve
forecast <- EmpMNSParameters[1] + EmpMNSParameters[2]*((1-exp(-EmpMNSParameters[4]*mat))/(EmpMNSParameters[4]*mat)) + EmpMNSParameters[3]*((1-exp(-EmpMNSParameters[4]*mat))/(EmpMNSParameters[4]*mat)-exp(-EmpMNSParameters[4]*mat))
plot(mat,EmpMeanYield,main="Empirical Mean versus Estimate M-N Model",
     xlab=c("Pillars in months"), type="o")
lines(mat,forecast, col=2)
legend("topleft",legend=c("empirical yield curve","estimated yield curve"),
       col=c(1,2),lty=1)
grid()


#TOPIC 2 - FACTORIAL ANALYSIS OF THE YIELD CURVE

#Calculating yield curve changes
diff.ECBYieldCurve <- diff(ECBYieldCurve)
diff.ECBYieldCurve2 <- diff.ECBYieldCurve[-1,]

#Principal Component Analysis

ECBYieldCurve.princomp<- princomp(diff.ECBYieldCurve2, cor=TRUE, score=TRUE)
summary(ECBYieldCurve.princomp)
plot(ECBYieldCurve.princomp)
plot(ECBYieldCurve.princomp, type="l")

factor.loadings <- ECBYieldCurve.princomp$loadings[,1:3]
factor.loadings

label.term<- c("3M","6M","1Y","2Y","3Y","4Y","5Y","6Y","7Y","8Y","9Y","10Y","11Y","12Y","13Y","14Y","15Y","16Y","17Y","18Y","19Y","20Y","21Y","22Y","23Y","24Y","25Y","26Y","27Y","28Y","29Y","30Y")
legend.loadings <- c("First principal component","Second principal component","Third principal component")
par(xaxt="n")
matplot(factor.loadings,type="l",
        lwd=3,lty=1,xlab = "Term", ylab = "Factor loadings")
legend(4,max(factor.loadings),legend=legend.loadings,col=1:3,lty=1,lwd=3,cex=0.5)
par(xaxt="s")
axis(1,1:length(label.term),label.term)

summary(ECBYieldCurve.princomp)