## Name:
## Group Number 7


## Loading in libraries
library(fpp3)
library(tsibble)
library(dplyr)
library(forecast)
library(fpp2)
library(tidyverse)
library(ggplot2)
library(fabletools)
# install.packages("imputeTS")
library(imputeTS)
# install.packages("zoo")
library(zoo)

## Setting working directory
setwd("C:/Users/schoo/OneDrive/Desktop/Grad_Spring2023")
# This will be adjusted base on where the working directory is that contains the
# dataset that will be used for this project.

## Loading Dataset
BNS <- read.csv("STAT Final Data_BNS.csv", stringsAsFactors = TRUE)

## Check to make sure that the dataset is loaded properly
view(BNS)
# Everything appears to be good

## Look at the summary of each dataset to see if there are any questionable factors 
## that we would need to take into consideration
summary(BNS)
# For the most part, everything looks ok.

#_______________________________________________________________________________

## Converting the dataset into tsibble so that it could be easier to manipulate when
## performing time series analysis

BNS$date <- as.Date(BNS$date)
bns <- as_tsibble(BNS, index = date)

## Check to ensure that the dataset is converted into tsibble
bns
# Yes, the data is converted into tsibble and that the number of indices match
# with the number of indices from the original dataset

#_______________________________________________________________________________

## bns (The Bank of Nova Scotia)
bns %>%
  autoplot(close) +
  labs(title = "Closing Price of BNS Stocks",
       y = "Closing Price",
       x = "Time")
# Based on the plot, there appears to be a positive trend where the closing of 
# bns is for the most part moving upward. There could be potential seasonality within
# the data but that would need to be check.

## check to see if there are any gaps within the dataset
timestamps <- bns$date
time_diff <- diff(timestamps)
gaps <- which(time_diff > 1)
gaps
# As expected of stock forecast analysis, we already expected there to be gaps
# within the dataset since we know that the stock market is not open every day of
# week, or even on holidays.

bns <- fill_gaps(bns)
# Now there are two potential ways that we can try to interpolate the data (in 
# the case that we do not want to just remove the NAs)

#### Option 1: imputeTS package
# ?na_interpolation
bns$close <- na_interpolation(bns$close, option = "linear")

#### Option 2: zoo ppackage
# bns$close <- na.appprox(bns$close)

## Performing a STL decomposition model to see determine seasonality and trends
## of BNS stock
bns %>%
  gg_season(close)

bns %>%
  model(STL(close)) %>%
  components() %>%
  autoplot()

stl_bns <- bns %>%
  features(close, feat_stl)
stl_bns

# Based on these the stl decomposition analysis, we can conclude that there is a 
# trend in the dataset (similar to what we concluded initially). We would most
# likely also need to do a data transformation such as a box-cox transformation 
# in order to normalize the data. It would also appear that there could potentially
# be no seasonality within the dataset because of the relatively low seasonal strength.

## First, we should seperate the training and the testing data prior to any transformation
## and modeling
set.seed(123)
split <- round(0.9 * nrow(bns))
train_bns <- bns[1:split, ]
test_bns <- bns[(split + 1):nrow(bns), ]
train_bns
test_bns
# The data appears to be separated now.

## Next we should determine the lambda value for the box-cox transformation
lambda <- train_bns %>%
  features(close, features=guerrero) %>%
  pull(lambda_guerrero)
lambda <- round(lambda,2)
# The lambda value is 0.24.

## Looking at the  STL decomposition of the training set using Lambda
train_bns %>%
  autoplot(box_cox(close, lambda)) +
  labs(title = "Closing Price of BNS Stocks",
       y = "Closing Price",
       x = "Time")

train_bns %>%
  model(STL(box_cox(close, lambda))) %>%
  components() %>%
  autoplot()

# From the transformation, we can see that the training dataset is normalized. We
# can also clearly see the drip in the stock price during the years 2008 and 2020,
# which we can potentially attribute to the recessions (2008 Financial Crisis and COVID).

#_______________________________________________________________________________
## We would like to build a baseline first using Mean, STL, NAIVE, Seasonal Naive, 
## Drift, and TSLM modeling
train_bns_fit <- train_bns %>%
  model(
    Mean = MEAN(box_cox(close, lambda)),
    stlf = decomposition_model(STL(box_cox(close, lambda) ~ trend(window = 7), robust= TRUE), NAIVE(season_adjust)),
    Naive = NAIVE(box_cox(close, lambda)),
    Seasonal_naive = SNAIVE(box_cox(close, lambda)),
    Drift = RW(box_cox(close, lambda) ~ drift()),
    tslm = TSLM(box_cox(close, lambda) ~ trend() + season())
  )

bns_fc <- train_bns_fit %>%
  forecast(h = 839)
view(bns_fc)

## Checking the accuracy of the models an forecast
fabletools::accuracy(bns_fc, bns)
# Based on the acuracy table, the best model to use for the forecast from the baseline
# would be the tslm model.

#_______________________________________________________________________________
## We would like to then compare this model with the ets and the ARIMA model to 
## see if there are any differences

## We need to check if the data is stationary
train_bns %>%
  autoplot(box_cox(close, lambda)) +
  labs(title = "Determining if the Closing Stock Price of BNS is Stationary",
       x = "Year")
# As previously seen, the data does not appear to be stationary so we would need
# to find the appropriate difference (we can clearly see that there are potential 
# seasonality and trend within the data)

## Determining Differences
# First Differences
train_bns %>%
  features(box_cox(close, lambda), unitroot_kpss)
# The p-value came back as 0.01 so we can reject the null hypothesis and conclude 
# that the dataset is in fact not stationary.

train_bns %>%
  mutate(diff_close = difference(box_cox(close, lambda))) %>%
  features(diff_close, unitroot_kpss)
# We can clearly see that first difference is necessary and that only one is enough
# since the p-value after the first difference returns 0.1, which we fail to reject 
# the null hypothesis thus concluding the the dataset is stationary.

# Seasonal Differences
train_bns %>%
  mutate(season_close = box_cox(close, lambda)) %>%
  features(season_close, unitroot_nsdiffs)
# Based on the return of unitroot_nsdiffs(), we can thus conclude that a seasonal 
# difference is not necessary, so only a first difference is needed.

## Plotting the stationary data to look at the ACF and PACF plots
train_bns %>%
  mutate(diff_close = difference(box_cox(close, lambda))) %>%
  gg_tsdisplay(difference(diff_close, 12), 
               plot_type = 'partial', lag = 36) +
  labs(title = "First differenced")
# There appears to be exponential decay in the PACF plot and that the first spike in ACF
# is at 3 so a potential ARIMA model would be (0,1,3).

fit <- train_bns %>%
  model(ARIMA(box_cox(close, lambda)))
report(fit)
# From the default ARIMA model, ARIMA(2,1,3)(0,0,1)[7] w/ drift is the best model
# chosen

#________________________________________________________________________________
## Comparing TLSM, ETS, and ARIMA models
train_bns_final <- train_bns %>%
  model(
    arima013 = ARIMA(box_cox(close, lambda) ~ pdq(0,1,3)),
    default = ARIMA(box_cox(close, lambda)),
    ets = ETS(box_cox(close, lambda)),
    tslm = TSLM(box_cox(close, lambda) ~ trend() + season())
  )
report(train_bns_final)
glance(train_bns_final) %>%
  arrange(AICc) %>%
  select(.model:BIC)
# Based on this, the ARIMA(2,1,3)(0,0,1)[7] w/ drift is the best model chosen as
# it has the lowest AICc value. In addition, it would appears that the ets model 
# is the least accurate model that we could choose.

train_bns_final %>%
  forecast(h = 10) %>%
  fabletools::accuracy(bns)
# From the forecast where h = 10 (days), the model with the best accuracy is also 
# ARIMA(2,1,3)(0,0,1)[7] w/ drift so that would be the model that would we actually
# be using to predict the stock price of BNS.

#_______________________________________________________________________________
# Using the ARIMA model to forecast BNS stock price and determine the accuracy of 
# the forecast
chosen_model <- train_bns %>%
  model(ARIMA(box_cox(close, lambda)))

## Looking at the residual plot and determine white noise
chosen_model %>% 
  gg_tsresiduals()
augment(chosen_model) %>%
  features(.innov, ljung_box, lag=24, dof = 3)
# Based on the p-value given, we reject the null hypothesis, which concludes that
# the residuals does look like white noise.

## Forecasting and Accuracy for ARIMA model
fc_final <- chosen_model %>%
  forecast(h = 839)
fabletools::accuracy(fc_final, bns)
# The RMSE of the ARIMA model forecast is 21.9.

## Plotting the forecasted Closing Price of BNS
chosen_model %>%
  forecast (h = "839 days") %>%
  autoplot(bns) + 
  labs(title = "Projected Closing Stock Price of BNS")
# From here, we can see how accurate were our predictions from the actual stock price.
