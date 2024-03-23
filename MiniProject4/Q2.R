library(boot)
library(leaps)
library(glmnet)

# Read in prostate cancer data
pc_data <- read.csv("prostate_cancer.csv")
# Eliminate subject number feature
pc_data <- pc_data[,-1]
# Treat vesinv as a qualitative variable
pc_data$vesinv <- factor(pc_data$vesinv, order=F, levels = c(0, 1))

# Conduct a natural log transformation on the response 
# to adjust it's distribution to something more appropriate.
pc_data[, 1] <- log(pc_data[, 1])
hist(pc_data[, 1])

# Make a full model of a linear regression with psa as response and all features as predictors
# Calculate test MSE via LOOCV
full_model <- glm(psa ~ ., data = pc_data)
cv.err <- cv.glm(pc_data, full_model)
cv.err$delta

# Find a reasonable subset of features to implement a linear regression model with
# via the best-subset selection accounting for the best adjusted R^2.
regfit_full = regsubsets(psa ~ ., data = pc_data, nvmax = 7)
regfit_summ <- summary(regfit_full)
which.max(regfit_summ$adjr2)
coef(regfit_full, 4)
# Find the test MSE via LOOCV
cv.glm(pc_data, glm(psa ~ cancervol + benpros + vesinv + gleason, data=pc_data))$delta

# Find a reasonable subset of features to implement a model with using forward
# subset selection with best adjusted R^2 value.
fit.fwd = regsubsets(psa ~ ., data = pc_data, nvmax = 7, method = "forward")
summary(fit.fwd)
which.max(summary(fit.fwd)$adjr2)
coef(fit.fwd, 4)
cv.glm(pc_data, glm(psa ~ cancervol + benpros + vesinv + gleason, data=pc_data))$delta

# Find a reasonable subset of features to implement a model with using backward
# subset selection with best adjusted R^2 value.
fit.bwd = regsubsets(psa ~ ., data = pc_data, nvmax = 7, method = "backward")
summary(fit.bwd)
which.max(summary(fit.bwd)$adjr2)
coef(fit.bwd, 4)
cv.glm(pc_data, glm(psa ~ cancervol + benpros + vesinv + gleason, data=pc_data))$delta

# Select response and feature data as y and X respectively
y <- pc_data$psa
X <- model.matrix(psa ~ ., pc_data)[, -1]
# Set up a grid of potential lambda values
grid <- 10^seq(10, -2, length = 100)
# Using alpha = 0, conduct a ridge regression
ridge.mod <- glmnet(X, y, alpha = 0)
# Use LOOCV to determine the best penalty parameter
cv_ridge <- cv.glmnet(X, y, alpha = 0)
best_ridge_lambda <- cv_ridge$lambda.min
best_ridge_lambda

# Use the best lambda to find the best ridge regression model
best_ridgemod <- glmnet(X, y, alpha = 0, lambda = best_ridge_lambda)
coef(best_ridgemod)
# Predict the values using the best ridge regression model and find the MSE
y_pred <- predict(ridge.mod, s = best_ridge_lambda, newx = X)
mean((y_pred - y)^2)

# Using alpha=1, conduct lasso
lasso.mod <- glmnet(X, y, alpha=1)
# Use LOOCV to find the best penalty parameter
cv_lasso <- cv.glmnet(X, y, alpha=1)
best_lasso_lambda <- cv_lasso$lambda.min
best_lasso_lambda
# Make the best lasso model with the best lambda value
best_lassomod <- glmnet(X, y, alpha=1, lambda=best_lasso_lambda)
coef(best_lassomod)
# Predict the appropriate values using the best lasso model and calculate MSE
y_pred <- predict(lasso.mod, s = best_lasso_lambda, newx = X)
mean((y_pred-y)^2)
