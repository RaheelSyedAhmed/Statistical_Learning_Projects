library(boot)
library(MASS)
library(glmulti)
library(glmnet)

# Read in german credit dataset and store as a dataframe
german_data <- read.csv("germancredit.csv", stringsAsFactors = T)
# Produce a logistic regression model using all the predictors
all_lr <- glm(Default ~ ., data = german_data, family = "binomial")
summary(all_lr)
coef(all_lr)

# Produce a logistic regression model using no predictors (yields intercept only model)
empty_lr <- glm(Default ~ 1, data=german_data, family = "binomial")
summary(empty_lr)


# Use boot package to estimate LOOCV for log-reg models
# Make cost function
cost <- function(r, pi = 0){mean(abs(r - pi) > 0.5)}
# Calculate LOOCV for both the full model and the null model
all_lr.err <- cv.glm(german_data, all_lr, cost, K=nrow(german_data))
all_lr.err$delta
empty_lr.err <- cv.glm(german_data, empty_lr, cost, K=nrow(german_data))
empty_lr.err$delta

# Find best subset selection model with fewer predictors to improve run time.
# AIC is our method of evaluation, run exhaustive checks for the best model.
best_sub <- glmulti(Default ~ checkingstatus1 + duration + history + purpose + savings + 
                      others + installment + status + amount + otherplans + foreign + 
                      age + housing + tele, data=german_data, level=1, 
        method="h", crit="aic", confsetsize = 2, 
        plotty=F, report=F, 
        fitfunction = "glm", family=binomial)
# Plug in the formula to the glm function and run a logistic regression
best_lr <- glm(best_sub@formulas[[1]], data=german_data, family="binomial")
summary(best_lr)
coef(best_lr)
# Print LOOCV estimate of test error.
best_lr.err <- cv.glm(german_data, best_lr, cost, K=nrow(german_data))
best_lr.err$delta

# Find forward stepwise selection model with AIC as our method of evaluation
# I started with the null model and progressed forward with a specified scope
forward_step <- stepAIC(empty_lr, 
                        scope = list(lower=empty_lr, upper=all_lr), 
                        direction = "forward")
# Feed the formula found into the glm function and conduct a logistic regression
forward_lr <- glm(Default ~ checkingstatus1 + duration + history + purpose + savings + 
                    others + installment + status + amount + otherplans + foreign + 
                    age + housing + tele, data = german_data, family="binomial")
summary(forward_lr)
coef(forward_lr)
# Calculate LOOCV estimate of test error
forward_lr.err <- cv.glm(german_data, forward_lr, cost, K=nrow(german_data))
forward_lr.err$delta

# Find backward stepwise selection model with AIC as our method of evaluation
# I started with the full model and moved backward with the specified scope
backward_step <- stepAIC(all_lr, 
                        scope = list(lower=empty_lr, upper=all_lr), 
                        direction = "backward")
# Feed the formula found into the glm function and conduct a logistic regression
backward_lr <- glm(Default ~ checkingstatus1 + duration + history + purpose + amount + 
                     savings + installment + status + others + age + otherplans + 
                     housing + tele + foreign, data = german_data, family = "binomial")
summary(backward_lr)
coef(backward_lr)
# Calculate LOOCV estimate of test error
backward_lr.err <- cv.glm(german_data, backward_lr, cost, K=nrow(german_data))
backward_lr.err$delta

# Assign response and feature matrix to y and X respectively
y <- german_data$Default
X <- model.matrix(Default ~ ., german_data)[, -1]
# Set up a grid of potential lambda values
grid <- 10^seq(10, -2, length = 100)
# Conduct a ridge regression where alpha=0
ridge.mod <- glmnet(X, y, alpha = 0)
# Use LOOCV to determine the best penalty parameter
cv_ridge <- cv.glmnet(X, y, alpha = 0)
best_ridge_lambda <- cv_ridge$lambda.min
best_ridge_lambda
# Make the best ridge regression model using the best lambda
best_ridgemod <- glmnet(X, y, alpha = 0, lambda = best_ridge_lambda)
coef(best_ridgemod)

# Predict the values of the training data using the best ridge regression model
y_pred <- predict(ridge.mod, s = best_ridge_lambda, newx = X)
# Compare the values to the true values to find test error
mean((y_pred - y)^2)

# Conduct a lasso model with alpha=1
lasso.mod <- glmnet(X, y, alpha=1)
# Use LOOCV to find the best lambda
cv_lasso <- cv.glmnet(X, y, alpha=1)
best_lasso_lambda <- cv_lasso$lambda.min
best_lasso_lambda
# Formulate the best lasso model with the best lasso lambda
best_lassomod <- glmnet(X, y, alpha=1, lambda=best_lasso_lambda)
coef(best_lassomod)
# Predict the values of our training data and compare to the true values to find the test error
y_pred <- predict(lasso.mod, s = best_lasso_lambda, newx = X)
mean((y_pred-y)^2)
