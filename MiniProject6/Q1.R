library(tree)
library(randomForest)
library(gbm)

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

#a
# Create a decision tree with psa as response and the rest as potential predictors
tree_pc <- tree(psa ~ ., pc_data)
# Print out a summary then visualize the tree made
summary(tree_pc)
plot(tree_pc)
text(tree_pc, pretty = 0, cex = 0.7)

#b
# Find optimal number of nodes via prune.tree function and cross-validation
cv.pc <- cv.tree(tree_pc, FUN = prune.tree, K=nrow(pc_data))
# Plot the deviance against size
plot(cv.pc$size, cv.pc$dev, type = "b")

# Find size at which you have minimum deviance
# Minimum is usually 8 or 9
cv.pc$size[which.min(cv.pc$dev)]
# But as we can see, the deviance with 4 terminal nodes is really close to 
# those at higher sizes and thus 4 serves as a great elbow point in my opinion 
cv.pc$dev[cv.pc$size == 8]
cv.pc$dev[cv.pc$size == 4]


# Prune the tree with the elbow in mind, our elbow point is at size=4
prune.pc <- prune.tree(tree_pc, best = 4)
plot(prune.pc)
text(prune.pc, pretty = 0, cex = 0.7)


# c.
# Perform bagging with specified parameters and check importance of predictors
bag.pc <- randomForest(psa ~ ., data = pc_data, mtry = 7, ntree = 1000, importance = TRUE)
importance(bag.pc)
varImpPlot(bag.pc)

# d.
# Perform random forest with the specified parameters and check importance of predictors
rf.pc <- randomForest(psa ~ ., data = pc_data, mtry = round(7/3), ntree = 1000, importance = TRUE)
importance(rf.pc)
varImpPlot(rf.pc)

# e.
# Perform boosting with gbm and specified parameters and check importance of predictors
boost.pc <- gbm(psa ~ ., data = pc_data, distribution = "gaussian", 
                    n.trees = 1000, interaction.depth = 1, shrinkage=0.01)
summary(boost.pc)


# Make a function to run LOOCV on all models we want to evaluate
LOOCV_tree <- function(){
  # Set k to number of observations for LOOCV
  k <- nrow(pc_data)
  # Select indices for each fold
  indices <- sample(1:nrow(pc_data))
  folds <- cut(indices, breaks = k, labels = FALSE)
  # Establish structures to store MSE data in
  unpruned_MSEs <- c()
  pruned_MSEs <- c()
  bagged_MSEs <- c()
  rf_MSEs <- c()
  boost_MSEs <- c()
  
  # Iterate through each fold
  for (i in 1:k){
    # Make validation and training data
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- pc_data[val_indices,]
    train_data <- pc_data[-val_indices,]
    
    # For each model, compute MSE and store it
    # Base Decision Tree model 
    train_tree_pc <- tree(psa ~ ., train_data)
    unpruned_MSE <- (val_data$psa - predict(train_tree_pc, val_data))^2
    unpruned_MSEs <- c(unpruned_MSEs, unpruned_MSE)
    
    # Pruned Tree model with potentially optimal best number of terminal nodes
    train_pruned_tree <- prune.tree(train_tree_pc, best=4)
    pruned_MSE <- (val_data$psa - predict(train_pruned_tree, val_data))^2
    pruned_MSEs <- c(pruned_MSEs, pruned_MSE)
    
    # Bagging model evaluation
    train_bag <- randomForest(psa ~ ., data = train_data, mtry = 7, ntree = 1000, importance = TRUE)
    bagged_MSE <- (val_data$psa - predict(train_bag, newdata = val_data))^2
    bagged_MSEs <- c(bagged_MSEs, bagged_MSE)
    
    # Random forest model evaluation
    train_rf <- randomForest(psa ~ ., data = train_data, mtry = round(7/3), ntree = 1000, importance = TRUE)
    rf_MSE <- (val_data$psa - predict(train_rf, newdata = val_data))^2
    rf_MSEs <- c(rf_MSEs, rf_MSE)
    
    # Boosted model evaluation
    train_boost <- gbm(psa ~ ., data = train_data, distribution = "gaussian", 
                       n.trees = 1000, interaction.depth = 1, shrinkage=0.01)
    boost_MSE <- (val_data$psa - predict(train_boost, newdata = val_data, n.trees = 1000))^2
    boost_MSEs <- c(boost_MSEs, boost_MSE)
  }
  # Return mean of MSEs per model
  result <- c(
              mean(unpruned_MSEs),
              mean(pruned_MSEs),
              mean(bagged_MSEs),
              mean(rf_MSEs),
              mean(boost_MSEs)
            )
  # Store result in named vector
  return(setNames(result, c("unpruned_est_MSE", "pruned_est_MSE", "bagged_est_MSE", "rand_forest_est_MSE", "boosted_est_MSE")))
}
# Store and print MSEs
MSEs <- LOOCV_tree()
MSEs
