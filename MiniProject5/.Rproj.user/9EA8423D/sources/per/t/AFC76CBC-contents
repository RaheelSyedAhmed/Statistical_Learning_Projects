library(datasets)

# Read in state data
state_data <- state.x77

# See dimensions of data
nrow(state_data)
ncol(state_data)

# correlation matrix of data
cor(state_data)

#Standard deviation of population, income, and area
sd(state_data[,1])
sd(state_data[,2])
sd(state_data[,8])

# Display histograms of all features
par(mfrow=c(2,4))
for (feature in 1:ncol(state_data)) {
  hist(state_data[,feature], main=colnames(state_data)[feature])
}

# Scale state data for PCA
state_data <- scale(state_data)
state_pca <- prcomp((state_data), center = T, scale = T)

# Eigenvectors, scores, and variances provided via rotation, x, and stdev squared
state_pca$rotation
state_pca$x
state_pca_var <- state_pca$sdev^2

# Find proportion of variance explained by normalizing variances
state_pve <- state_pca_var / sum(state_pca_var)

# Prove that the cumulative sum of the PVEs is 1.
cumsum(state_pve)

par(mfrow=c(1,1))
# Plot PVE on a scree plot to find number of principal components to use
plot(state_pve, 
     xlab = "Principal Component", ylab = "Proportion of Variance Explained", 
     ylim = c(0,1), type = 'b')
    #ylim is 0 to 1 because it's a probability

# Plotting cumulative sum can also help you to make the same decision
plot(cumsum(state_pve), 
     xlab = "Principal Component", ylab = "Proportion of Variance Explained", 
     ylim = c(0,1), type = 'b')
    #ylim is 0 to 1 because it's a probability

# Correlation between variable X and PC y is equivalent to (loading value for that X and y) / (standard deviation of that feature X).
# Due to standardization we can see that standard deviations of each feature is 1.
for(i in 1:ncol(state_data)){
  print(paste0("standard deviation of feature X", i, ": ", sd(state_data[,i])))
}

# Correlation between all features and PCs 1 and 2 can thus be shown via this cropping of the rotation matrix
# multiplied by the corresponding standard deviation associated with that principal component

state_corr <- cbind(data.frame(state_pca$rotation[,1]*state_pca$sdev[1]), state_pca$rotation[,2]*state_pca$sdev[2])
colnames(state_corr) <- c("PC1", "PC2")


# Proportion of variance explained summed for the first two Principal Components.
cumsum(state_pve)[1:2]

# Biplot of scores and loadings
biplot(state_pca, scale=0)
