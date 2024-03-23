
# Read in planet data from planet.csv
planet_data <- read.csv("planet.csv")

# Number of observations and features
nrow(planet_data)
ncol(planet_data)

# The standard deviation of planet data features can be large and vary greatly
sd(planet_data$Period)
# Range via visual inspection of values is also quite different.
max(planet_data$Mass) - min(planet_data$Mass)
max(planet_data$Period) - min(planet_data$Period)
max(planet_data$Eccentricity) - min(planet_data$Eccentricity)
# As such I decided it would be a good idea to standardize the variables

# Univariate distributions
par(mfrow=c(1,3))
hist(planet_data$Mass, main="Mass")
hist(planet_data$Period, main="Period")
hist(planet_data$Eccentricity, main="Eccentricity")

# Bivariate relations
cor(planet_data)
plot(planet_data$Mass, planet_data$Period)
plot(planet_data$Mass, planet_data$Eccentricity)
plot(planet_data$Eccentricity, planet_data$Period)

# Variable index of normalized planet data
plot(planet_data$Period/sum(planet_data$Period), type="l")
lines(planet_data$Mass/sum(planet_data$Mass), col="red")
lines(planet_data$Eccentricity/sum(planet_data$Eccentricity), col = "blue")
legend("topleft",legend=c("Period","Mass","Eccentricity"),
       pch=15, col=c("black", "red","blue"))

# Variable index of scaled planet data
scaled_planet_data <- data.frame(scale(planet_data))
plot(scaled_planet_data$Period, type="l")
lines(scaled_planet_data$Mass, col="red")
lines(scaled_planet_data$Eccentricity, col = "blue")
legend("topleft",legend=c("Period","Mass","Eccentricity"),
       pch=15, col=c("black", "red","blue"))

# Hierarchical clustering of scaled planet data using metric (euclidean) distance
# and complete link
hc.complete <- hclust(dist(scaled_planet_data), method = "complete")
plot(hc.complete, main = "Complete Linkage HC Dendrogram", 
     xlab = "", sub = "", cex = 0.6)

# Get cluster assignments of all observations
three_cut <- cutree(hc.complete, k = 3)

# Find cluster-specific means by selecting rows matching cutree results
clust1 <- planet_data[which(three_cut == 1),]
# Then applying mean function over columns to figure out feature means
clust1_means <- apply(clust1, 2, mean)

# Repeat for cluster 2 and cluster 3 observations respectively
clust2 <- planet_data[which(three_cut == 2),]
clust2_means <- apply(clust2, 2, mean)

clust3 <- planet_data[which(three_cut == 3),]
clust3_means <- apply(clust3, 2, mean)

# Show pairwise plot of original planet data with colors based off of cutree results
# to indicate which observation is in which cluster
pairs(planet_data, pch=19, col=(three_cut+1), upper.panel = NULL)

# Repeat the process but with K-means where K = 3
km.out <- kmeans(scaled_planet_data, 3, nstart = 20)
# Print out the centers values to get cluster-specific means
km.out$centers

# Display pairwise plot of original planet data with 
# colors based off of K-means cluster assignments
plot(planet_data, col = (km.out$cluster + 1), 
     main = "K-Means Clustering Results with K=3", 
     pch = 19, cex = 1, upper.panel = NULL)
