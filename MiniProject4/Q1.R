library(boot)

# Obtain oxygen monitoring data from the oxygen saturation txt file
o2_data <- read.table("oxygen_saturation.txt", header = T, sep="\t")
# Obtain data from pulse oximeter and oxygen saturation monitor separately
pos_data <- o2_data[, 1]
osm_data <- o2_data[, 2]
# Find the vector of differences between the two methods per individual patient
diff_data <- pos_data - osm_data
# See the number of patients and the 
# distribution of values found using the two methods
nrow(o2_data)
hist(pos_data)
hist(osm_data)
# Print out the mean values for pulse oximeter readings 
# and oxygen saturation monitors separately
mean(pos_data)
mean(osm_data)

# Point estimate of the difference in means between the two methods
pt_estim_mean <- mean(pos_data) - mean(osm_data)

# Bias of the vector produced from looking at the difference in methods per patient (diff_data)
mean(diff_data) - pt_estim_mean
# Standard error of diff_data
st_err <- sd(diff_data) / sqrt(72)
# Confidence interval via percentile approach: quantile(diff_data, c(0.025, 0.975))
# 95% confidence interval of diff_data 
c(mean(diff_data) - st_err, mean(diff_data) + st_err)


# Number of entries
n <- nrow(o2_data)
# number of resamples
b <- 1000

# Track estimates of difference of means
estimates <- c()
for(i in 1:b){
  # For each resample, sample from 1 to n with replacement
  indices_sampled <- sample(1:n, n, replace = T)
  # Use these indices to select rows to calculate the difference of means from
  estimates <- c(estimates, 
                 mean(pos_data[indices_sampled]) - mean(osm_data[indices_sampled])
                )
}
# Metrics you can evaluate
# Mean estimate of the difference of values between the two methods
mean(estimates)
# Variance found via bootstrapping
var(estimates)
# Bias found via bootstrapping
mean(estimates) - pt_estim_mean
# Standard Error found via bootstrapping
sd(estimates) / sqrt(72)
# Confidence Interval (via percentile approach)
quantile(estimates, c(.025, .975))

# Establish difference of means function
mean.fn <- function(x, indices) {
  # Return difference of means over two methods for certain indices
  result <- mean(x[indices,1]) - mean(x[indices,2])
  return(result)
}
# Test the function for the original data
mean.fn(o2_data, 1:nrow(o2_data))

# Conduct a boostrapping using the boot package.
mean.boot <- boot(o2_data, mean.fn, R = 1000)
mean.boot
# Estimate for difference of means
mean.boot$t0
# Bias for estimate of difference of means
mean(mean.boot$t) - mean.boot$t0
# Standard errorof the estimate of difference of means
sd(mean.boot$t) / sqrt(72)
# Confidence intervals produced with 95% levels 
boot.ci(mean.boot, type = "perc")
