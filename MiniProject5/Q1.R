library(pls)

# Read in pc data and remove subject column
pc_data <- read.csv("prostate_cancer.csv")
pc_data <- pc_data[,-1]

# Treat vesinv as a qualitative variable
pc_data$vesinv <- factor(pc_data$vesinv, order=F, levels = c(0, 1))

# Conduct a natural log transformation on the response
# to adjust it's distribution to something more appropriate.
pc_data[, 1] <- log(pc_data[, 1])
hist(pc_data$psa)

# Calculate LOOCV estimate of MSE via pcr regression on prostate cancer data
prostate_pcr <- pcr(psa ~ ., data = pc_data, scale = TRUE, validation="LOO")
# Summary will provide more information
summary(prostate_pcr)

# Store MSE information for all components
pcr_MSE <- MSEP(prostate_pcr)$val[1, 1,]

# Find number of components corresponding to minimum MSE
which.min(pcr_MSE)
# Confirmation of minimum MSE point
validationplot(prostate_pcr, val.type = "MSEP")

# Test MSE
min(pcr_MSE)

# Train MSE
prostate_pcr_pred <- predict(prostate_pcr, pc_data, ncomp = 1)
mean((prostate_pcr_pred - pc_data$psa)^2)

# Calculate LOOCV estimate of MSE via a PLS regression on the prostate cancer data
prostate_pls <- plsr(psa ~ ., data = pc_data, scale = TRUE, validation="LOO")

# Store MSE information for all components for the PLS regression
pls_MSE <- MSEP(prostate_pls)$val[1, 1,]

# Find number of components corresponding to minimum MSE for the PLS regression
which.min(pls_MSE)
# Confirm minimum MSE point for the PLS regression
validationplot(prostate_pls, val.type="MSEP")

# Test MSE
min(pls_MSE)
