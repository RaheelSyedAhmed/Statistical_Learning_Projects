# Read in prostate cancer data
pc_data <- read.csv("prostate_cancer.csv")
# Eliminate subject number feature 
pc_data <- pc_data[,-1]

# Convert gleason and treat vesinv as qualitative variables
pc_data$vesinv <- factor(pc_data$vesinv, order=F, levels = c(0, 1))
pc_data$gleason <- factor(pc_data$gleason, order=F, levels = c(6, 7, 8))

# Part a
# Preliminary findings
nrow(pc_data)
colnames(pc_data)
summary(pc_data$vesinv) # There's more people without seminal vesicle invasion than with
summary(pc_data$gleason) # There's a mix of people with varying gleason scores.
hist(pc_data$age) # Most people with pancreatic cancer information are older (50-70+)
cor(pc_data[,unlist(lapply(pc_data, is.numeric))])
# age seems to have a high correlation with the amount of prostatic hyperplasia
#, which is indicative of early stages of prostatic abnormality
# Capsular penetration, which indicates the outgrowth of cancerous tissue, has a correlation with cancer volume


# Part b
# Examine distribution of psa to determine if it's an appropriate response variable.
hist(pc_data[, 1])

# Since psa is not, transform it with a natural log transformation and check again.
pc_data[, 1] <- log(pc_data[, 1])
hist(pc_data[, 1])

# Part c - For each predictor, fit a simple linear regression model to predict the response
summary(glm(psa ~ cancervol, data = pc_data))$coefficients  #Significance found
summary(glm(psa ~ weight, data = pc_data))$coefficients
summary(glm(psa ~ age, data = pc_data))$coefficients        
summary(glm(psa ~ benpros, data = pc_data))$coefficients
summary(lm(psa ~ vesinv, data = pc_data))$coefficients      #Significance found
summary(glm(psa ~ capspen, data = pc_data))$coefficients    #Significance found
summary(lm(psa ~ gleason, data = pc_data))$coefficients     #Significance found
par(mfrow=c(2,2))
plot(pc_data$cancervol, pc_data$psa, xlab = "cancervol", ylab="psa")
plot(pc_data$vesinv, pc_data$psa, xlab = "vesinv", ylab="psa")
plot(pc_data$capspen, pc_data$psa, xlab = "capspen", ylab="psa")
plot(pc_data$gleason, pc_data$psa, xlab = "gleason", ylab="psa")

# Part d
summary(glm(psa ~ ., data = pc_data))
# We can reject the null hypothesis for cancervol, benpros, vesinv, and gleason (linear)

# Part e
# I excluded capspen as it did not seem statistically significant when other predictors were involved
# I excluded benpros as it did not have statistical significance to the psa response until other predictors were accounted for
final_model <- lm(psa ~ cancervol + vesinv + gleason, data = pc_data)
summary(final_model)
final_model$coefficients

# Part f
# psa = psa = 1.60467 + 0.05875*cancervol + 0.6259312*vesinv1 + 0.3543990*gleason7 + 0.7863444*gleason8


#Part g
sample_patient <- data.frame(mean(pc_data$cancervol), names(sort(table(pc_data$vesinv)))[-1], names(sort(table(pc_data$gleason)))[c(-1,-2)])
colnames(sample_patient) <- c("cancervol", "vesinv", "gleason")
predict(final_model, sample_patient)

