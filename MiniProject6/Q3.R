library(keras)

# Obtain boston dataset information
boston <- dataset_boston_housing()
# Separate boston dataset into train and test data
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston

# Obtain mean, stdev, and scale the data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# Specify a function to create a 2 hidden layer model with 64 hidden units
# using ReLU activation and linear 1-node output
build_model <- function(){
  # specify the model
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
}

# K-fold CV
# Specify 4 folds
k <- 4
# Partition indices and determine folds
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)
# Supply num epochs and create variable to track histories
num_epochs <- 200
all_mae_histories <- c()
for (i in 1:k){
  cat("Processing fold #", i, "\n")
  # Partition data into validation and training data
  val_indices <- which(folds == i, arr.ind = TRUE) # prepares the validation data: data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,] # prepares the training data: data from all other partitions
  partial_train_targets <- train_targets[-val_indices]
  
  # Use build model function to make architecture and compile
  model <- build_model()
  
  # Fit and track history on partial training data
  history <- model %>% fit(partial_train_data, partial_train_targets,
                validation_data = list(val_data, val_targets),
                epochs = num_epochs, batch_size = 16, 
                verbose = 0) # trains the model in silent mode (verbose = 0)
  # Obtain validation MAE from history
  mae_history <- history$metrics$val_mae
  # Store the MAE data
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}
# Show MAE per epoch
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

# Plot validation MAE against epoch.
# We can see from the results that there is not much improvement after epoch 75
plot(validation_mae ~ epoch, average_mae_history, ylim = c(2, 5), type ="l")


# a.
# Repeat previous process of 4-fold CV for 2-layer, 64 hidden unit model with early stopping
all_scores <- c()
for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # prepares the validation data: data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,] # prepares the training data: data from all other partitions
  partial_train_targets <- train_targets[-val_indices]
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
  
  opt_history <- model %>% fit(partial_train_data, partial_train_targets, epochs = 75, 
                               batch_size = 16, verbose = 0)
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results["mae"])
}
# Print mean MAE
mean(all_scores)



# b.
# Repeat previous process of 4-fold CV for 1-layer, 128 hidden unit model with early stopping
all_scores <- c()
for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # prepares the validation data: data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,] # prepares the training data: data from all other partitions
  partial_train_targets <- train_targets[-val_indices]
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 128, activation = "relu", 
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
  opt_history <- model %>% fit(partial_train_data, partial_train_targets, epochs = 75, 
                               batch_size = 16, verbose = 0)
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results["mae"])
}
mean(all_scores)



# c.
# Repeat previous process of 4-fold CV for 2-layer, 64 hidden unit model with early stopping
# with L2 regularization
all_scores <- c()
for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # prepares the validation data: data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,] # prepares the training data: data from all other partitions
  partial_train_targets <- train_targets[-val_indices]
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[2],
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dense(units = 64, activation = "relu",
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
  
  opt_history <- model %>% fit(partial_train_data, partial_train_targets, epochs = 75, 
                               batch_size = 16, verbose = 0)
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results["mae"])
}
mean(all_scores)




# d.
# Repeat previous process of 4-fold CV for 1-layer, 128 hidden unit model with early stopping
# with L2 regularization
all_scores <- c()
for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # prepares the validation data: data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,] # prepares the training data: data from all other partitions
  partial_train_targets <- train_targets[-val_indices]
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 128, activation = "relu", 
                input_shape = dim(train_data)[2],
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
  opt_history <- model %>% fit(partial_train_data, partial_train_targets, epochs = 75, 
                               batch_size = 16, verbose = 0)
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results["mae"])
}
mean(all_scores)



# e.
# Take our considered optimal model and fit it again
model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", 
              input_shape = dim(train_data)[2],
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dense(units = 64, activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dense(units = 1)
# compile the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae") # mean absolute error
)

opt_history <- model %>% fit(train_data, train_targets, epochs = 75, batch_size = 16, verbose = 0)

# Evaluate the model on the test data and print out the results
results <- model %>% evaluate(test_data, test_targets, verbose = 0)
results
