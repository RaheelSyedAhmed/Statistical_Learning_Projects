library(keras)

# Get mnist data
mnist <- dataset_mnist()
# Partition training and test images from mnist
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Reshape and scale data where needed
train_images <- array_reshape(train_images, c(60000, 28*28)) # matrix
train_images <- train_images/255 # ensures all values are in [0, 1]
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# Obtain categorical versions of training and test labels
cat_train_labels <- to_categorical(train_labels)
cat_test_labels <- to_categorical(test_labels)

# Store results in a dataframe
results_df <- data.frame(matrix(nrow=0, ncol=7))
colnames(results_df) <- c("layers", "units", "epochs", "dropout", "L2_reg", "train_acc", "test_acc")

# Make a function to create different network architectures depending on parameters
# The function will also compile each network, fit the network on training data, and evaluate on testing data.
exec_network <- function(num_layers, num_units, num_epochs, dropout, L2_reg){
  # Make models with 2 layers
  if(num_layers == 2){
    # Base type of model with specified number of nodes
    network <- keras_model_sequential() %>%
      layer_dense(units = num_units, activation = "relu", input_shape = c(28*28)) %>%
      layer_dense(units = num_units, activation = "relu", input_shape = c(28*28)) %>%
      layer_dense(units = 10, activation = "softmax")
  }
  else{
    # Models with only 1 layer
    if(dropout){
      # Make a model with dropout
      network <- keras_model_sequential() %>%
        layer_dense(units = num_units, activation = "relu", input_shape = c(28*28)) %>%
        layer_dropout(rate = 0.5) %>% 
        layer_dense(units = 10, activation = "softmax")
    }
    else if(L2_reg){
      # Make a model with L2 regularization and lambda = 0.001
      network <- keras_model_sequential() %>%
        layer_dense(units = num_units, activation = "relu", input_shape = c(28*28),
                    kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_dense(units = 10, activation = "softmax")
    }
    else{
      # Make a model without dropout or L2 regularization
      network <- keras_model_sequential() %>%
        layer_dense(units = num_units, activation = "relu", input_shape = c(28*28)) %>%
        layer_dense(units = 10, activation = "softmax")
    }
  }
  # Compile the network
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",  # loss function to minimize
    metrics = c("accuracy") # monitor classification accuracy
  )
  # Fit the network on the training data and categorical train labels. 
  # This is where number of epochs parameters is passed to the fitting function
  history <- network %>% fit(train_images, cat_train_labels, epochs = num_epochs, batch_size = 128, verbose = F)
  
  # Evaluate the network on test images and the categorical test labels
  metrics <- network %>% evaluate(test_images, cat_test_labels, verbose = F)
  
  # Use metrics to report on test accuracy and history to report on training accuracy
  # Store that data in the results dataframe
  return(data.frame(layers=num_layers, units=num_units, epochs=num_epochs, 
                    dropout=dropout, L2_reg=L2_reg,
                    train_acc=history$metrics$accuracy[num_epochs], test_acc=metrics["accuracy"][[1]]))
  
}
# Try different permutations of networks with different numbers 
# of layers, units, epochs, and dropout and L2 regularization status
results_df <- rbind(results_df, exec_network(num_layers = 1, num_units = 512, num_epochs = 5, dropout = F, L2_reg = F))
results_df <- rbind(results_df, exec_network(num_layers = 1, num_units = 512, num_epochs = 10, dropout = F, L2_reg = F))
results_df <- rbind(results_df, exec_network(num_layers = 1, num_units = 256, num_epochs = 5, dropout = F, L2_reg = F))
results_df <- rbind(results_df, exec_network(num_layers = 1, num_units = 256, num_epochs = 10, dropout = F, L2_reg = F))

results_df <- rbind(results_df, exec_network(num_layers = 2, num_units = 512, num_epochs = 5, dropout = F, L2_reg = F))
results_df <- rbind(results_df, exec_network(num_layers = 2, num_units = 512, num_epochs = 10, dropout = F, L2_reg = F))
results_df <- rbind(results_df, exec_network(num_layers = 2, num_units = 256, num_epochs = 5, dropout = F, L2_reg = F))
results_df <- rbind(results_df, exec_network(num_layers = 2, num_units = 256, num_epochs = 10, dropout = F, L2_reg = F))

results_df <- rbind(results_df, exec_network(num_layers = 1, num_units = 512, num_epochs = 5, dropout = F, L2_reg = T))
results_df <- rbind(results_df, exec_network(num_layers = 1, num_units = 512, num_epochs = 5, dropout = T, L2_reg = F))

# store results for retrial attempts
saved2_df <- results_df
