install.packages("readr")
library(readr)

install.packages("dplyr")
library(dplyr)
library(ggplot2)
library(lattice)
install.packages("caret")
library(caret)
library(stats4)
library(splines)
install.packages("VGAM")
library(VGAM)


my_data <- read.csv("/Users/meltcado/Documents/GitHub/text-success/Data/Processed Dataset TF-IDF.csv")
# Set the seed for reproducibility
set.seed(123)

# Split the dataset into training and test sets
splitIndex <- createDataPartition(my_data$retweets, p = 0.75, list = FALSE)
train_data <- my_data[splitIndex, ]
test_data <- my_data[-splitIndex, ]



#test_data <- read_csv("/Users/meltcado/Documents/GitHub/text-success/Data/Processed Dataset TF-IDF Test")

dependent_var <- train_data$retweets
independent_vars <- c(train_data$tweet_length, train_data$num_mentions)
tobit_model <- vglm(y_obs ~ x, family = tobit(Lower = 0), data = train_data)

tobit_model <- vglm(train_data$likes ~ train_data$Compound.Score + train_data$tweet_length + train_data$user_followers, family = tobit(Lower = 0, Upper = Inf), data = train_data)
# Predict retweets using the tobit model for the test set
predicted_retweets <- predict(tobit_model, newdata = test_data, type = "response")

# Create a new data frame with the same number of rows as test_data
results <- data.frame(test_data)

# Add the predicted values to the results data frame
results$predicted_values <- predicted_retweets

predictions <- predict(tobit_model, test_data, type = "response")
test_data$predicted_values <- predict(tobit_model, newdata = test_data, type = "response")

# Choose your evaluation metric, e.g., mean squared error (MSE)
mse <- mean((test_data[[dependent_var]] - test_data$predicted_values)^2)
print(paste("Mean Squared Error:", mse))