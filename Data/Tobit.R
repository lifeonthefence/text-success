# Load required packages
install.packages("censReg")
library(censReg)

install.packages("readr")
library(readr)

install.packages("dplyr")
library(dplyr)

install.packages("caret")
library(caret)

# Import data from a CSV file
my_data <- read_csv("/Users/meltcado/Documents/GitHub/text-success/Data/Processed Dataset TF-IDF Train")
test_data <- read_csv("/Users/meltcado/Documents/GitHub/text-success/Data/Processed Dataset TF-IDF Test")
# Set censoring limits
lower_limit <- 0

# Create a binary variable indicating left-censoring
my_data$left_censored <- ifelse(my_data$retweets <= lower_limit, 1, 0)

y <- my_data$retweets
x1 <- my_data$tweet_length
x2 <- my_data$num_mentions
# Estimate the Tobit model
tobit_model <- censReg(y ~ x1 + x2, left = 0, right = Inf, data = my_data)

# Display the results
summary(tobit_model)

# View coefficients and their standard errors
coef(tobit_model)

# Function to compute the inverse Mills ratio
inverse_mills_ratio <- function(z) {
  return(dnorm(z) / (1 - pnorm(z)))
}

predict_tobit <- function(model, new_data) {
  intercept <- coef(model)["(Intercept)"]
  coef_tweet_length <- coef(model)["tweet_length"]
  coef_num_mentions <- coef(model)["num_mentions"]
  sigma <- coef(model)["sigma"]
  
  linear_predictor <- intercept + new_data$tweet_length * coef_tweet_length + new_data$num_mentions * coef_num_mentions
  expected_y_given_y_above_limit <- linear_predictor + sigma * inverse_mills_ratio(linear_predictor / sigma)
  
  return(ifelse(linear_predictor <= lower_limit, lower_limit, expected_y_given_y_above_limit))
}


test_data$predicted_y <- predict_tobit(tobit_model, test_data)

print(head(test_data$predicted_y))

mse <- mean((test_data$retweets - test_data$predicted_y)^2, , na.rm = TRUE)
print(paste("Mean Squared Error:", mse))




# Calculate the expected value of y, conditional on y > lower_limit
expected_y_given_y_above_limit <- linear_predictor + sigma * inverse_mills_ratio(linear_predictor / sigma)

# Calculate the final predictions
predicted_y <- ifelse(my_data$left_censored == 1, lower_limit, expected_y_given_y_above_limit)

# Add the predictions to the dataset
my_data$predicted_y <- predicted_y

# Export the dataset with predictions to a new CSV file
write_csv(my_data, "my_data_with_predictions.csv")
