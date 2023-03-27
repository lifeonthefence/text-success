

install.packages("censReg")
library(censReg)

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

train_data <- read.csv("/Users/meltcado/Documents/GitHub/text-success/Data/Processed Dataset Train")
test_data <- read_csv("/Users/meltcado/Documents/GitHub/text-success/Data/Processed Dataset Test")

attach(train_data)
#name_changes <- c("Negative Score" = "negative_score", "Compound Score" = "compound_score", "C" = "new_C")
dependent_var <- train_data$retweets
independent_vars <- c(train_data$tweet_length, train_data$num_mentions)

# tobit_formula <- as.formula(paste(dependent_var, "~", paste(independent_vars, collapse = " + ")))

# Set censoring limits
left_limit <- 0
right_limit <- Inf

tobit_model <- vglm(retweets ~ Compound.Score, family = tobit(Lower = 0, Upper = Inf), data = train_data)
#tobit_model <- vglm(retweets ~ tweet_length + num_mentions, formula = tobit(Lower = left_limit, Upper = right_limit), data = train_data)


test_data$predicted_values <- predict(tobit_model, newdata = test_data, type = "response")

# Choose your evaluation metric, e.g., mean squared error (MSE)
mse <- mean((test_data[[dependent_var]] - test_data$predicted_values)^2)
print(paste("Mean Squared Error:", mse))

summary(tobit_model)
print(head(test_data$predicted_values))
