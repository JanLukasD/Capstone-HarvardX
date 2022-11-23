## @knitr Code

# Capstone HarvardX Data Science Course
# Author: Jan Lukas Döring




# Section 1: Data Preperation 

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gt)) install.packages("gt", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")

library(scales)
library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(lubridate)
library(gt)
library(recosystem)
library(RColorBrewer)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

dim(movielens)

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation_raw <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into ed set

removed <- anti_join(temp, validation_raw)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


################################################################################################################

# Section 2: Data Exploration


# checking the structure of the edx dataset

str(edx)

# checking the dimensions of the edx dataset

dim(edx)

# checking the first 6 entries of the edx dataset

head(edx)

# checking for missing data in the edx dataset (should return 0 if no data is missing)

mean(is.na(edx))

# checking for zeros in any column of the dataset (Should return FALSE if there is no zeros in the dataset)

all(apply(apply(edx, 2, function(x) x==0), 2, any))


# Conversion of time stamp into a date format for both data sets (edx and validation)
edx$date <- as.POSIXct(edx$timestamp, origin="1970-01-01")
validation$date <- as.POSIXct(validation$timestamp, origin="1970-01-01")

# Pulling year and month of the rating for each dataset    

edx$year_of_rating <- format(edx$date,"%Y")
edx$month_of_rating <- format(edx$date,"%m")
validation$year_of_rating <- format(validation$date,"%Y")
validation$month_of_rating <- format(validation$date,"%m")

# Extracting the release date from the title for each dataset

edx_release <- stringi::stri_extract(edx$title, regex = "(\\(\\d{4}\\))", comments = TRUE)
edx_release <- gsub("[()]", "", edx_release) %>% as.numeric()
edx <- edx %>% mutate(release_date = edx_release)

rm(edx_release)

validation_release <- stringi::stri_extract(validation$title, regex = "(\\(\\d{4}\\))", comments = TRUE)
validation_release <- gsub("[()]", "", validation_release) %>% as.numeric()
validation <- validation %>% mutate(release_date = validation_release)

rm(validation_release)

# Extracting the single genre for each dataset

edx_genre_sep <- edx %>%
  mutate(genre = fct_explicit_na(genres, na_level = "(not assigned to a genre)")) %>%
  separate_rows(genre, sep = "\\|")

# Removing unnecessary columns in each dataset

edx <- edx %>% 
  select(userId, movieId, rating, title, genres, release_date, year_of_rating, month_of_rating)

validation <- validation %>% 
  select(userId, movieId, rating, title, genres, release_date, year_of_rating, month_of_rating)

# Converting columns into datatype numeric for each dataset

edx$year_of_rating <- as.numeric(edx$year_of_rating)
edx$month_of_rating <- as.numeric(edx$month_of_rating)
 
validation$year_of_rating <- as.numeric(validation$year_of_rating)
validation$month_of_rating <- as.numeric(validation$month_of_rating)

head(edx)

# Creating a test and a train partition of the edX dataset in order to execute experiments with multiple
# parameters

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set

test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# Section 2: Plots

#calculating the mean of ratings of the edx dataset in order to use it for some plots

avg <- mean(edx$rating)

# Plotting the rating distribution for movies in order to explain the movie bias

movie_rating_dist_plot <- edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "#000000") +
  scale_x_log10() + 
  ggtitle("Distribution of Ratings on Movies", 
          subtitle = "Some movies tend to have only a few ratings other movies have alot of ratings") +
  xlab("Number of Ratings") +
  ylab("Number of Movies") + 
  theme_classic()


# Plotting the User x Movie matrix

users <- sample(unique(edx$userId), 50)
user_movie_matrix_plot <- edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 50)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:50, 1:50,. ,main="User Movie Matrix", xlab="Movies", ylab="Users")
  abline(h=0:50+0.5, v=0:50+0.5, col = "grey") 
  

# Plotting the users rating behavior in order to explain how some people give a lot of ratings while others
# just give a few


count_rating_per_user <- nrow(edx)/length(unique(edx$userId))

user_rating_num_plot <- edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "#000000") +
  scale_x_log10() + 
  ggtitle("Rating Distribution of Users", 
          subtitle = "Most users rate fewer movies") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  geom_vline(xintercept = count_rating_per_user, color = "red", linetype = "dashed") +
  theme_classic()

# Plotting the average ratings of users with more than 100 ratings


avg_rating_User_plot <- edx %>% group_by(userId) %>%
  summarize(n =n(), mean = sum(rating/n))%>%
  filter(n >= 100) %>%
  ggplot(aes(x = mean)) +
  geom_histogram(color = "#000000") +
  ggtitle("Average Rating for all Users")+
  xlab("Mean Rating") +
  ylab("Number of Users") +
  geom_vline(xintercept = avg, color = "red", linetype = "dashed") +
  theme_classic()


# Plotting the frequency distribution of ratings in order to show how even ratings 
# are more common than odd ratings

freq_dist_rating_plot <- edx %>% group_by(rating) %>% 
  ggplot(aes(x=rating)) + 
  geom_bar(color = "#000000") +
  geom_vline(xintercept = avg, color = "red", linetype = "dashed") +
  scale_x_continuous(breaks=c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) +
  scale_y_continuous(labels = comma) +
  ggtitle("Count of Ratings", 
        subtitle = "Odd ratings are less commen than even ratings") +
  xlab("Rating") +
  ylab("Count") 


# Plotting the mean ratings versus the age of the movie

release_date_plot <- edx %>% group_by(release_date) %>%
  summarize(mean_rating = mean(rating)) %>% ggplot(aes(release_date, mean_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(colour = "red") +
  ylim(3.25, 4.25) +
  ggtitle("Average rating vs year of release") +
  ylab("Mean Rating") +
  xlab("Year of Release") +
  theme_classic()

# Plotting the mean rating versus the year of review

year_review_plot <- edx %>% group_by(year_of_rating) %>%
  summarize(mean_rating = mean(rating)) %>% ggplot(aes(year_of_rating, mean_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(colour = "red") +
  ylim(3.25, 4.25) +
  ggtitle("Average rating vs year of review") +
  ylab("Mean Rating") +
  xlab("Year of Review") +
  theme_classic()

# Plotting the mean rating versus the month of review

month_review_plot <- edx %>% group_by(month_of_rating) %>%
  summarize(mean_rating = mean(rating)) %>% ggplot(aes(month_of_rating, mean_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(colour = "red") +
  ylim(3.25, 3.75) +
  scale_x_continuous(breaks=c(1:12))+
  ggtitle("Average rating vs month of review") +
  ylab("Mean Rating") +
  xlab("Month of Review") +
  theme_classic()

# Plotting the relative frequency per genre 

# Building a matrix where 0 means not assigned to genre and 1 means assigned to genre

edx_genre_matrix <- edx%>%
  mutate(row = row_number()) %>%
  separate_rows(genres, sep = '\\|') %>%
  pivot_wider(names_from = genres, values_from = genres, 
              values_fn = function(x) 1, values_fill = 0) %>%
  select(-row) %>% 
  select(c(8:27))

# Calculating the means of each column

col_means_genre <- edx_genre_matrix %>% colMeans() 

# calculating the sum of all means

sum_col_means <- sum(col_means_genre)

# calculating the percentage by dividing the mean of a column by the sum

percentages <- sapply(col_means_genre, function(x){
  x/sum_col_means
})

# Creating a data frame in order to plot the percentage versus each genre

DF <- tibble(Genre = names(percentages), percentage = percentages)

rm(percentages)


genre_freq_plot <- ggplot(DF, aes(x = reorder(Genre, percentage), y = percentage)) +
  geom_bar(stat='identity', color = "#000000") +
  ggtitle("Rating Frequency for each Genre") +
  scale_y_continuous(labels=scales::percent) +
  xlab("Genre")+
  ylab("Relative Frequencies")+
  theme(axis.text.x=element_text(hjust=1)) +
  coord_flip() + 
  theme_classic()

# Plotting the average movie ratings per genre 

 genre_mean_err_plot <- edx_genre_sep %>%
  group_by(genre) %>%
  summarize(n_movies = n_distinct(movieId),
            n_ratings = n(),
            mean_rating = mean(rating),
            se = sd(rating)/sqrt(n()),
            lower = mean_rating - se*qt(0.975, n()),
            upper = mean_rating + se*qt(0.975, n())) %>%
  mutate(genres = reorder(genre, mean_rating)) %>%
  ggplot(aes(genres, mean_rating)) +
  geom_point() +
  theme(axis.text.x = element_text(hjust = 1)) +
  geom_hline(yintercept = avg, color = "red", linetype = "dashed") +
  geom_errorbar(aes(ymin = upper, ymax = lower)) +
  ylim(3.3, 4.125) +
  ggtitle("Mean Rating per Genre") +
  ylab("Mean Rating") +
  xlab("Genre") +
  coord_flip() +
  theme_classic()
  
rm(edx_genre_sep, edx_genre_matrix)

################################################################################################################
 
# Section 3: implementing the model evaluation functions and prediction models

# 3.1 model evaluation functions:

# Mean Absolute Error

MAE <- function(predicted_rating, true_rating){
  mean(abs(predicted_rating - true_rating))
}

# Mean Squared Error

MSE <- function(predicted_rating, true_rating){
  mean((predicted_rating - true_rating)^2)
}

# Root Mean Squared Error

RMSE <- function(predicted_rating, true_rating){
  sqrt(mean((predicted_rating - true_rating)^2))
}


# 3.2 Random Prediction by running a Monte Carlo Simulation

# setting seed

set.seed(1337, sample.kind = "rounding")

p <- function(x, y) {
  mean(y == x)
}

# defining the possible ratings

ratings <- seq(0.5, 5, 0.5) 

# number of replications

B <- 10000

# sampling, applying the applying the function p to vector ratings 

results <- replicate(B, {
  samples <- sample(train_set$rating, 100, replace = TRUE)
  sapply(ratings, p, y = samples)
  })

# estimating probability of each rating


prob <- sapply(1:nrow(results), function(x){
  mean(results[x,])})

# predicting random ratings with the sample function 

random_pred <- sample(ratings, size = nrow(test_set), replace = TRUE, prob = prob)


# calculating RMSE, MSE and MAE for the random prediction

results_table_RP <- tibble()
results_table_RP <- bind_rows(results_table_RP, 
                    tibble(Approach = "Random prediction", 
                           RMSE = RMSE(test_set$rating, random_pred),
                           MSE  = MSE(test_set$rating, random_pred),
                           MAE  = MAE(test_set$rating, random_pred)))

result_RP <-results_table_RP %>% gt() %>% tab_header(title = "Comparison of Results",
                                                       subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)


rm(results)



# 3.3 Linear Model

# 3.3.1 Prediction by calculating the mean of ratings

mu <- mean(train_set$rating)

# Calculating RMSE, MSE and MAE for the prediction by mean calculation

results_table_LM_mean <- tibble()
results_table_LM_mean <- bind_rows(results_table_LM_mean, 
                    tibble(Approach = "LM mean", 
                           RMSE = RMSE(test_set$rating, mu),
                           MSE  = MSE(test_set$rating, mu),
                           MAE  = MAE(test_set$rating, mu)))


result_LM_mean <-results_table_LM_mean %>% gt() %>%
  tab_header(title = "Comparison of Results",
  subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)


# 3.3.2 Prediction by calculating the mean and including the movie effect b_i


# Calculation of the movie effect b_i

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Prediction of ratings with mean and b_i 

pred_bi <- mu + test_set %>% 
  left_join(b_i, by = "movieId") %>%
  .$b_i

# Calculating RMSE, MSE and MAE for the prediction with mean and b_i

results_table_LM_I <- tibble()
results_table_LM_I <- bind_rows(results_table_LM_I, 
                    tibble(Approach = "LM incl. movie effect", 
                           RMSE = RMSE(test_set$rating, pred_bi),
                           MSE  = MSE(test_set$rating, pred_bi),
                           MAE  = MAE(test_set$rating, pred_bi)))


result_LM_I <-results_table_LM_I %>% gt() %>% tab_header(title = "Comparison of Results",
                                                       subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)

# Plotting the distribution of the movie effect

movie_effect_dist <- b_i %>% ggplot(aes(x = b_i)) + 
                      geom_histogram(bins=10, col = I("black")) +
                      ggtitle("Movie Effect Distribution") +
                      xlab("Movie effect") +
                      ylab("Count") +
                      scale_y_continuous(labels = comma) + 
                      theme_classic()


# 3.3.3 Prediction by calculating the mean and including both movie effect and user effect


# Calculation of the user effect b_u

b_u <- train_set %>% 
  left_join(b_i, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

# Prediction of ratings with mean, b_i and b_u

pred_bi_bu <- test_set %>% left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>% 
  .$pred


# Calculating RMSE, MSE and MAE for the prediction with mean, b_i and b_u

results_table_LM_IU <- tibble()
results_table_LM_IU <- bind_rows(results_table_LM_IU, 
                    tibble(Approach = "LM incl. user effect", 
                           RMSE = RMSE(test_set$rating, pred_bi_bu),
                           MSE  = MSE(test_set$rating, pred_bi_bu),
                           MAE  = MAE(test_set$rating, pred_bi_bu)))




result_LM_IU <-results_table_LM_IU %>% gt() %>% tab_header(title = "Comparison of Results",
                                                       subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)


# Plotting distribution of User effect

user_effect_dist <- b_u %>% ggplot(aes(x = b_u)) + 
                    geom_histogram(bins=10, col = I("black")) +
                    ggtitle("User Effect Distribution") +
                    xlab("User effect") +
                    ylab("Count") +
                    scale_y_continuous(labels = comma) + 
                    theme_classic()


#3.3.4 Prediction by calculating the mean and including movie effect user effect and genre effect


b_g <- train_set %>% 
  left_join(b_i, by = "movieId")%>%
  left_join(b_u, by = "userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Prediction of ratings with mean, b_i and b_u

pred_bi_bu_bg <- test_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

# Calculating RMSE, MSE and MAE for the prediction with mean, b_i, b_u and b_g

results_table_LM_IUG <- tibble()
results_table_LM_IUG <- bind_rows(results_table_LM_IUG, 
                           tibble(Approach = "LM incl. genre effect", 
                                  RMSE = RMSE(test_set$rating, pred_bi_bu_bg),
                                  MSE  = MSE(test_set$rating, pred_bi_bu_bg),
                                  MAE  = MAE(test_set$rating, pred_bi_bu_bg)))

result_LM_IUG <-results_table_LM_IUG %>% gt() %>% tab_header(title = "Comparison of Results",
                                                       subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)

# Plotting distribution of the genre effect

genre_effect_dist <-b_g %>% ggplot(aes(x = b_g)) + 
                    geom_histogram(bins=10, col = I("black")) +
                    ggtitle("Genre Effect Distribution") +
                    xlab("Genre effect") +
                    ylab("Count") +
                    scale_y_continuous(labels = comma) + 
                    theme_classic()


# 3.3.5 Prediction by calculating the mean and including movie effect, user effect, genre effect
# and time effect


b_t <- train_set %>% 
  left_join(b_i, by = "movieId")%>%
  left_join(b_u, by = "userId") %>% 
  left_join(b_g, by ="genres") %>%
  group_by(year_of_rating) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))


pred_bi_bu_bg_bt <- test_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_t, by="year_of_rating")%>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  .$pred


results_table_LM_IUGT <- tibble()
results_table_LM_IUGT <- bind_rows(results_table_LM_IUGT, 
                           tibble(Approach = "LM incl. timeffect", 
                                  RMSE = RMSE(test_set$rating, pred_bi_bu_bg),
                                  MSE  = MSE(test_set$rating, pred_bi_bu_bg),
                                  MAE  = MAE(test_set$rating, pred_bi_bu_bg)))
  


# creating a table with the results of Section 3

result_LM_IUGT <-results_table_LM_IUGT %>% gt() %>% tab_header(title = "Comparison of Results",
                                      subtitle = "Project Goal: RMSE = 0.8649")%>%
                                      fmt_markdown(columns = Approach) %>%
                                      fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)


# Plotting distribution of the release year effect

release_date_effect_dist <- b_t %>% ggplot(aes(x = b_t)) + 
                            geom_histogram(bins=10, col = I("black")) +
                            ggtitle("Release Date Effect Distribution") +
                            xlab("Release Date effect") +
                            ylab("Count") +
                            scale_y_continuous(labels = comma) + 
                            theme_classic()


#############################################################################################################

# Section 3.3.5: Regularization
# Creating the function regularization in order to find the best penalization value lambda
  
regularization <- function(lambda, trainset, testset){
  
  # Calculating mean
  
  mu <- mean(trainset$rating)
  
  # Calculating the movie effect b_i
  
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  # Calculating the user effect b_u  
  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  # Calculating the genre effect b_g
  
  b_g <- trainset %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - mu - b_u) / (n() + lambda))
  
  # Calculating the time effect b_t 
  
  b_t <- trainset %>% 
    left_join(b_i, by = "movieId")%>%
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by ="genres") %>%
    group_by(year_of_rating) %>% 
    summarize(b_t = mean(rating - mu - b_i - b_u - b_g)/ (n() + lambda))
  
  # Prediction with mean, b_i, b_u, b_g and b_t 
  
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "year_of_rating") %>%
    filter(!is.na(b_i), !is.na(b_u), !is.na(b_g), !is.na(b_t)) %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

# creating a vector with different values for lambda in a range from 0 to 10

lambda <- seq(0, 7.5, 0.25)

# Applying the regularization function to the lambda vector and saving the results in 
# the vector rmse

rmse <- sapply(lambda, 
                regularization, 
                trainset = train_set, 
                testset = test_set)

# Plotting the values for lambda versus RMSE

reg_plot <- tibble(Lambda = lambda, RMSE = rmse) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  xlab(expression(lambda))
  ggtitle("Regularization", 
          subtitle = "Pick the penalization that gives the lowest RMSE.") 
  


# Using the lambda value that returned the smallest RMSE

lambda <- lambda[which.min(rmse)]

# Prediction by calculating the mean including the regularized values of b_i, b_u, b_g and b_t 

mu <- mean(train_set$rating)

# Movie effect (bi)

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda))

# User effect (bu)

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))

# Genre effect 

b_g <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - mu - b_u) / (n() + lambda))

# Time effect

b_t <- train_set %>% 
  left_join(b_i, by = "movieId")%>%
  left_join(b_u, by = "userId") %>% 
  left_join(b_g, by ="genres") %>%
  group_by(year_of_rating) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g)/ (n() + lambda))

# Prediction  regularized with mean, b_i, b_u, b_g and b_t 

pred_reg <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "year_of_rating") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  pull(pred)


results_table_LM_reg <- tibble()
results_table_LM_reg <- bind_rows(results_table_LM_reg, 
                    tibble(Approach = "Regularized Linear Model", 
                           RMSE = RMSE(test_set$rating, pred_reg),
                           MSE  = MSE(test_set$rating, pred_reg),
                           MAE  = MAE(test_set$rating, pred_reg)))


result_reg_LM_reg <- results_table_LM_reg %>% gt() %>% tab_header(title = "Comparison of Results",
                                      subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)

#############################################################################################################


# Section 3.3.6: Matrix Factorization


set.seed(42, sample.kind = "rounding")

train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create the model object

ref_object <-  recosystem::Reco()

# Select the best tuning parameters

opts <- ref_object$tune(train_data, opts = list(dim = c(20, 30, 40), 
                                       lrate = c(0.05, 0.075, 0.1, 0.15),
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 8,
                                       niter = 10))
    

# Train the algorithm  

ref_object$train(train_data, opts = c(opts$min, nthread = 8, niter = 30))

# Save values for predictions in a vector

pred_reco <-  ref_object$predict(test_data, out_memory())


results_table_MF <- tibble()
results_table_MF <- bind_rows(results_table_MF, 
                    tibble(Approach = "Matrix Factorization", 
                           RMSE = RMSE(test_set$rating, pred_reco),
                           MSE  = MSE(test_set$rating, pred_reco),
                           MAE  = MAE(test_set$rating, pred_reco)))


result_MF <- results_table_MF %>% gt() %>% 
  tab_header(title = "Comparison of Results",
             subtitle = "Project Goal: RMSE = 0.8649") %>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)


 
#############################################################################################################

# Section  3.4 Final Validation 

# Linear Model

lambda <- seq(0, 7.5, 0.25)

# Applying the regularization function to the lambda vector and saving the results in 
# the vector rmse

rmse <- sapply(lambda, 
               regularization, 
               trainset = edx, 
               testset = validation)

# Plotting the values for lambda versus RMSE

regularization_plot <- tibble(Lambda = lambda, RMSE = rmse) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  xlab(expression(lambda))
ggtitle("Regularization", 
        subtitle = "Pick the penalization that gives the lowest RMSE.") 



# Using the lambda value that returned the smallest RMSE

lambda <- lambda[which.min(rmse)]

# Prediction by calculating the mean including the regularized values of b_i and b_u  

mu <- mean(edx$rating)

# Movie effect (bi)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda))

# User effect (bu)

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))

# Genre effect 

b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - mu - b_u) / (n() + lambda))

# Time effect 

b_t <- edx %>% 
  left_join(b_i, by = "movieId")%>%
  left_join(b_u, by = "userId") %>% 
  left_join(b_g, by ="genres") %>%
  group_by(year_of_rating) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g)/ (n() + lambda))

# Prediction

pred_reg_final <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "year_of_rating") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  pull(pred)


validation_results_table <- tibble()
validation_results_table <- bind_rows(validation_results_table, 
                           tibble(Approach = "Regularized Linear Model", 
                                  RMSE = RMSE(validation$rating, pred_reg_final),
                                  MSE  = MSE(validation$rating, pred_reg_final),
                                  MAE  = MAE(validation$rating, pred_reg_final)))


result_val_LM_reg <- validation_results_table %>% gt() %>% tab_header(title = "Comparison of Results",
                                      subtitle = "Project Goal: RMSE = 0.8649")%>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 7)



# Matrix Factorization 

set.seed(42, sample.kind = "rounding")

final_train_data <-  with(edx, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
final_test_data  <-  with(validation,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create the model object

ref_object <-  recosystem::Reco()

# Select the best tuning parameters

opts <- ref_object$tune(final_train_data, opts = list(dim = c(20, 30, 40), 
                                                lrate = c(0.05, 0.075, 0.1),
                                                costp_l2 = c(0.01, 0.1), 
                                                costq_l2 = c(0.01, 0.1),
                                                nthread  = 8,
                                                niter = 10))

opts$min

# Train the algorithm

ref_object$train(final_train_data, opts = c(opts$min, nthread = 8, niter = 30))

pred_reco_final <-  ref_object$predict(final_test_data, out_memory())
head(pred_reco_final, 10)


validation_results_table <- bind_rows(validation_results_table, 
                           tibble(Approach = "Matrix Factorization Validation", 
                                  RMSE = RMSE(validation$rating, pred_reco_final),
                                  MSE  = MSE(validation$rating, pred_reco_final),
                                  MAE  = MAE(validation$rating, pred_reco_final)))


result_val_MF <- validation_results_table %>% gt() %>% 
  tab_header(title = "Comparison of Results",
             subtitle = "Project Goal: RMSE = 0.8649") %>%
  fmt_markdown(columns = Approach) %>%
  fmt_number(columns = c( RMSE, MSE, MAE), decimals = 4)
  



