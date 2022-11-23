install.packages("corrplot")
install.packages("pROC")
install.packages("smotefamily")

library(tidyverse)
library(caret)
library(randomForest)
library(neuralnet)
library(smotefamily)
library(scales)
library(corrplot)
library(pROC) 
library(gt)

# Reading the csv file and saving it as object creditcard with class data frame

setwd("C:/Users/User/Documents/RProjects/Capstone Project/Credit_Card_Fraud")
creditcard <- read.csv("creditcard.csv")
class(creditcard)

# Exploring the structure and the first 6 entries of the data set

str(creditcard)
head(creditcard)

# Checking for missing values, should return 0 for each column if there is no missing values

colSums(is.na(creditcard))

#############################################################################################################
# Plots

# Plotting the frequencies of fraudulent and not fraudulent transactions in the dataset
# in order to show the imbalance

freq_plot <- creditcard %>% 
  group_by(Class) %>% 
  summarise(n = n()) %>%
  ggplot(aes(x = factor(Class), n, fill = Class)) +
  geom_bar(stat='identity',color = c("#47ABD8", "#D01B1B"), fill = c("#95D2EC", "#FF817E")) +
  scale_x_discrete(labels = c("FALSE","TRUE")) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) + 
  ggtitle("Distribution of Fraudalent Activities", 
          subtitle = "The amount of true fraudalent transactions is very small in comparison to the number valid transactions") +
  xlab("fraudalent") +
  ylab("count") +
  theme_classic()
  

# Plotting amount versus class

amount_vs_class_plot <- creditcard %>% ggplot(aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
                        labs(x = 'Class', y = 'Amount') +
                        scale_x_discrete(labels = c("FALSE","TRUE")) +
                        ggtitle("Distribution of transaction amount by class")+
                        theme_classic()


# Plotting time versus class

time_plot <- creditcard %>%
              ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
              labs(x = 'Time in seconds since first Transaction', y = 'Number of Transactions') +
              ggtitle('Time versus Number of Transactions grouped by Class') +
              facet_grid(Class ~ ., scales = 'free_y') + theme_classic()


# Plotting correlations between the unknown features and class

correlations <- cor(creditcard[,-1],method="pearson")
corr_plot <- corrplot(correlations, number.cex = .9, method = "circle",
             addCoef.col = 'black',
             type = "full",
             tl.cex=0.8,tl.col = "black")


##############################################################################################################
# Model: Neural Network using Backpropagation


# setting seed in order to get reproduceable results

set.seed(42, sample.kind = "rounding")

# Creating two partitions of the original dataset (ratio 80/20) for a neuronal network model

index_NN <- createDataPartition(y = creditcard$Class, p = 0.6, list = F)
NN_train <- creditcard[index_NN,]
NN_test_and_val <- creditcard[-index_NN,]

index_val <- createDataPartition(y = NN_test_and_val$Class, p = 0.5, list = F)
NN_test <- NN_test_and_val[index_val,]
NN_val <- NN_test_and_val[-index_val,]

# Normalizing all independent variables using the scaling function to ensure,
# that the minimums and maximums do not affect the performance of the neural network model.
# scaling also the test data in order to compute predictions

NN_val_scaled <- NN_val %>% mutate_at(c(1:30), funs(c(scale(.))))
NN_test_scaled <- NN_test %>% mutate_at(c(1:30), funs(c(scale(.))))
NN_train_scaled <- NN_train %>% mutate_at(c(1:30), funs(c(scale(.))))

# Setting number of fraud and legitimate cases and the desired percentage of legitimate cases

n0 <- nrow(subset(NN_train, Class==0))
n1 <- nrow(subset(NN_train, Class==1))
r0 <-0.65 

# Calculating the number of duplicates

ntimes <- ((1 - r0) / r0) * (n0/n1) - 1

# Creating smote output with K = 5 nearest neighbors and duplication size equal to the calculation of ntimes

SMOTE_output_NN = SMOTE( X= NN_train_scaled[,-c(1,30)],target = NN_train_scaled$Class, K = 5, dup_size = ntimes)

# Creating the SMOTE data

NN_train_SMOTE <- SMOTE_output_NN$data
colnames(NN_train_SMOTE)[30] <- "Class"
prop.table(table(NN_train_SMOTE$Class))

# creating a list with 3 vectors in order to test the neuronal network with one, two or three neurons

list_layers <- list(c(1),c(2),c(3))

for (i in 1:length(list_layers)){
NN_model <- neuralnet(Class ~ ., data = NN_train_SMOTE, hidden = list_layers[[i]], linear.output = F)

# linear.output is set to false because this is a binary classification problem 
# hidden argument gets a vector, the length represents the number
# of hidden layers and the value represents the number of neurons in this layer

# computing the predicted values

predicted_values_NN <- neuralnet::compute(NN_model, NN_val_scaled)

# transform vectors in factors in order to produce a confusion matrix 
# with library(caret)
# applying rounding on all predicted values in order to compare the predicted and actual values

predictions_NN <- factor(sapply(predicted_values_NN$net, round))
actual_val_NN <- factor(NN_val_scaled$Class)

# printing the number of iteration steps and corresponding confusion matrix in order to check the performance

print(confusionMatrix(data = predictions_NN, reference = actual_val_NN))
print(i)

# plotting the ROC- curve and AUC - Value in order to check the sensitivity/specificity trade off

par(pty = "s")
roc(actual_val_NN, as.numeric(predictions_NN), plot = TRUE, legacy.axes = TRUE, percent = TRUE, col = "blue",lwd = 4, print.auc=TRUE)

# plotting the neural network

plot(NN_model, rep ="best")

# plotting the neuronal network with error and number of steps

print(NN_model$result.matrix)
}

############################################################################################################

# Applying the trained Neural Network to the unseen test set

# computing the predicted values

predicted_values_NN <- neuralnet::compute(NN_model, NN_test_scaled)

predictions_NN <- factor(sapply(predicted_values_NN$net, round))

actual_test_NN <- factor(NN_test_scaled$Class)

CM_NN <- confusionMatrix(data = predictions_NN, reference = actual_test_NN)



par(pty = "s")
roc_NN <- roc(actual_test_NN, as.numeric(predictions_NN), plot = TRUE, legacy.axes = TRUE, percent = TRUE, col = "blue",lwd = 4, print.auc=TRUE)

#############################################################################################################

# Model: Random Forest Classifier

# Creating two partitions of the original dataset (ratio 80/20) for the random forest model

index_RF <- createDataPartition(y = creditcard$Class, p = 0.8, list = F)
RF_train <- creditcard[index_RF,]
RF_test <- creditcard[-index_RF,]
RF_test$Time <- NULL

# Setting number of fraud and legitimate cases and the desired percentage of legitimate cases

n0 <- nrow(subset(RF_train, Class==0))
n1 <- nrow(subset(RF_train, Class==1))
r0 <-0.65 

# Calculating the number of duplicates

ntimes <- ((1 - r0) / r0) * (n0/n1) - 1

# Creating smote output with K = 5 nearest neighbors and duplication size equal to the calculation of n times

SMOTE_output_RF = SMOTE( X= RF_train[,-c(1,31)],target = RF_train$Class, K = 5, dup_size = ntimes)

# removing the Class column in SMOTE_output$data

SMOTE_output_RF$data$Class <- NULL

# saving the smote_output$data column as an object

RF_train_SMOTE <- SMOTE_output_RF$data

# rename the column with synthetic and original data as "Class"

colnames(RF_train_SMOTE)[30] <- "Class"

# transform the Class column as numeric

RF_train_SMOTE$Class <- as.factor(RF_train_SMOTE$Class)

# treating the test data set like the train set for compareability
# changing the column class to character and turning them into factor 
# results in factor levels 1 and 2 instead of 0 and 1

RF_test$Class <- as.character(RF_test$Class)
RF_test$Class <- as.factor(RF_test$Class)



prop.table(table(RF_train_SMOTE$Class))

# Setting train control parameters for 10 fold cross validation with 3 repetitions

ctrl_RF <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     classProbs = T,
                     verboseIter = T,
                     savePredictions = T)

# applying the random forest model to the training set

levels(RF_train_SMOTE$Class) <- make.names(c(0, 1))

classifier_RF <- train(Class ~ .,
                       method = "rf",
                       data = RF_train_SMOTE,
                       trControl = ctrl_RF,
                       metric = "ROC")


# making the predictions with our classifier and the test set

predicted_values_RF <- predict(classifier_RF, newdata = RF_test)


levels(predicted_values_RF) <- c(0, 1)
CM_RF <- confusionMatrix(data = predicted_values_RF, reference = RF_test$Class)

# Plotting model
  
plot(classifier_RF)

# Importance plot



# Variable importance plot

important_var <- varImp(classifier_RF)
plot(important_var)

roc_RF <- roc(RF_test$Class, as.numeric(predicted_values_RF), plot = TRUE, legacy.axes = TRUE, percent = TRUE, col = "blue",lwd = 4, print.auc=TRUE)
 
############################################################################################################

# Model: Logistic Regression

# Setting seed in order to get reproduceable results

set.seed(42, sample.kind = "rounding")

# Create two a training and a test data set
index_glm <- createDataPartition(y = creditcard$Class, p = 0.8, list = F)
glm_train <- creditcard[index_glm,]
glm_test <- creditcard[-index_glm,]
glm_test$Time <- NULL

# Setting number of fraud and legitimate cases and the desired percentage of legitimate cases

n0 <- nrow(subset(glm_train, Class==0))
n1 <- nrow(subset(glm_train, Class==1))
r0 <-0.65 

# Calculating the number of duplicates

ntimes <- ((1 - r0) / r0) * (n0/n1) - 1

# Creating smote output with K = 5 nearest neighbors and duplication size equal to the calculation of n times

SMOTE_output_glm = SMOTE( X= glm_train[,-c(1,31)],target = glm_train$Class, K = 5, dup_size = ntimes)

# Creating the SMOTE data

SMOTE_output_glm$data$Class <- NULL
glm_train_SMOTE <- SMOTE_output_glm$data
colnames(glm_train_SMOTE)[30] <- "Class"
glm_train_SMOTE$Class <- as.factor(glm_train_SMOTE$Class)
glm_test$Class <- as.character(glm_test$Class)
glm_test$Class <- as.factor(glm_test$Class)



prop.table(table(glm_train_SMOTE$Class))


# defining the train control parameter for cross validation with k = 10 and 3 repetitions

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     verboseIter = T,
                     classProbs = T,
                     savePredictions = T)

levels(glm_train_SMOTE$Class) <- make.names(c(0, 1))
glm_fit <- train(Class ~ ., family = "binomial",method ="glm", data = glm_train_SMOTE, trControl = ctrl, metric = "ROC")
summary(glm_fit)


predicted_values_glm <- predict(glm_fit, newdata = glm_test, type = 'raw')

actual_glm <- factor(glm_test$Class)
levels(predicted_values_glm) <- c(0, 1)
CM_glm <- confusionMatrix(data = predicted_values_glm, reference = actual_glm)

par(pty = "s")
roc_glm <- roc(glm_test$Class, as.numeric(predicted_values_glm), plot = TRUE, legacy.axes = TRUE, percent = TRUE, col = "blue",lwd = 4, print.auc=TRUE)


############################################################################################################

# Draw designed Confusion matrix 

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Legitimate', cex=1.2)
  rect(250, 430, 340, 370, col="#FF817E")
  text(295, 435, 'Fraud', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col="#FF817E")
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Legitimate', cex=1.2, srt=90)
  text(140, 335, 'Fraud', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

# draw example matrix

layout(matrix(c(1,1,1))) 
par(mar=c(2,2,2,2)) 
plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n') 
title('CONFUSION MATRIX', cex.main=2) 

# create the matrix 
rect(150, 430, 240, 370, col='#3F97D0') 
text(195, 435, 'Legitimate', cex=1.2) 
rect(250, 430, 340, 370, col="#FF817E") 
text(295, 435, 'Fraud', cex=1.2) +
text(125, 370, 'Predicted', cex=1.3, srt=90, font=2) 
text(245, 450, 'Actual', cex=1.3, font=2) 
rect(150, 305, 240, 365, col="#FF817E") 
rect(250, 305, 340, 365, col='#3F97D0') 
text(140, 400, 'Legitimate', cex=1.2, srt=90) 
text(140, 335, 'Fraud', cex=1.2, srt=90) 

# add in the cm results 
text(195, 400, "TP", cex=1.6, font=2, col='white') 
text(195, 335, "FP", cex=1.6, font=2, col='white') 
text(295, 400, "FN", cex=1.6, font=2, col='white') 
text(295, 335, "TN", cex=1.6, font=2, col='white')


# draw example figure for the SMOTE algorithm


# create data frame

sample_data <- data.frame( Feature1 = c(2,4,5,5,6,7,8),
                           
                           Feature2 = c(4,2,3,4,4,5,6),
                           
                           category1 = c('Neighbor','Neighbor','Synthetic','Synthetic','Original',
                                         'Synthetic','Neighbor'),
                           
                           category2 = c(0,1,1,0,0,1,1))

# create plot

smote_example <- ggplot(sample_data, aes(x=Feature1,y=Feature2)) +
  geom_point(aes(color=category1),size=5) +
  geom_line(aes(group = category2),linetype = "dashed") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank() 
  ) +
  ggtitle('Visualization of generating Syntethic Data') 

