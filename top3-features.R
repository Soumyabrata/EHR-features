# Load the different libraries
library(dplyr)
library(caret)
library(ggplot2)
library(imputeTS)
library(caTools)
library(e1071)
library(nnet)
library(rpart)
library(ggpubr)
library("randomForest")
library(factoextra)


# ============================================================================================================================================
# Load the dataset. The blank spaces are replaced by NA.
input_data <- read.csv("./dataset/train_2v.csv", na.strings=c("","NA"))

# Omit the rows that contain NA.
input_data<- na.omit(input_data)
colnames(input_data)

# The patient ID is not a feature, so we remove it. We also remove the stroke feature for some input feature analysis.
input_features = input_data[,2:11]
data_without_id = input_data[,2:12]
head(input_features) #just check the structure of the dataframe.

# Converting into factor type. Typecasting into factors.
input_features$gender <- as.factor(input_features$gender)
input_features$ever_married <- as.factor(input_features$ever_married)
input_features$work_type <- as.factor(input_features$work_type)
input_features$Residence_type <- as.factor(input_features$Residence_type)
input_features$smoking_status <- as.factor(input_features$smoking_status)

input_features[] <- data.matrix(input_features)



# ============================================================================================================================================
# Random downsampling on actual features. Evaluation of different approaches
no_of_exps <-100

datamatrix_case1 = data_without_id[,c("age","heart_disease","avg_glucose_level", "stroke")]
data_matrix <- datamatrix_case1
minority_class <- data_matrix[data_without_id$stroke == 1,]
majority_class <- data_matrix[data_without_id$stroke == 0,]



# Neural network method on top 3 features

file_name <- './results/top3originalfeatures_neuralnetwork.csv'
cat(c("Experiment-number,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)

nnetwork_result = c()
for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  nnetm <- train(as.factor(stroke) ~., data=train_set, method='nnet', trace = FALSE)
  #print(nnetm)
  #plot(nnetm)
  pred_nn <- predict(nnetm, newdata=test_set)
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 4])
  out_labels<-t(out_labels)
  
  confusion_matrix = table(pred_nn, out_labels) # The order is important
  
  TP <- confusion_matrix[1,1]
  FN <- confusion_matrix[2,1]
  FP <- confusion_matrix[1,2]
  TN <- confusion_matrix[2,2]
  #computation of the values
  precision <- TP/(TP+FP)
  recall <- TP/(TP+FN)
  fscore <- 2*TP/(2*TP + FP + FN)
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  miss_rate <- FN/(TN+TP)
  fall_out_rate <- FP/(FP+TN)
  my_vect <- c(i, TP, FN, FP, TN, precision, recall, fscore, accuracy, miss_rate, fall_out_rate)
  vectStr=paste(as.character(my_vect), sep="' '", collapse=",")
  cat(vectStr,file=file_name,append=TRUE)
  cat("\n",file=file_name,append=TRUE)
  
  nnetwork_result[length(nnetwork_result)+1] = accuracy
}

nnetwork_result




# Decision tree method on top 3 features

dtree_result = c()

file_name <- './results/top3originalfeatures_decisiontree.csv'
cat(c("Experiment-number,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  
  train_set$stroke <- factor(train_set$stroke)
  classifier_dt = rpart(formula =stroke~ .,
                        data = train_set)
  #print(classifier_dt)
  # Predicting the Test set results
  y_pred_dt= predict(classifier_dt, newdata = test_set, type = 'class')
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 4])
  out_labels<-t(out_labels)
  
  confusion_matrix = table(y_pred_dt, out_labels) # The order is important
  
  TP <- confusion_matrix[1,1]
  FN <- confusion_matrix[2,1]
  FP <- confusion_matrix[1,2]
  TN <- confusion_matrix[2,2]
  #computation of the values
  precision <- TP/(TP+FP)
  recall <- TP/(TP+FN)
  fscore <- 2*TP/(2*TP + FP + FN)
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  miss_rate <- FN/(TN+TP)
  fall_out_rate <- FP/(FP+TN)
  my_vect <- c(i, TP, FN, FP, TN, precision, recall, fscore, accuracy, miss_rate, fall_out_rate)
  vectStr=paste(as.character(my_vect), sep="' '", collapse=",")
  cat(vectStr,file=file_name,append=TRUE)
  cat("\n",file=file_name,append=TRUE)
  
  
  
  
  dtree_result[length(dtree_result)+1] = accuracy
}

dtree_result



# Random forest method on top 3 features

rforest_result = c()
file_name <- './results/top3originalfeatures_randomforest.csv'
cat(c("Experiment-number,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  trainData = subset(balanced_dataset, split == TRUE)
  testData = subset(balanced_dataset, split == FALSE)
  
  
  
  trainData$stroke <- as.character(trainData$stroke)
  trainData$stroke <- as.factor(trainData$stroke)
  stroke_rf = randomForest(stroke~., data=trainData, ntree=100, proximity=T)
  strokePred = predict(stroke_rf, newdata=testData)
  
  confusion_matrix = table(strokePred, testData$stroke)
  
  TP <- confusion_matrix[1,1]
  FN <- confusion_matrix[2,1]
  FP <- confusion_matrix[1,2]
  TN <- confusion_matrix[2,2]
  #computation of the values
  precision <- TP/(TP+FP)
  recall <- TP/(TP+FN)
  fscore <- 2*TP/(2*TP + FP + FN)
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  miss_rate <- FN/(TN+TP)
  fall_out_rate <- FP/(FP+TN)
  my_vect <- c(i, TP, FN, FP, TN, precision, recall, fscore, accuracy, miss_rate, fall_out_rate)
  vectStr=paste(as.character(my_vect), sep="' '", collapse=",")
  cat(vectStr,file=file_name,append=TRUE)
  cat("\n",file=file_name,append=TRUE)
  
  
  rforest_result[length(rforest_result)+1] = accuracy
}

rforest_result



nnetwork_result
dtree_result
rforest_result


x_name <- "Method"
y_name <- "Accuracy"

a1 <- replicate(no_of_exps, "Neural Network")
a2 <- replicate(no_of_exps, "Decision Tree")
a3 <- replicate(no_of_exps, "Random Forest")
a <- c(a1,a2,a3)

all_accuracies <- c(nnetwork_result,dtree_result,rforest_result)
df <- data.frame(a, all_accuracies)
colnames(df) <- c(x_name, y_name)
print(df)
head(df)


# Figure 4
pdf('./results/distribution-3features.pdf',width=6,height=4,paper='special')
ggdensity(df, x = "Accuracy",
          add = "mean", rug = TRUE,
          color = "Method", fill = "Method",
          palette = c("#0073C2FF", "#FC4E07", "#07fc9e"))
dev.off()


# saving the df
write.csv(df, file = "./results/top3features_myresults.csv")

