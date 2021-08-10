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


# Figure: percentage of variance explained by different principal components.
res.pca <- prcomp(input_features, scale = TRUE, retx = TRUE)
fviz_eig(res.pca)

# Computation of percentage of variance
standard_deviations = res.pca$sdev
variances = standard_deviations*standard_deviations
var_pc1 = (variances[1]/sum(variances))*100
var_pc2 = (variances[2]/sum(variances))*100
var_pc3 = (variances[3]/sum(variances))*100
var_pc4 = (variances[4]/sum(variances))*100
var_pc5 = (variances[5]/sum(variances))*100
var_pc6 = (variances[6]/sum(variances))*100
var_pc7 = (variances[7]/sum(variances))*100
var_pc8 = (variances[8]/sum(variances))*100
var_pc9 = (variances[9]/sum(variances))*100
var_pc10 = (variances[10]/sum(variances))*100

# More than 80% of explained variance
var_pc1 + var_pc2 + var_pc3 + var_pc4 + var_pc5 + var_pc6 + var_pc7 + var_pc8

# We now perform some evaluation of experiments with the transformed data of PCA. We evaluate the performance of algorithms on PCA transformed data.
transformed_data <- res.pca$x


# Upto 8 PCs, because it explains more than 80%
pca_till8_labels <- cbind(transformed_data[,1], transformed_data[,2], transformed_data[,3], transformed_data[,4], transformed_data[,5], transformed_data[,6], transformed_data[,7], transformed_data[,8], data_without_id$stroke)
pca_till8_labels <- as.data.frame(pca_till8_labels)
names(pca_till8_labels) <- c("trans1", "trans2", "trans3", "trans4", "trans5", "trans6", "trans7", "trans8", "stroke")
head(pca_till8_labels)


# ============================================================================================================================================
# Random downsampling on PCA transformed data. Evaluation of different approaches
no_of_exps <- 100
minority_class <- pca_till8_labels[pca_till8_labels$stroke == 1,]
majority_class <- pca_till8_labels[pca_till8_labels$stroke == 0,]



# Neural network method on PCA transformed data
file_name <- './results/top8pca_NN.csv'
cat(c("Experiment-number,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  nnetm <- train(as.factor(stroke) ~., data=train_set, method='nnet', trace = FALSE)
  pred_nn <- predict(nnetm, newdata=test_set)
  
  # Output labels
  out_labels<-as.data.frame(test_set$stroke)
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
  
}






# --------------------------------------------------------------------------------------------------------------------------------------------------


# Decision tree method on PCA transformed data
file_name <- './results/top8pca_decisiontree.csv'
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
  # Predicting the Test set results
  y_pred_dt= predict(classifier_dt, newdata = test_set, type = 'class')
  
  # Output labels
  out_labels<-as.data.frame(test_set$stroke)
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
  
}




# ---------------------------------------------------------------------------------------------------------------------


# Random forest method on PCA transformed data
file_name <- './results/top8pca_randomforest.csv'
cat(c("Experiment-number,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  train_set$stroke <- as.character(train_set$stroke)
  train_set$stroke <- as.factor(train_set$stroke)
  stroke_rf = randomForest(stroke~., data=train_set, ntree=100, proximity=T)
  strokePred = predict(stroke_rf, newdata=test_set)
  
  confusion_matrix = table(strokePred, test_set$stroke)
  
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
  
}

# ============================================================================================================================================

