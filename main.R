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

# Loading factors
res.pca$rotation

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

# Only 2 PCs
pca_data_labels <- cbind(transformed_data[,1], transformed_data[,2], data_without_id$stroke) # We consider PC1, PC2 and stroke labels.
pca_data_labels <- as.data.frame(pca_data_labels)
names(pca_data_labels) <- c("trans1", "trans2", "stroke")
head(pca_data_labels)




# ============================================================================================================================================
# Random downsampling on PCA transformed data. Evaluation of different approaches
no_of_exps <- 100
minority_class <- pca_data_labels[pca_data_labels$stroke == 1,]
majority_class <- pca_data_labels[pca_data_labels$stroke == 0,]



# Neural network method on PCA transformed data
file_name <- './results/top2pca_NN.csv'
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
file_name <- './results/top2pca_decisiontree.csv'
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
file_name <- './results/top2pca_randomforest.csv'
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

# Figure: Biplot  representation  of  the  actual  patient  attributes projected on the first two principal components
fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



# Section II (a): the pca values
std_dev <- res.pca[1]
sdev <- std_dev$sdev
eig_values <- sdev^2
pca_values <- eig_values/sum(eig_values)
pca_values <- pca_values*100
pca_values


# Figure: Sub-space representation of the different observations in the electronic health records transformed on the reference of the first two principal components. 
fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



# Figure: Each observation is color coded
groups <- as.factor(data_without_id$stroke)
fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = groups, # color by groups
             palette = c("#00AFBB", "#FC4E07"),
             
             ellipse.type = "confidence",
             legend.title = "Groups",
             repel = TRUE,
             addEllipses = TRUE # Concentration ellipses
)




# ============================================================================================================================================
# Random downsampling on actual features. Evaluation of different approaches
no_of_exps <-5
minority_class <- data_without_id[data_without_id$stroke == 1,]
majority_class <- data_without_id[data_without_id$stroke == 0,]



# Neural network method on original ALL features

file_name <- './results/allfeatures_neuralnetwork.csv'
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
  
  nnetm <- train(as.factor(stroke) ~., data=train_set, method='nnet')
  #print(nnetm)
  #plot(nnetm)
  pred_nn <- predict(nnetm, newdata=test_set)
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 11])
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





# Decision tree method on original ALL features

dtree_result = c()

file_name <- './results/allfeatures_decisiontree.csv'
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
  out_labels<-as.data.frame(test_set[, 11])
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




# Random forest method on original ALL features

rforest_result = c()
file_name <- './results/allfeatures_randomforest.csv'
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
ggdensity(df, x = "Accuracy",
          add = "mean", rug = TRUE,
          color = "Method", fill = "Method",
          palette = c("#0073C2FF", "#FC4E07", "#07fc9e"))


# saving the df
write.csv(df, file = "./results/my_results.csv")


# Table 1
cat("Mean accuracy of neural network: ", mean(nnetwork_result))
cat("Mean accuracy of decision tree: ", mean(dtree_result))
cat("Mean accuracy of random forest: ", mean(rforest_result))




# ----------------

# Adding features sequentially. 
datamatrix_case1 = data_without_id[,c("age","heart_disease","avg_glucose_level", "stroke")]
datamatrix_case2 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "stroke")]
datamatrix_case3 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "stroke")]
datamatrix_case4 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "stroke")]
datamatrix_case5 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status", "stroke")]
datamatrix_case6 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender", "stroke")]
datamatrix_case7 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi", "stroke")]
datamatrix_case8 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]


# Neural network method. We use this for extended analysis
file_name <- './results/performance_adding.csv'
cat(c("Case,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)
no_of_exps <- 100

for (case_number in 1:8){
  
  if (case_number == 1){
    data_matrix <- datamatrix_case1
  }else if (case_number == 2) {
    data_matrix <- datamatrix_case2
  }else if (case_number == 3) {
    data_matrix <- datamatrix_case3
  }else if (case_number == 4) {
    data_matrix <- datamatrix_case4
  }else if (case_number == 5) {
    data_matrix <- datamatrix_case5
  }else if (case_number == 6) {
    data_matrix <- datamatrix_case6
  }else if (case_number == 7) {
    data_matrix <- datamatrix_case7
  }else{
    data_matrix <- datamatrix_case8
  } 
  
  
  minority_class <- data_matrix[data_without_id$stroke == 1,]
  majority_class <- data_matrix[data_without_id$stroke == 0,]
  
  for (i in 1:no_of_exps) {
    cat("Current experiment: ", i)
    
    
    majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
    balanced_dataset <- rbind(minority_class, majority_sample)
    
    split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
    train_set = subset(balanced_dataset, split == TRUE)
    test_set = subset(balanced_dataset, split == FALSE)
    
    nnetm <- train(as.factor(stroke) ~., data=train_set, method='nnet', trace = FALSE)
    #print(nnetm)
    pred_nn <- predict(nnetm, newdata=test_set)
    
    # Output labels
    out_labels<-as.data.frame(test_set[, c("stroke")])
    out_labels<-t(out_labels)
    
    cm_nn = table(pred_nn, out_labels) # The order is important
    TP <- cm_nn[1,1]
    FN <- cm_nn[2,1]
    FP <- cm_nn[1,2]
    TN <- cm_nn[2,2]
    
    #computation of the values
    precision <- TP/(TP+FP)
    recall <- TP/(TP+FN)
    fscore <- 2*TP/(2*TP + FP + FN)
    
    accuracy <- (TP+TN)/(TP+TN+FP+FN)
    miss_rate <- FN/(TN+TP)
    fall_out_rate <- FP/(FP+TN)
    
    my_vect <- c(case_number, TP, FN, FP, TN, precision, recall, fscore, accuracy, miss_rate, fall_out_rate)
    vectStr=paste(as.character(my_vect), sep="' '", collapse=",")
    cat(vectStr,file=file_name,append=TRUE)
    cat("\n",file=file_name,append=TRUE)
    
  }

}



# ----------------------
# Removing only one feature to understand its impact 
datamatrix_case1 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case2 = data_without_id[,c("heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case3 = data_without_id[,c("age","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case4 = data_without_id[,c("age","heart_disease", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case5 = data_without_id[,c("age","heart_disease","avg_glucose_level",  "ever_married", "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case6 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" ,  "work_type", "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case7 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married",  "smoking_status","gender","bmi","Residence_type", "stroke")]
datamatrix_case8 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "gender","bmi","Residence_type", "stroke")]
datamatrix_case9 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","bmi","Residence_type", "stroke")]
datamatrix_case10 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","Residence_type", "stroke")]
datamatrix_case11 = data_without_id[,c("age","heart_disease","avg_glucose_level", "hypertension" , "ever_married", "work_type", "smoking_status","gender","bmi", "stroke")]


# Neural network method. We use this for extended analysis
file_name <- './results/performance_removing.csv'
cat(c("Case,", "TP,", "FN,", "FP,", "TN,", "Precision,", "Recall,", "F-score,","Accuracy,", "Miss-rate,", "Fall-out-rate"),file=file_name)
cat("\n",file=file_name,append=TRUE)
no_of_exps <- 100


for (case_number in 1:11){
  
  if (case_number == 1){
    data_matrix <- datamatrix_case1
  }else if (case_number == 2) {
    data_matrix <- datamatrix_case2
  }else if (case_number == 3) {
    data_matrix <- datamatrix_case3
  }else if (case_number == 4) {
    data_matrix <- datamatrix_case4
  }else if (case_number == 5) {
    data_matrix <- datamatrix_case5
  }else if (case_number == 6) {
    data_matrix <- datamatrix_case6
  }else if (case_number == 7) {
    data_matrix <- datamatrix_case7
  }else if (case_number == 8) {
    data_matrix <- datamatrix_case8
  }else if (case_number == 9) {
    data_matrix <- datamatrix_case9
  }else if (case_number == 10) {
    data_matrix <- datamatrix_case10
  }else{
    data_matrix <- datamatrix_case11
  } 
  
  
  minority_class <- data_matrix[data_without_id$stroke == 1,]
  majority_class <- data_matrix[data_without_id$stroke == 0,]
  
  for (i in 1:no_of_exps) {
    cat("Current experiment: ", i)
    
    
    majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
    balanced_dataset <- rbind(minority_class, majority_sample)
    
    split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
    train_set = subset(balanced_dataset, split == TRUE)
    test_set = subset(balanced_dataset, split == FALSE)
    
    nnetm <- train(as.factor(stroke) ~., data=train_set, method='nnet', trace = FALSE)
    #print(nnetm)
    pred_nn <- predict(nnetm, newdata=test_set)
    
    # Output labels
    out_labels<-as.data.frame(test_set[, c("stroke")])
    out_labels<-t(out_labels)
    
    cm_nn = table(pred_nn, out_labels) # The order is important
    TP <- cm_nn[1,1]
    FN <- cm_nn[2,1]
    FP <- cm_nn[1,2]
    TN <- cm_nn[2,2]
    
    #computation of the values
    precision <- TP/(TP+FP)
    recall <- TP/(TP+FN)
    fscore <- 2*TP/(2*TP + FP + FN)
    
    accuracy <- (TP+TN)/(TP+TN+FP+FN)
    miss_rate <- FN/(TN+TP)
    fall_out_rate <- FP/(FP+TN)
    
    my_vect <- c(case_number, TP, FN, FP, TN, precision, recall, fscore, accuracy, miss_rate, fall_out_rate)
    vectStr=paste(as.character(my_vect), sep="' '", collapse=",")
    cat(vectStr,file=file_name,append=TRUE)
    cat("\n",file=file_name,append=TRUE)
    
  }
  
}


