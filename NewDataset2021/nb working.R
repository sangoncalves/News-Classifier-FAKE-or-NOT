library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyr)
library(tm)
library(textstem)
library(tidytext)
library(pROC)
library(ROCR)
library(randomForest)
library(naivebayes)
library(caret)
library(caTools)

setwd("C:\\Users\\sande\\Documents\\MyProjects\\News-Classifier-FAKE-or-NOT")

fake <- read.csv('NewDataset2021/Fake.csv', stringsAsFactors = F)
fake$label <- 1

true <- read.csv('NewDataset2021/True.csv', stringsAsFactors = F)
true$label <- 0

data <- rbind.data.frame(fake, true)
set.seed(50)

rows <- sample(nrow(data))
shuffled_data <- data[rows, ]

datapreprocessing <- function(data) {
  data$label <- as.factor(data$label)
  data$subject <- as.factor(data$subject)
  data <- data %>% select(title, text, label) %>% unite(col = text, title, text, sep = ' ') %>% mutate(ID = as.character(1:nrow(data)))
  
  return(data)
}

clean_data <- datapreprocessing(shuffled_data)

generateDTM <- function(df) {
  corpus <- Corpus(VectorSource(df$text))
  
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords('en'))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(str_remove_all), "[[:punct:]]")
  corpus <- tm_map(corpus, content_transformer(lemmatize_strings))
  
  freq <- DocumentTermMatrix(corpus)
  inspect(freq)
  
  freq_clean <- removeSparseTerms(freq, 0.97)
  inspect(freq_clean)
  
  freq_matrix <- as.matrix(freq_clean)
  dim(freq_matrix)
  
  freq_matrix <- cbind(freq_matrix, label = shuffled_data$label)
  
  summary(freq_matrix[, 'label'])
  
  freq_df <- as.data.frame(freq_matrix)
  freq_df$label <- ifelse(freq_df$label == 1, 0, 1)
  freq_df$label <- as.factor(freq_df$label)
  
  return(freq_df);
}

DTM_df <- generateDTM(clean_data)

set.seed(50)
spl = sample.split(DTM_df$label, 0.7)
train_dtm = subset(DTM_df, spl == TRUE)
test_dtm = subset(DTM_df, spl == FALSE)

calculateAccuracy <- function(test_labels, predict_labels) {
  cf <- caret::confusionMatrix(test_labels, predict_labels)
  acc <- cf[['overall']]['Accuracy']
  return(acc);
}

NB <-  function(train, test){
  mdl_nb <- naive_bayes(label ~ ., data = train)
  # Model Summary
  summary(mdl_nb)
  # Predicted values
  train$pred_nb <- predict(mdl_nb, type = 'class')
  # Predicted Values for test set
  test$pred_nb <- predict(mdl_nb, newdata = test)
  return(test$pred_nb)
}
  
predict_NB <- NB(traindtm,test_dtm)

accuracy_nb <- calculateAccuracy(test_dtm$label, test_dtm$pred_nb)
paste('Accuracy of RF: ', accuracy_nb)