###################################### LOADING LIBRARIES
library(keras)
library(ggplot2)
library(purrr)
library(e1071)
library(gmodels)
library(MLmetrics)
library(kernlab)
library(ROCR)
library(tm)
library(dplyr)
library(tidyselect)
library(tidyr)
library(caTools)
library(RTextTools)
library(readr)
library(naivebayes)
library(bnlearn)


###################################### DEFINING FUNCTIONS

#1 - PREPROCESSING FUNCTION - USED BY ALL MODELS
dataPreprocessing <- function(df) {
  df_train <- data.frame('id' = df$tid2 , 
                         'news' = df$title2_en, 
                         'label' = df$label);
  df_train_unique <-  unique(df_train);
  df_train_unique$label <- as.factor(df_train_unique$label);
  train_label <- pivot_wider(df_train_unique, 
                             id_cols = c('id','news'),
                             names_from = 'label', 
                             values_from ='label');
  train_label <- as.data.frame(train_label);
  train_label$label <- NA;
  
  #It provide the final label. If agreed->fake news, else not fake
  
  for (i in 1:length(train_label$id)){
    if(!is.na(train_label$agreed[i])) {
      train_label$label[i] <-'fake';
    }
    else {
      train_label$label[i] <-'not fake';
    }
  }
  train_label$label[train_label$label=='not fake'] <- as.integer(0);
  train_label$label[train_label$label=='fake'] <- as.integer(1);
  
  train_label_final <- train_label[c('id', 'news','label')];
  train_label_final$news <- as.character(train_label_final$news);
  train_label_final$label  <- factor(train_label_final$label)
  
  return(train_label_final);
}

#2-  Creating Corpus and Generating DTM - NAIVE BAYES AND LOGISTIC REGRESSION REQUIREMENT
createCorpusAndDTM <- function(dataset) {
  nb_corpus <- VCorpus(VectorSource(dataset$news));
  nb_corpus_clean <- tm_map(nb_corpus, content_transformer(tolower));
  nb_corpus_clean <- tm_map(nb_corpus_clean, content_transformer(removeNumbers));
  nb_corpus_clean <- tm_map(nb_corpus_clean, removePunctuation);
  nb_corpus_clean <- tm_map(nb_corpus_clean, removeWords,stopwords());
  nb_corpus_clean <- tm_map(nb_corpus_clean, stemDocument);
  nb_corpus_clean <- tm_map(nb_corpus_clean, stripWhitespace);
  doc_matrix = DocumentTermMatrix(nb_corpus_clean);
  dtm = removeSparseTerms(doc_matrix, 0.98);
  dtm_sparse <- as.data.frame(as.matrix(dtm))
  colnames(dtm_sparse) = make.names(colnames(dtm_sparse))
  #dtm_sparse$label = as.factor(dataset$label)
  
  return(dtm_sparse);
}

#3 -CONVERTING THE FACTOR FROM STRING TO INTEGER - NAIVE BAYES MODEL REQUIREMENT
convert_counts <- function(x){
  x <- ifelse(x > 0, "Yes", "No")
}

###################################### PREPROCESSING - ALL MODELS
fake_news_data = read_csv('train.csv', sep = ',', stringsAsFactors = F)
preprocessed.data <- dataPreprocessing(fake_news_data)
preprocessed.label <- preprocessed.data %>% select('label')

#splitting the data and getting the variables ready for the models
smp_size <- floor(0.75 * nrow(preprocessed.data))
train_index <- sample(seq_len(nrow(preprocessed.data)), size = smp_size)
##train variables
train.input <- preprocessed.data[train_index, ] %>% select('news')
train.label <- preprocessed.data[train_index, ] %>% select('label')
train.data  <- preprocessed.data[train_index, ]
##test variables
test.input <- preprocessed.data[-train_index, ] %>% select('news')
test.label <- preprocessed.data[-train_index, ] %>% select('label')
test.data  <- preprocessed.data[-train_index, ]

# DTM 
dtm.train <- as.factor(preprocessed.label) %>%  createCorpusAndDTM(train.data)
dtm.test  <- as.factor(preprocessed.label) %>%  createCorpusAndDTM(test.data)
dtm.data  <- as.factor(preprocessed.label) %>%  createCorpusAndDTM(preprocessed.data)
dtm.data.news <- createCorpusAndDTM(preprocessed.label)


# FT, INTEGER LABEL AND REDUCED DTM - RELATED TO NAIVE BAYES MODEL

#frequent terms and reduce dtm of train and test data. 
dtm_freq_terms = findFreqTerms(dtm.data, 5);
dtm_freq.train <- dtm.train[, dtm_freq_terms]
dtm_freq.test <- dtm.test[, dtm_freq_terms]

# NAIVE BAYES - uses 0|1 input instead of string (fake | not fake)
reduced_dtm.train <- apply(dtm_freq.train, MARGIN=2, convert_counts);
reduced_dtm.test  <- apply(dtm_freq.test, MARGIN=2, convert_counts);


###################################### LOGISTIC REGRESSION MODEL
lr <- glm(label ~ ., data = lr_train, family = "binomial")
lr_pred.train <- predict(lr, type = "response")

# Accuracy on training 
lr_prediction_trainLog = prediction(lr_pred.train, lr_train$label)
lr_train_accuracy <- as.numeric(performance(lr_prediction_trainLog, "auc")@y.values)
lr_train_accuracy

lr_pred_test = predict(lr, newdata = dtm.train, type="response")
lr_prediction_testLog = prediction(lr_pred_test, dtm.train$label)
lr_accuracy <- as.numeric(performance(lr_prediction_testLog, "auc")@y.values)
lr_accuracy