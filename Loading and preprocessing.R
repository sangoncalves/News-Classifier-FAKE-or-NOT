###################################### LOADING LIBRARIES
library(tidyr)
library(readr)
library(dplyr)
library(tm)
library(naivebayes)
library(caTools)
library(ROCR)
library(RTextTools)
library(MLmetrics)

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
  return(doc_matrix);
}

#3 -CONVERTING THE FACTOR FROM STRING TO INTEGER - NAIVE BAYES MODEL REQUIREMENT
convert_counts <- function(x){
  x <- ifelse(x > 0, "Yes", "No")
}

###################################### PREPROCESSING - ALL MODELS
# fake_news_data = read.csv2('train-3.csv', sep = ',', stringsAsFactors = F)
fake_news_data = read_csv('train-3.csv')
processed.data <- dataPreprocessing(fake_news_data)
processed.label <- processed.data %>% select('label')

#splitting the data and getting the variables ready for the models
smp_size <- floor(0.75 * nrow(processed.data))
train_index <- sample(seq_len(nrow(processed.data)), size = smp_size)
##train variables
train.input <- processed.data[train_index, ] %>% select('news')
train.label <- processed.data[train_index, ] %>% select('label')
train.data  <- processed.data[train_index, ]
##test variables
test.input <- processed.data[-train_index, ] %>% select('news')
test.label <- processed.data[-train_index, ] %>% select('label')
test.data  <- processed.data[-train_index, ]


###################################### NAIVE BAYES MODEL
## PROCESSING
# DTM 
dtm <- createCorpusAndDTM(processed.data)
# split the document-term matrix
dtm.train <- dtm[1:smp_size, ]
dtm.test <- dtm[smp_size:nrow(processed.data), ]
freq_terms = findFreqTerms(dtm.train, 5)
dtm_freq.train <- dtm.train[, freq_terms]
dtm_freq.test <- dtm.test[, freq_terms]
reduced_dtm.train <- apply(dtm_freq.train, MARGIN=2, convert_counts)
reduced_dtm.test  <- apply(dtm_freq.test, MARGIN=2, convert_counts)

## MODEL
# Training
nb_classifier <- naiveBayes(reduced_dtm.train, train.labels)
# Predicting
nb_predict <- predict(nb_classifier, reduced_dtm.test)

# Accuracy
nb_accuracy <- Accuracy(nb_predict, test.labels)
nb_accuracy

###################################### LOGISTIC REGRESSION MODEL
## PROCESSING
dtm_sparse <- removeSparseTerms(dtm, 0.98)
dtm_sparse <- as.data.frame(as.matrix(dtm_sparse))
colnames(dtm_sparse) = make.names(colnames(dtm_sparse))
dtm_sparse$label <- as.factor(processed.label)
dtm_sparse$label = as.factor(processed.data$label)
set.seed(123)
spl = sample.split(dtm_sparse$label, 0.7)
dtm_sparse.train = subset(dtm_sparse, spl == TRUE)
dtm_sparse.test = subset(dtm_sparse, spl == FALSE)

## MODEL
lr <- glm(label ~ ., data = dtm_sparse.train, family = "binomial")
lr_pred.train <- predict(lr, type = "response")

# Accuracy on training 
lr_prediction_trainLog = prediction(lr_pred.train, dtm_sparse.train$label)
lr_train_accuracy <- as.numeric(performance(lr_prediction_trainLog, "auc")@y.values)
lr_train_accuracy

# Accuracy on test 
lr_pred_test = predict(lr, newdata = dtm_sparse.test, type="response")
lr_prediction_testLog = prediction(lr_pred_test, dtm_sparse.test$label)
lr_accuracy <- as.numeric(performance(lr_prediction_testLog, "auc")@y.values)
lr_accuracy

###################################### SUPPORT VECTOR MACHINE MODEL
## PROCESSING
dtm_train.input <- createCorpusAndDTM(train.input)
train_size = nrow(train.input)
train_input_container <- create_container(dtm_train.input,t(train.input),trainSize = 1:train_size,virgin=FALSE)

## MODEL
# Training the svm model will take more time, you can load the our pretrained model from the 'models' folder 
model_svm <- train_model(train_input_container, "SVM", kernel="linear", cost=1)

set.seed(333)
test_index <- sample(seq_len(nrow(test.input)), size =5 )
svm_prediction_data <- as.list(test.input[test_index, ])
svm_pred_matrix <- create_matrix(svm_prediction_data, originalMatrix=dtm.data.news) 

# create the corresponding container
svm_pred_size = length(svm_prediction_data);
snm_prediction_container <- create_container(svm_pred_matrix, labels=rep(0, svm_pred_size), testSize=1:svm_pred_size, virgin=FALSE) 

results <- classify_model(snm_prediction_container, model_svm)
svm_test_label[test_index, ]

# Accuracy on test
Accuracy(results$SVM_LABEL, test.label)




