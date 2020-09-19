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

train = read_csv('train.csv', sep = ',', stringsAsFactors = F)


#DEFINING FUNTIONS
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
  train_label$final_label <- NA;
  
  #It provide the final label. If agreed->fake news, else not fake
  
  for (i in 1:length(train_label$id)){
    if(!is.na(train_label$agreed[i])) {
      train_label$final_label[i] <-'fake';
    }
    else {
      train_label$final_label[i] <-'not fake';
    }
  }
  train_label$final_label[train_label$final_label=='not fake'] <- as.integer(0);
  train_label$final_label[train_label$final_label=='fake'] <- as.integer(1);
  
  train_label_final <- train_label[c('id', 'news','final_label')];
  train_label_final$news<- as.character(train_label_final$news);
  
  return(train_label_final);
}


# Creating Corpus and Generating DTM
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
  ## 75% of the sample size
splitDataset <- function(dataset, id) {
  smp_size <- floor(0.75 * nrow(dataset))
  
  ## set the seed to make your partition reproducible
  set.seed(123)
  train_index <- sample(seq_len(nrow(dataset)), size = smp_size)
  if (id == 1) {
    return(dataset[train_index, ])
  } else {
    return(dataset[-train_index, ])
  }
}



###################################### Support Vector Machines (SVM)
preprocessed_train_data <- dataPreprocessing(fake_news_data)

svm_train <- splitDataset(preprocessed_train_data, 1)
svm_test <- splitDataset(preprocessed_train_data, 2)

svm_train_input <-  svm_train %>% select('news')
svm_train_label <- svm_train %>% select('final_label')

svm_test_input <- svm_test %>% select('news')
svm_test_label <- svm_test %>% select('final_label')

matrix <- create_matrix(svm_train_input, language="english",removeNumbers=FALSE,removeStopwords = TRUE, stemWords=TRUE, removePunctuation=TRUE,toLower=TRUE, weighting=weightTfIdf)

train_size = nrow(svm_train_input)
train_container <- create_container(matrix,t(svm_train_label),trainSize = 1:train_size,virgin=FALSE)

# Training the svm model will take more time, you can load the our pretrained model from the 'models' folder 
model_svm <- train_model(train_container, "SVM", kernel="linear", cost=1)

set.seed(333)
test_index <- sample(seq_len(nrow(svm_test_input)), size =5 )

svm_prediction_data <- as.list(svm_test_input[test_index, ])

svm_pred_matrix <- create_matrix(svm_prediction_data, originalMatrix=matrix) 

# create the corresponding container
svm_pred_size = length(svm_prediction_data);
snm_prediction_container <- create_container(svm_pred_matrix, labels=rep(0, svm_pred_size), testSize=1:svm_pred_size, virgin=FALSE) 


results <- classify_model(snm_prediction_container, model_svm)

svm_test_label[test_index, ]
Accuracy(results$SVM_LABEL, svm_test$final_label)