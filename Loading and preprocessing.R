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

