library(keras)
library(dplyr)
library(ggplot2)
library(readr)
library(purrr)
library(tidyr)

train = read_csv('train.csv', sep = ',', stringsAsFactors = F)

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

## 75% of the sample size
smp_size <- floor(0.75 * nrow(preprocessed_train_data))

## set the seed to make your partition reproducible
set.seed(123)
train_index <- sample(seq_len(nrow(preprocessed_train_data)), size = smp_size)

train <- preprocessed_train_data[train_index, ]
verification <- preprocessed_train_data[-train_index, ]
