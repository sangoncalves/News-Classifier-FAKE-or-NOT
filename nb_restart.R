library(fastNaiveBayes)
library(tidyr)
library(dplyr)
library(e1071)
library(tm)

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
fake_news_data = read.csv('Reduced Data/train-3.csv')
processed.data <- dataPreprocessing(fake_news_data)
processed.label <- processed.data %>% select('label')

#splitting the data and getting the variables ready for the models
smp_size <- floor(0.75 * nrow(processed.data))
train_index <- sample(seq_len(nrow(processed.data)), size = smp_size)
##train variables
train.input <- processed.data[train_index, ] %>% select('news')
train.input <- train.input$news
train.label <- processed.data[train_index, ] %>% select('label')
train.label <- train.label$label
train.data  <- processed.data[train_index, ]
##test variables
test.input <- processed.data[-train_index, ] %>% select('news')
test.label <- processed.data[-train_index, ] %>% select('label')
test.data  <- processed.data[-train_index, ]

###################################### NAIVE BAYES MODEL
#Theory
{

# Why is Naive Bayes "naive"? Naive Bayes is "naive" because of its strong independence assumptions. It assumes that all features are equally important and that all features are independent. If you think of n-grams and compare unigrams and bigrams, you can intuitively understand why the last assumption is a strong assumption. A unigram counts each word as a gram ("I" "like" "walking" "in" "the" "sun") whereas a bigram counts two words as a gram ("I like" "like walking" "walking in" "in the" "the sun").
# 
# However, even when the assumptions are not fully met, Naive Bayes still performs well.
# A Naive Bayes classifier that assumes independence between the feature variables. Currently, either a Bernoulli, multinomial, or Gaussian distribution can be used. 
# The bernoulli distribution should be used when the features are 0 or 1 to indicate the presence or absence of the feature in each document.
# The multinomial distribution should be used when the features are the frequency that the feature occurs in each document. 
# Finally, the Gaussian distribution should be used with numerical variables. The distribution parameter is used to mix different distributions for different columns in the input matrix

}

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

x <- dtm_freq.train
y <- train.label 


#https://cran.r-project.org/web/packages/fastNaiveBayes/vignettes/fastnaivebayes.html

mod <- fastNaiveBayes::fastNaiveBayes(x = train.input,y = train.label, distribution = fnb.detect_distribution(x))
# Mixed event models
dist <- fastNaiveBayes::fnb.detect_distribution(x, nrows = nrow(x))
print(dist)
nb_multinomial <- fastNaiveBayes::fnb.multinomial(train.input,train.label)
mod <- fastNaiveBayes.mixed(x,y,laplace = 1)
pred <- predict(mod, newdata = x)
mean(pred!=y)

# Bernoulli only
vars <- c(dist$bernoulli, dist$multinomial)
newx <- x[,vars]
for(i in 1:ncol(newx)){
  newx[[i]] <- as.factor(newx[[i]])
}