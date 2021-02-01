setwd("C:\\Users\\sande\\Documents\\MyProjects\\News-Classifier-FAKE-or-NOT")

##Load Libraries

library(ggplot2)
library(tidyverse)
library(dplyr)
library(readr)
library(cowplot)
library(olsrr)
library(readr) 
library(caret)
library(pscl)
library(lmtest)
library(ipred)
library(survival)
library(ResourceSelection)
library(survey)
library(pROC)
library(DescTools)

## NOTE: This is a proof of concept. Further validation work needs to take place.
##need to create train/datasets

data = read.csv('NewDataset/Fake.csv', stringsAsFactors = F)
head(data) #allows you to check the data, first few entries 
summary(data) #produce result summaries of the results of various model fitting functions.
dim(data) #the dimension (e.g. the number of columns and rows) of a matrix, array or data frame. 
str(data) 

data$text <- as.character(data$text)
data$subject <- as.integer(data$subject=="News")
str(data)

library(tm)
corpus <- Corpus(VectorSource(data$text))

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stemDocument)

library(wordcloud)
library(RColorBrewer)
wordcloud(corpus,max.words=150,random.order=FALSE, rot.per=0.15, colors=brewer.pal(8,"Dark2"))

freq <- DocumentTermMatrix(corpus)
freq

freq <- removeSparseTerms(freq, 0.90)
freq

newcorpus <- as.data.frame(as.matrix(freq))
colnames(newcorpus) <- make.names(colnames(newcorpus))
newcorpus$subject <- data$subject

library(caTools)
library(e1071)
set.seed(1)
split <- sample.split(newcorpus$subject, SplitRatio = 0.7)
train <- subset(newcorpus, split==TRUE)
test <- subset(newcorpus, split==FALSE)
naivesubject <- naiveBayes(as.factor(subject)~., data=train)
predictnaivesubject <- predict(naivesubject, newdata = test, type="class")
table(predictnaivesubject, test$subject)

library(MLmetrics)
nb_accuracy <- Accuracy(y_pred = predictnaivesubject, y_true = test$subject)
