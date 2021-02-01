setwd("C:\\Users\\sande\\Documents\\MyProjects\\News-Classifier-FAKE-or-NOT")
library(tm)
library(caTools)
library(e1071)
library(MLmetrics)

fake <- read.csv("NewDataset/Fake.csv")
fake$label <- 1
true <- read.csv("NewDataset/True.csv")
true$label <- 0
data <- rbind.data.frame(fake,true)
data$subject <- as.factor(data$subject)
data$label <- as.factor(data$label)

str(fake)
str(true)
str(data)

(length(true$title)+ length(fake$title)) == length(data$title)

set.seed(50)

rows <- sample(nrow(data))
shuffled_data <- data[rows, ]
summary(data)

corpus <- Corpus(VectorSource(data$text))

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stemDocument)

freq <- DocumentTermMatrix(corpus)
freq

freq <- removeSparseTerms(freq, 0.90)
freq

newcorpus <- as.data.frame(as.matrix(freq))
colnames(newcorpus) <- make.names(colnames(newcorpus))
newcorpus$subject <- data$subject


set.seed(1)
split <- sample.split(newcorpus$subject, SplitRatio = 0.7)
train <- subset(newcorpus, split==TRUE)
test <- subset(newcorpus, split==FALSE)
naivesubject <- naiveBayes(label ~., data=train)
predictnaivesubject <- predict(naivesubject, newdata = test, type="class")
table(predictnaivesubject, test$subject)


nb_accuracy <- Accuracy(y_pred = predictnaivesubject, y_true = test$subject)
