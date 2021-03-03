# Text classifier project - Fake news or True?
This project was done on the course "Statistics" from University of Trento. The idea is to have an open project where we would apply the tecniques learned on the subject in order to propose a solution for a problem.

This is a summary of the report. The html of the full report can be found on this [link](https://sangoncalves.github.io/News-Classifier-FAKE-or-NOT/).

## Objective

Identify if is viable to use algorithms to classify pieces of news in order to enhance its confiability.

## Data

The data is composed by two labeled datasets, one containing "True" news and the other containing "Fake" news.

## Models analysed

* Logistic Regression
* Random Forest
* Naive Bayes

## Evaluation

| Model | Accuracy | 
|---|---|
|Logistic Regression|0.986 |
|Random Forest | 0.998|
| Naive Bayes |0.884 |

## Conclusion
In our case, the most important metric parameter that we can analyze is accuracy. The reason is that, in our case, a wrong output does not represents loss or damage to any entity. For example, in a case where a model classifies as edible or poisonous fungus, the precision would be the most important metric, since we would prefer to avoid risk. Or by the other hand, in a case of classification of the extracted mineral as rare or not, recall would be the most important, since we would prefer to minimize the loss. 

Analyzing the confusion matrix we can perceive that the model with highest accuracy is Random Forest (*~99.8%*) and the model with the lowest accuracy is Naive Bayes (*~88%*). 
The Random Forest (RF) classifiers are suitable for dealing with the high dimensional noisy data in text classification. The RF model comprises a set of decision trees each of which is trained using random subsets of features. Given the high dimension of the input and the higher number of decision used for training, we have a good accuracy rate for this model. Some of the reason for the NB to have the lowest accuracy can be due to its assumption of independence among the words. This means that it takes not in consideration the order of the words, but treats the input as a "bag of words" and calculate the probabilities (hence the name naive).
