####
# Nearly every email user has at some point encountered a "spam" email, which is an 
# unsolicited message often advertising a product, containing links to malware, or 
# attempting to scam the recipient. Roughly 80-90% of more than 100 billion emails 
# sent each day are spam emails, most being sent from botnets of malware-infected 
# computers. The remainder of emails are called "ham" emails.
# 
# As a result of the huge number of spam emails being sent across the Internet each 
# day, most email providers offer a spam filter that automatically flags likely spam 
# messages and separates them from the ham. Though these filters use a number of
# techniques (e.g. looking up the sender in a so-called "Blackhole List" that contains
# IP addresses of likely spammers), most rely heavily on the analysis of the contents 
# of an email via text analytics.
# 
# In this homework problem, we will build and evaluate a spam filter using a publicly 
# available dataset first described in the 2006 conference paper "Spam Filtering with 
# Naive Bayes -- Which Naive Bayes?" by V. Metsis, I. Androutsopoulos, and G. Paliouras.
# The "ham" messages in this dataset come from the inbox of former Enron Managing 
# Director for Research Vincent Kaminski, one of the inboxes in the Enron Corpus. 
# One source of spam messages in this dataset is the SpamAssassin corpus, which 
# contains hand-labeled spam messages contributed by Internet users. The remaining 
# spam was collected by Project Honey Pot, a project that collects spam messages 
# and identifies spammers by publishing email address that humans would know not
# to contact but that bots might target with spam. The full dataset we will use 
# was constructed as roughly a 75/25 mix of the ham and spam messages.
# 
# The dataset contains just two fields:
#   text: The text of the email.
#   spam: A binary variable indicating if the email was spam.



#### setting work directory
rm(list=ls())
getwd()
setwd('C:/Users/Desk 1/Desktop/the_analytics_edge/week5/assignment')
dir()



####
# Begin by loading the dataset emails.csv into a data frame called emails. Remember 
# to pass the stringsAsFactors=FALSE option when loading the data.
# 
# How many emails are in the dataset?
emails <- read.csv('emails.csv', stringsAsFactors=FALSE)
emails[1, ]
dim(emails)



####
# How many of the emails are spam?
table(emails$spam)



####
# Which word appears at the beginning of every email in the dataset? Respond as a 
# lower-case word with punctuation removed.
emails[2, ]



####
# Could a spam classifier potentially benefit from including the frequency of the
# word that appears in every email?
# 
# A. No -- the word appears in every email so this variable would not help us 
# differentiate spam from ham. 
# 
# B. Yes -- the number of times the word appears might help us differentiate spam 
# from ham.

# Answer: B



####
# The nchar() function counts the number of characters in a piece of text. How many 
# characters are in the longest email in the dataset (where longest is measured in 
# terms of the maximum number of characters)?
names(emails)
max(nchar(emails$text))



####
# Which row contains the shortest email in the dataset?
min(nchar(emails$text))
emails[nchar(emails$text)==13, ]



####
# Follow the standard steps to build and pre-process the corpus:
library(tm)
library(SnowballC)

# 1) Build a new corpus variable called corpus.
corpus <- Corpus(VectorSource(emails$text))

# 2) Using tm_map, convert the text to lowercase.
corpus <- tm_map(corpus, tolower)

# 3) Using tm_map, remove all punctuation from the corpus.
corpus <- tm_map(corpus, removePunctuation)

# 4) Using tm_map, remove all English stopwords from the corpus.
stopwords('english')[1:5]
corpus <- tm_map(corpus, removeWords, stopwords('english'))

# 5) Using tm_map, stem the words in the corpus.
corpus <- tm_map(corpus, stemDocument)

# 6) Build a document term matrix from the corpus, called dtm.
dtm <- DocumentTermMatrix(corpus)

# If the code length(stopwords("english")) does not return 174 for you, then please 
# run the line of code in this file, which will store the standard stop words in a 
# variable called sw. When removing stop words, use tm_map(corpus, removeWords, sw) 
# instead of tm_map(corpus, removeWords, stopwords("english")).
length(stopwords('english'))

# How many terms are in dtm?
dtm



####
# To obtain a more reasonable number of terms, limit dtm to contain terms appearing 
# in at least 5% of documents, and store this result as spdtm (don't overwrite dtm, 
# because we will use it in a later step of this homework). How many terms are in spdtm?
spdtm <- removeSparseTerms(dtm, 0.95)
spdtm



####
# Build a data frame called emailsSparse from spdtm, and use the make.names function 
# to make the variable names of emailsSparse valid.
# 
# colSums() is an R function that returns the sum of values for each variable in our 
# data frame. Our data frame contains the number of times each word stem (columns) 
# appeared in each email (rows). Therefore, colSums(emailsSparse) returns the number 
# of times a word stem appeared across all the emails in the dataset. What is the
# word stem that shows up most frequently across all the emails in the dataset? 
# Hint: think about how you can use sort() or which.max() to pick out the maximum 
# frequency.

emailsSparse <- as.data.frame(as.matrix(spdtm))
emailsSparse[1:10, 1:10]
sort(colSums(emailsSparse))



####
# Add a variable called "spam" to emailsSparse containing the email spam labels.
# 
# How many word stems appear at least 5000 times in the ham emails in the dataset? 
# Hint: in this and the next question, remember not to count the dependent variable
# we just added.

emailsSparse$spam <- emails$spam
hamSparse <- subset(emailsSparse, spam==0)
sort(colSums(hamSparse))



####
# How many word stems appear at least 1000 times in the spam emails in the dataset?
spamSparse <- subset(emailsSparse, spam==1)
sort(colSums(spamSparse))

# Answer: 3 (not 4!)
#
# Explanation: "subject", "will", and "compani" are the three stems that appear at
# least 1000 times. Note that the variable "spam" is the dependent variable and is
# not the frequency of a word stem.



####
# The lists of most common words are significantly different between the spam and 
# ham emails. What does this likely imply?
# 
# A. The frequencies of these most common words are unlikely to help differentiate 
# between spam and ham. 
# 
# B. The frequencies of these most common words are likely to help differentiate 
# between spam and ham.

# Answer: B



####
# Several of the most common word stems from the ham documents, such as "enron", 
# "hou" (short for Houston), "vinc" (the word stem of "Vince") and "kaminski", 
# are likely specific to Vincent Kaminski's inbox. What does this mean about the 
# applicability of the text analytics models we will train for the spam filtering
# problem?
# 
# A. The models we build are still very general, and are likely to perform well as a 
# spam filter for nearly any other person. 
# 
# B. The models we build are personalized, and would need to be further tested before 
# use as spam filters for other.

# Answer: B



####
# First, convert the dependent variable to a factor with 
# "emailsSparse$spam = as.factor(emailsSparse$spam)".
emailsSparse$spam <- as.factor(emailsSparse$spam)

# Next, set the random seed to 123 and use the sample.split function to split 
# emailsSparse 70/30 into a training set called "train" and a testing set called 
# "test". Make sure to perform this step on emailsSparse instead of emails.
library(caTools)
set.seed(123)
split <- sample.split(emailsSparse$spam, SplitRatio=0.7)
train <- emailsSparse[split==TRUE, ]
test <- emailsSparse[split==FALSE, ]

# Using the training set, train the following three machine learning models. The
# models should predict the dependent variable "spam", using all other available 
# variables as independent variables. Please be patient, as these models may take
# a few minutes to train.

# 1) A logistic regression model called spamLog. You may see a warning message 
# here - we'll discuss this more later.
spamLog <- glm(spam ~ ., data = train, family = binomial)

# 2) A CART model called spamCART, using the default parameters to train the model 
# (don't worry about adding minbucket or cp). Remember to add the argument 
# method="class" since this is a binary classification problem.
library(rpart)
library(rpart.plot)
spamCART <- rpart(spam ~ ., data = train, method = 'class')
prp(spamCART, main = 'Spam Prediction Decision Tree')
dev.copy(png, './figures/spam_pred_decision_tree.png')
dev.off()

# 3) A random forest model called spamRF, using the default parameters to train
# the model (don't worry about specifying ntree or nodesize). Directly before 
# training the random forest model, set the random seed to 123 (even though 
# we've already done this earlier in the problem, it's important to set the seed 
# right before training the model so we all obtain the same results. Keep in mind 
# though that on certain operating systems, your results might still be slightly
# different).
library(randomForest)
set.seed(123)

spamRF <- randomForest(spam ~ ., data = train)
'000' %in% names(train)
names(train)

colnames(train) <- make.names(colnames(train))
colnames(test) <- make.names(colnames(test))

spamRF <- randomForest(spam ~ ., data = train)

# For each model, obtain the predicted spam probabilities for the training set. 
# Be careful to obtain probabilities instead of predicted classes, because we 
# will be using these values to compute training set AUC values. Recall that 
# you can obtain probabilities for CART models by not passing any type parameter 
# to the predict() function, and you can obtain probabilities from a random forest 
# by adding the argument type="prob". For CART and random forest, you need to 
# select the second column of the output of the predict() function, corresponding 
# to the probability of a message being spam.

predPercLog.train <- predict(spamLog, type='response')
predPercCART.train <- predict(spamCART)[ , 2]
predPercRF.train <- predict(spamRF, type='prob')[ , 2]

head(predPercLog)
head(predPercCART)
head(predPercRF)

# You may have noticed that training the logistic regression model yielded the
# messages "algorithm did not converge" and "fitted probabilities numerically 0 or 1
# occurred". Both of these messages often indicate overfitting and the first 
# indicates particularly severe overfitting, often to the point that the training 
# set observations are fit perfectly by the model. Let's investigate the predicted 
# probabilities from the logistic regression model.

# How many of the training set predicted probabilities from spamLog are less 
# than 0.00001?
table(predPercLog.train < 0.00001)

# How many of the training set predicted probabilities from spamLog are more 
# than 0.99999?
table(predPercLog.train > 0.99999)  

# How many of the training set predicted probabilities from spamLog are
# between 0.00001 and 0.99999?
table(predPercLog.train > 0.00001 & predPercLog.train < 0.99999)



####
# How many variables are labeled as significant (at the p=0.05 level) in the 
# logistic regression summary output?
summary(spamLog)



####
# How many of the word stems "enron", "hou", "vinc", and "kaminski" appear in the 
# CART tree? Recall that we suspect these word stems are specific to Vincent Kaminski
# and might affect the generalizability of a spam filter built with his ham data.
prp(spamCART)



####
# What is the training set accuracy of spamLog, using a threshold of 0.5 for predictions?
predLog.train <- ifelse(predPercLog.train > 0.5, 1, 0)
table(predLog.train, train$spam)
(3052 + 954) / (3052 + 4 + 0 + 954)



####
# What is the training set AUC of spamLog?
library(ROCR)
predictionLog.train <- prediction(predPercLog.train, train$spam)
performanceLog.train <- performance(predictionLog.train, 'tpr', 'fpr')
plot(performanceLog.train, colorize=TRUE)

AUC.tmp <- performance(predictionLog.train, 'auc')
AUC.Log.train <- as.numeric(AUC.tmp@y.values)
AUC.Log.train



####
# What is the training set accuracy of spamCART, using a threshold of 0.5 for 
# predictions? (Remember that if you used the type="class" argument when making 
# predictions, you automatically used a threshold of 0.5. If you did not add in 
# the type argument to the predict function, the probabilities are in the second 
# column of the predict output.)
predCART.train <- ifelse(predPercCART.train > 0.5, 1, 0)
table(predCART.train, train$spam)
(2885 + 894) / (2885 + 64 + 167 + 894)



####
# What is the training set AUC of spamCART? (Remember that you have to pass the 
# prediction function predicted probabilities, so don't include the type argument 
# when making predictions for your CART model.)
predictionCART.train <- prediction(predPercCART.train, train$spam)
performanceCART.train <- performance(predictionCART.train, 'tpr', 'fpr')
plot(performanceCART.train, colorize=TRUE)

AUC.tmp <- performance(predictionCART.train, 'auc')
AUC.CART.train <- as.numeric(AUC.tmp@y.values)
AUC.CART.train

# Answer: 0.9696044



####
# What is the training set accuracy of spamRF, using a threshold of 0.5 for 
# predictions? (Remember that your answer might not match ours exactly, due 
# to random behavior in the random forest algorithm on different operating systems.)
head(predPercRF.train)
predRF.train <- ifelse(predPercRF.train > 0.5, 1, 0)
table(predRF.train, train$spam)
(3013 + 914) / (3013 + 44 + 39 + 914)  # sometimes this
(3013 + 917) / (3013 + 41 + 39 + 917)  # sometimes this


####
# What is the training set AUC of spamRF? (Remember to pass the argument type="prob"
# to the predict function to get predicted probabilities for a random forest model.
# The probabilities will be the second column of the output.)

predictionRF.train <- prediction(predPercRF.train, train$spam)
performanceRF.train <- performance(predictionRF.train, 'tpr', 'fpr')
plot(performanceRF.train, colorize=TRUE)

AUC.tmp <- performance(predictionRF.train, 'auc')
AUC.RF.train <- as.numeric(AUC.tmp@y.values)
AUC.RF.train



####
# Which model had the best training set performance, in terms of accuracy and AUC?
# A. Logistic regression 
# B. CART 
# C. Random forest

AUC.Log.train
AUC.CART.train
AUC.RF.train

# Answer: A



####
# Obtain predicted probabilities for the testing set for each of the models, again
# ensuring that probabilities instead of classes are obtained.
# 
# What is the testing set accuracy of spamLog, using a threshold of 0.5 for predictions?
predPercLog.test <- predict(spamLog, newdata = test, type='response')
head(predPercLog.test)
predLog.test <- ifelse(predPercLog.test > 0.5, 1, 0)
table(predLog.test, test$spam)
(1257 + 376) / (1257 + 34 + 51 + 376)



####
# What is the testing set AUC of spamLog?
predictionLog.test <- prediction(predPercLog.test, test$spam)
performanceLog.test <- performance(predictionLog.test, 'tpr', 'fpr')
plot(performanceLog.test, colorize=T)

AUC.tmp <- performance(predictionLog.test, 'auc')
AUC.Log.test <- as.numeric(AUC.tmp@y.values)
AUC.Log.test



####
# What is the testing set accuracy of spamCART, using a threshold of 0.5 for predictions?
predPercCART.test <- predict(spamCART, newdata = test)[ , 2]
head(predPercCART.test)
predCART.test <- ifelse(predPercCART.test > 0.5, 1, 0)
table(predCART.test, test$spam)
(1228 + 386) / (1228 + 24 + 80 + 386)



####
# What is the testing set AUC of spamCART?
predictionCART.test <- prediction(predPercCART.test, test$spam)
performanceCART.test <- performance(predictionCART.test, 'tpr', 'fpr')
plot(performanceCART.test, colorize = TRUE)

AUC.tmp <- performance(predictionCART.test, 'auc')
AUC.CART.test <- as.numeric(AUC.tmp@y.values)
AUC.CART.test  



####
# What is the testing set accuracy of spamRF, using a threshold of 0.5 for predictions?
predPercRF.test <- predict(spamCART, newdata = test)[ , 2]
head(predPercRF.test)
predRF.test <- ifelse(predPercRF.test > 0.5, 1, 0)
table(predRF.test, test$spam)
(1228 + 386) / (1228 + 24 + 80 + 386)



####
# What is the testing set AUC of spamRF?
predictionRF.test <- prediction(predPercRF.test, test$spam)
performanceRF.test <- performance(predictionRF.test, 'tpr', 'fpr')
plot(performanceRF.test, colorize=T)

AUC.tmp <- performance(predictionRF.test, 'auc')
AUC.RF.test <- as.numeric(AUC.tmp@y.values)
AUC.RF.test



####
# Which model had the best testing set performance, in terms of accuracy and AUC?
# A. Logistic regression 
# B. CART 
# C. Random forest

AUC.Log.test
AUC.CART.test
AUC.RF.test

# Answer: C



####
# Which model demonstrated the greatest degree of overfitting?
# A. Logistic regression 
# B. CART 
# C. Random forest

# Answer: A
#
# Explanation: Both CART and random forest had very similar accuracies on the training
# and testing sets. However, logistic regression obtained nearly perfect accuracy and 
# AUC on the training set and had far-from-perfect performance on the testing set. This 
# is an indicator of overfitting.


