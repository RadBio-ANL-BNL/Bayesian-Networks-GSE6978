#####################################
### Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
### - Associated publication:
### url: 
### doi: 
### github: 
#####################################

#################
### step 1
#################
set.seed(1)
data <- read.csv("data/pathway/pathway_all.csv")
data <- data[, 2:ncol(data)]

for(i in 1:ncol(data)){ data[,i] <- as.factor(data[,i]) }

#################
### step 2
#################
library(bnlearn)

train_idx <- 1:99
train_data <- data[train_idx,]
test_data <- data[-train_idx,]

#################
### step 3
#################
# network structure
net = 'tree'
if(net == 'tree')
  {
  nb.net <- tree.bayes(train_data, "y")     # Tree-Augmented naive Bayes (TAN)
  } else 
  {
  nb.net <- naive.bayes(train_data, "y")  # naive Bayes
  }

nb.fit <- bn.fit(nb.net, train_data)      # fit the network
nb.pred <- predict(nb.fit, test_data)     # oos prediction

mean(nb.pred == test_data$y)

graphviz.plot(nb.fit)


