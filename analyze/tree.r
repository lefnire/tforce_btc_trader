# https://www.statmethods.net/advstats/cart.html
data = read.csv("runs.csv")

# Regression Tree Example
library(rpart)

# grow tree
fit <- rpart(target ~ ., method="anova", data=data)

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# create additional plots
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(fit) # visualize cross-validation results

# plot tree
plot(fit, uniform=TRUE, main="Regression Tree")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postcript plot of tree
post(fit, file = "tree2.ps", title = "Regression Tree")


### Random Forest visualization, see https://stats.stackexchange.com/a/241684/107199 for install instructions ####

# Random Forest prediction of Kyphosis data
library(randomForest)
library(reprtree)

model <- randomForest(target ~ ., data=data, importance=TRUE, ntree=500, mtry = 2, do.trace=100)
reprtree:::plot.getTree(model)