# https://www.statmethods.net/advstats/cart.html
data = read.csv("runs_out.csv")

###
# Decision Tree
###
library(rpart)

# grow tree
fit <- rpart(reward ~ ., method="anova", data=data)

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


###
# Random Forest
###

# Random Forest prediction of Kyphosis data
library(randomForest)
library(reprtree)

model <- randomForest(reward ~ ., data=data, importance=TRUE, ntree=500, mtry = 2, do.trace=100)

# visualization, see https://stats.stackexchange.com/a/241684/107199 for install instructions
reprtree:::plot.getTree(model)


### 
# hyper search
###
library(randomForest)
library(mlbench)
library(caret)

# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)