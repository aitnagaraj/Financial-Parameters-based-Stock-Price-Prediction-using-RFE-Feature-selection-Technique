set.seed(7)
# load the library
library(mlbench)
library(caret)
library(e1071)

# load the data

data <- read.csv(file.choose(), header = T)

data <- na.omit(data)


set.seed(50)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(data[,1:13], data[,14], sizes=c(1:13), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


library("writexl")

library(xlsx)

write.csv(results,file='a.csv')



#decision tree

library(DAAG)
library(party)
library(rpart)
library(rpart.plot)
library(mlbench)
library(caret)
library(pROC)
library(tree)
data <- read.csv(file.choose(), header = T)

data$class=as.factor(data$class)

str(data) 

mydata <-data


set.seed(1234)
ind <- sample(2, nrow(mydata), replace = T, prob = c(0.5, 0.5))
train <- mydata[ind == 1,]
test <- mydata[ind == 2,]

tree <- rpart(class ~., data = train)
rpart.plot(tree)

printcp(tree)
rpart(formula = class ~ ., data = train)

plotcp(tree)

tree <- rpart(class ~., data = train,cp=0.07444)


p <- predict(tree, train, type = 'class')
confusionMatrix(p, train$class, positive='y')


p1 <- predict(tree, test, type = 'prob')
p1 <- p1[,2]
r <- multiclass.roc(test$class, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')



#DNN

# Libraries
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)
library(tensorflow)
data <- read.csv(file.choose(), header = T)


str(data)


data <- as.data.frame(apply(data, 2, as.numeric))
data <- na.omit(data)



data %<>% mutate_if(is.factor, as.numeric)

# Neural Network Visualization
n <- neuralnet(Close ~ Open+High+Low+Volume,
               data = data,
               hidden = c(10,5),
               linear.output = F,
               lifesign = 'full',
               rep=1)
plot(n,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue')

# Matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(.8, .2))
training <- data[ind==1,1:4]
test <- data[ind==2, 1:4]
trainingtarget <- data[ind==1, 5]
testtarget <- data[ind==2, 5]

# Normalize
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)
test <- scale(test, center = m, scale = s)

# Create Model
set.seed(1234)
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = c(4)) %>%
  layer_dense(units = 5, activation = 'relu', input_shape = c(4)) %>%
  layer_dense(units = 5, activation = 'relu', input_shape = c(4)) %>%
  
  
  layer_dense(units = 1)

# Compile
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')
# Fit Model
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 500,
      batch_size = 32,
      validation_split = 0.2)

# Evaluate
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)

mse = mean((testtarget - pred)^2)
mae = caret::MAE(testtarget, pred)
rmse = caret::RMSE(testtarget, pred)


cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

#XGBOOST

library(caret)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)

require(magrittr)

data <- read.csv(file.choose(), header = T)
str(data)


data <- as.data.frame(apply(data, 2, as.numeric))
data <- na.omit(data)

set.seed(12)

indexes = createDataPartition(data$Close, p =.85, list = F)
train = data[indexes, ]
test = data[-indexes, ]

train_x = data.matrix(train[, -5])
train_y = train[,5]

test_x = data.matrix(test[, -5])
test_y = test[, 5]

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#fitting


xgbc = xgboost(data = xgb_train, max.depth = 2, nrounds = 50)
print(xgbc)



pred_y = predict(xgbc, xgb_test)

#accuracy

mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)



