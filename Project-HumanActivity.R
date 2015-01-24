library(caret)
library(dplyr)
library(doParallel)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


set.seed(123)
registerDoParallel(cores = 4)

train <- read.csv('pml-training.csv')
test <- read.csv('pml-testing.csv')

test$classe <- NA
test$set <- "Test"
test$problem_id <- NULL
train$set <- "Train"
full_set <- rbind(train, test)

full_set$classe <- factor(full_set$classe)

dataToDrop <- list()
j <- 0
for (i in names(full_set)) {
  if (!is.numeric(full_set[[i]]) | sum(is.na(full_set[[i]]))>0) {
    if (j<159) { ## hold classe and set attributes
      dataToDrop[j] <- i      
    }
  }
  j <- j + 1
}


data_set <- full_set[, -which(names(full_set) %in% dataToDrop)]
data_set$X <- NULL

## -num_window, -raw_timestamp_part_1, -raw_timestamp_part_2
data_set <- select(data_set, -num_window, -raw_timestamp_part_1, -raw_timestamp_part_2)

## Split data_set into train and 
index_set <- createDataPartition(data_set[data_set$set=="Train",]$classe, p=0.6, list=F)


## preprocess, maintain 38 features of 161 by scaling and centering as well
prep <- preProcess(select(filter(data_set, set=="Train"), -set, -classe)[index_set,], method=c("center","scale", "pca"), thresh=0.99)

cv <- trainControl(method = "cv", number = 10)


prepTrain <- predict(prep, select(filter(data_set, set=="Train")[index_set,], -set, -classe))
prepCv <- predict(prep, select(filter(data_set, set=="Train")[-index_set,], -set, -classe))


prepTest <- predict(prep, select(filter(data_set, set=="Test"), -set, -classe))

svmModel <- train(prepTrain, data_set[index_set,]$classe, method="svmRadial", trControl = cv)
svmResultTrain <-  predict(svmModel, prepTrain)
svmResultCv <-  predict(svmModel, prepCv)
confusionMatrix(filter(data_set, set=="Train")[-index_set,]$classe, svmResultCv)

rfModel <- train(prepTrain, data_set[index_set,]$classe, method="rf", trControl = cv, ntree=300)
rfResultTrain <-  predict(rfModel, prepTrain)
rfResultCv <-  predict(rfModel, prepCv)
confusionMatrix(filter(data_set, set=="Train")[-index_set,]$classe, rfResultCv)



svmResult <- predict(svmModel, prepTest)
rfResult <- predict(rfModel, prepTest)
logTrain <- data.frame(svmResultTrain, rfResultTrain, prepTrain)
logCv <- data.frame(svmResultTrain=svmResultCv, rfResultTrain=rfResultCv, prepCv)
logTest <- data.frame(svmResultTrain=svmResult, rfResultTrain=rfResult, prepTest)
logModel <- train(y=filter(data_set, set=="Train")[index_set,]$classe, x=logTrain, model="glm", family=binomial())
resultCv <- predict(logModel, logCv)

confusionMatrix(filter(data_set, set=="Train")[-index_set,]$classe, resultCv)

result <- predict(logModel, logTest)

## print rfResult. The 3 models output the same correct result.
pml_write_files(rfResult)
