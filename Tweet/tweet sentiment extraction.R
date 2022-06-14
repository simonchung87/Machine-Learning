#####Necessary function#####
library(ggplot2)
library(yardstick)
accuracy <- function(predict, actual){
  predict_right <- sum(predict == actual)
  return(sum(predict == actual)/length(actual))
} ##計算準確率
confusion_matrix <- function(predict, actual){
  print(table(predict, actual))
  cat("\n Accuracy: ", accuracy(predict, actual))
  combine_data <- data.frame(prediction = predict,
                             real = actual)
  combine_data$prediction <- as.factor(combine_data$prediction)
  combine_data$real <- as.factor(combine_data$real)
  cm <- conf_mat(data = combine_data, real, prediction)
  autoplot(cm, type = "heatmap") +
    scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
    theme(legend.position = "right")
} ##print出準確率，並畫出confusion matrix的heat map
#####data pre-processing#####
library(dplyr)
library(stringr)
library(tm)
library(stringi)

train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")
train_data$text <- gsub("http[[:alnum:][:punct:]]*", "", train_data$text) #remove string start with http
train_data$text <- as.character(train_data$text) 
train_data$text <- removeNumbers(removePunctuation(train_data$text)) #remove punctuation and numbers
train_data$text <- strsplit(str_to_lower(train_data$text), " ") #convert string into lower case

test_data$text <- gsub("http[[:alnum:][:punct:]]*", "", test_data$text) #remove string start with http
test_data$text <- as.character(test_data$text) 
test_data$text <- removeNumbers(removePunctuation(test_data$text)) #remove punctuation and numbers
test_data$text <- strsplit(str_to_lower(test_data$text), " ") #convert string into lower case

train_data$selected_text <- as.character(train_data$selected_text) 
train_data$selected_text <- removeNumbers(removePunctuation(train_data$selected_text))
train_data$selected_text <- strsplit(str_to_lower(train_data$selected_text), " ")

sentiment <- gsub("positive", 1, gsub("negative", -1, gsub("neutral", 0, train_data$sentiment))) 
test_sentiment <- gsub("positive", 1, gsub("negative", -1, gsub("neutral", 0, test_data$sentiment))) ##將sentiment轉化成數字
train_data$sentiment <- sentiment <- as.integer(sentiment)
test_data$sentiment <- test_sentiment <- as.integer(test_sentiment)

train_data$selected_text <- sapply(train_data$selected_text, stri_remove_empty)
train_data$text <- sapply(train_data$text, stri_remove_empty)
test_data$text <- sapply(test_data$text, stri_remove_empty)

train_set <- (train_data[-grep("^0$", train_data$sentiment),])[1:10000,] ##去除neutral的資料並取前10000筆資料
test_set <- test_data

corpus <- names(table(unlist(train_set$text))) ##提取這些推文中的單字作為語料庫
count_table <- dplyr::select(train_set,
                             textID = textID,
                             sentiment = sentiment)

test_count_table <- dplyr::select(test_set,
                                  textID = textID,
                                  sentiment = sentiment)

corpus_table <- data.frame(text = corpus,
                           weight = rep(0, length(corpus)))
positive_data <- train_set[grep("^1$", train_set$sentiment),] ##將sentiment為positive和negative的資料分開
negative_data <- train_set[grep("^-1$", train_set$sentiment),]
positive_words <- unlist(positive_data$selected_text)
negative_words <- unlist(negative_data$selected_text) ##將他們的selected text的單字提取出來
for(i in 1:dim(corpus_table)[1]){
  word <- corpus_table[i,1]
  corpus_table[i,2] <- length(grep(paste("^", word, "$", sep = ""), positive_words)) - length(grep(paste("^", word, "$", sep = ""), negative_words))
} ##計算在positive中出現的次數減去在negative中出現的次數
corpus_table$weight <- scale(corpus_table$weight, center = FALSE, scale = TRUE) #regulization
feature_matrix <- matrix(NA, ncol = 1, nrow = dim(count_table)[1])
test_feature_matrix <- matrix(NA, ncol = 1, nrow = dim(test_count_table)[1])
i <- 2
for(word in corpus_table$text){
  word_count <- c()
  i <- i + 1
  for(row in train_set$text){
    #row_word_count <- length(grep(paste("^", word, "$", sep = ""), unlist(row)))
    row_word_count <- length(which(unlist(row) == word))
    word_count <- c(word_count, row_word_count)
  }
  feature_matrix <- cbind(feature_matrix, word_count)
  word_count <- c()
  for(row in test_set$text){
    #row_word_count <- length(grep(paste("^", word, "$", sep = ""), unlist(row)))
    row_word_count <- length(which(unlist(row) == word))
    word_count <- c(word_count, row_word_count)
  }
  test_feature_matrix <- cbind(test_feature_matrix, word_count)
  cat("\f")
  cat(100*((i-2)/length(corpus)),"%")
}##計算語料集中的單字在每個句子出現的次數
feature_matrix <- feature_matrix[,-1]
test_feature_matrix <- test_feature_matrix[,-1]
count_table <- cbind(count_table, feature_matrix)
test_count_table <- cbind(test_count_table, test_feature_matrix)
colnames(count_table)[3:(length(corpus)+2)] <- corpus
colnames(test_count_table)[3:(length(corpus)+2)] <- corpus

#####logistic regression#####
predict_probability <- function(feature_matrix, coefficients){
  scores <- feature_matrix %*% coefficients #1
  predictions <- 1/(1 + exp(-scores))
  return(predictions)
} ##計算機率值

feature_derivative <- function(errors, feature){
  derivative <- matrix(errors, nrow = 1) %*% feature 
  return(derivative)
}

predict_sentiment <- function(logistic_coefficients, test_feature_matrix){
  test_prediction_score <- test_feature_matrix %*% logistic_coefficients
  return(2*(test_prediction_score>0) - 1)
} ##預測情感

compute_log_likelihood <- function(feature_matrix, sentiment, coefficients){
  indicator <- sentiment==1
  scores <- feature_matrix %*% as.matrix(coefficients) #2
  logexp <- log(1 + exp(-scores))
  
  lp <- sum((indicator-1)*scores - logexp)
  return(lp)
}

logistic_regression <- function(feature_matrix, sentiment, initial_coefficients, step_size, max_iter, messages = TRUE){
  coefficients <- initial_coefficients
  for(itr in 1:max_iter){
    predictions <- predict_probability(feature_matrix, coefficients)
    indicator <- (sentiment == 1)
    errors <- indicator - predictions
    for(j in 1:length(coefficients)){
      derivative <- feature_derivative(errors, feature_matrix[,j])
      coefficients[j] <- coefficients[j] + step_size *derivative /sqrt(itr) 
    }
    if(messages){
      if(itr <= 15 || (itr <= 100 & itr %% 10 == 0) || (itr <= 1000 & itr %% 100 == 0) || (itr <= 10000 & itr %% 1000 == 0) || itr %% 10000 == 0){
        cat("\n iteration:", itr,", log likelihood of observed labels",compute_log_likelihood(feature_matrix, sentiment, coefficients))
      }
    }
  }
  return(coefficients)
}

tune_logistic_regression <- function(feature_matrix, sentiment, initial_coefficients, step_size, max_iter, valid_p = 0.8, messages = TRUE){
  coefficients <- initial_coefficients
  data_split_line <- length(sentiment) * c(valid_p, 1)
  training_matrix <- feature_matrix[1:data_split_line[1],]
  valid_matrix <- feature_matrix[(data_split_line[1]+1):data_split_line[2],]
  train_sentiment <- sentiment[1:data_split_line[1]]
  valid_sentiment <- sentiment[(data_split_line[1]+1):data_split_line[2]]
  best_accuracy <- 0
  for(itr in 1:max_iter){
    predictions <- predict_probability(training_matrix, coefficients)
    indicator <- (train_sentiment == 1)
    errors <- indicator - predictions
    for(j in 1:length(coefficients)){
      derivative <- feature_derivative(errors, training_matrix[,j])
      coefficients[j] <- coefficients[j] + step_size *derivative /sqrt(itr) 
    }
    valid_prediction <- predict_sentiment(coefficients, valid_matrix)
    valid_accuracy <- accuracy(valid_prediction, valid_sentiment)
    if(valid_accuracy>best_accuracy){
      best_accuracy <- valid_accuracy
      best_iteration <- itr
    }
    if(messages){
      if(itr <= 15 || (itr <= 100 & itr %% 10 == 0) || (itr <= 1000 & itr %% 100 == 0) || (itr <= 10000 & itr %% 1000 == 0) || itr %% 10000 == 0){
        cat("\n iteration:", itr,", log likelihood of observed labels",compute_log_likelihood(training_matrix, train_sentiment, coefficients), ", validation accuracy:", valid_accuracy)
      }
    }
  }
  return(list(accuracy = best_accuracy, iteration = best_iteration))
} ##用來找到最合適的迭代次數

lr_training_matrix <- feature_matrix
lr_training_sentiment <- count_table$sentiment

best_logistic_tune <- tune_logistic_regression(lr_training_matrix, lr_training_sentiment, corpus_table$weight, 10^(-1.5),500) ##先找到合適的迭代次數
logistic_coefficients <- logistic_regression(lr_training_matrix, lr_training_sentiment, corpus_table$weight, 10^(-1.5), best_logistic_tune$iteration) ##在使用整筆資料訓練

test_prediction_sentiment <- predict_sentiment(logistic_coefficients, test_feature_matrix[-grep(0, test_count_table$sentiment),]) #predict test set's sentiment
confusion_matrix(test_prediction_sentiment,test_count_table$sentiment[-grep(0, test_count_table$sentiment)]) #print the accuracy and confusion matrix of test set

#####random forest with PSO#####
library(ranger)       # a faster implementation of randomForest
library(dplyr)

randomforest_training_data <- count_table
names(randomforest_training_data) <- make.names(names(randomforest_training_data))
randomforest_training_data$sentiment <- as.factor(randomforest_training_data$sentiment)

p_num <- 5 ##粒子個數為5
itr_num <- 10 ##迭代次數為10
w <- 0.5
c1 <- 2
c2<- 2
X <- data.frame(
  trees = sample(seq(0,1000, by = 10), p_num),
  mtry = sample(seq(0, 100, by = 1), p_num),
  node_size = sample(seq(1, 20, by = 1), p_num),
  OOB_RMSE = vector("double", p_num),
  seed = vector("integer", p_num)
) ##初始化粒子，也就是隨機生成參數組合
for(i in 1:p_num){
  seed.number = sample.int(10000, 1)[[1]]
  model <- ranger(
    formula = sentiment ~ .-textID,
    data = randomforest_training_data, 
    num.trees = X$trees[i], 
    mtry = X$mtry[i],
    min.node.size = X$node_size[i], 
    sample.fraction = 0.8,
    seed = seed.number,
    num.threads = 3
  )
  X$OOB_RMSE[i] <- sqrt(model$prediction.error)
  X$seed[i] <- seed.number
} ##計算初始粒子的out of bag error
V <-data.frame(
  trees = rep(0, p_num),
  mtry = rep(0, p_num),
  node_size = rep(0, p_num)
) ##紀錄每個粒子的速度
Pb <- data.frame(
  trees = rep(0, p_num),
  mtry = rep(0, p_num),
  node_size = rep(0, p_num),
  OOB_RMSE = rep(Inf, p_num),
  seed = rep(0, p_num)
) ##紀錄每個粒子經歷的最佳的位置
Gb <- data.frame(
  trees = 0,
  mtry = 0,
  node_size = 0,
  OOB_RMSE = Inf,
  seed = 0
) ##紀錄有史以來最佳的位置
l <- 0
for(i in 1:itr_num){
  for(j in 1:p_num){
    if(Pb$OOB_RMSE[j]>X$OOB_RMSE[j]) Pb[j,]<-X[j,]
  }
  if(Gb$OOB_RMSE > min(Pb$OOB_RMSE)) Gb <- Pb[grep(min(Pb$OOB_RMSE), Pb$OOB_RMSE)[1],]
  for(j in 1:p_num){
    for(k in 1:3){
      V[j,k] <- as.integer(w * V[j,k] + c1 * runif(1) * (Pb[j,k]-X[j,k]) + c2 * runif(1) * (Gb[k]-X[j,k])) ##計算速度
      X[j,k] <- X[j,k] + V[j,k] ##計算新的位置
      if(X[j,k]<1) X[j,k] <- 1
      if(X[j,k]>1000) X[j,k] <- 1000 ##粒子的上界和下界
    }
    seed.number = sample.int(10000, 1)[[1]]
    model <- ranger(
      formula = sentiment ~ .-textID,
      data = randomforest_training_data, 
      num.trees = X$trees[j],
      mtry = X$mtry[j],
      min.node.size = X$node_size[j], 
      sample.fraction = 0.8,
      seed = seed.number
    )
    X$OOB_RMSE[j] <- sqrt(model$prediction.error)
    X$seed[j] <- seed.number
  }
  if(i == itr_num){
    for(j in 1:p_num){
      if(Pb$OOB_RMSE[j]>X$OOB_RMSE[j]) Pb[j,]<-X[j,]
    }
    if(Gb$OOB_RMSE > min(Pb$OOB_RMSE)) Gb <- Pb[grep(min(Pb$OOB_RMSE), Pb$OOB_RMSE)[1],]
  }
  print(Gb$OOB_RMSE)
} ##迭代

rf_Gb <- Gb
rf_model <- ranger(
  formula = sentiment ~ .-textID,
  data = randomforest_training_data,
  num.trees = rf_Gb$trees, 
  mtry = rf_Gb$mtry,
  min.node.size = rf_Gb$node_size, 
  sample.fraction = 0.8,
  seed = rf_Gb$seed
) ##使用最佳的參數去訓練模型

randomforest_testing_data <- test_count_table[-grep(0, test_count_table$sentiment),] 
names(randomforest_testing_data) <- make.names(names(randomforest_testing_data)) 
rf_prediction_sentiment <- predict(rf_model, data = randomforest_testing_data) ##預測
confusion_matrix(rf_prediction_sentiment$predictions, test_count_table$sentiment) ##混淆矩陣

#####Extremely Randomized Tree with PSO#####
library(MASS)
library(caret)
options(java.parameters = "-Xmx8g")
library(extraTrees)

erf_training_data <- count_table[,-c(1,2)]
names(erf_training_data) <- make.names(names(erf_training_data))
erf_sentiment <- as.factor(count_table$sentiment)
training_index <- c(1:(0.8*dim(erf_training_data)[1]))

p_num <- 5
itr_num <- 5
w <- 0.5
c1 <- 2
c2<- 2
X <- data.frame(
  trees = sample(seq(0,1000, by = 10), p_num),
  mtry = sample(seq(0, 100, by = 1), p_num),
  node_size = sample(seq(1, 20, by = 1), p_num),
  seed = vector("integer", p_num),
  accuracy = vector("double", p_num)
)
for(i in 1:p_num){
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  model <- extraTrees(x = erf_training_data[training_index,],
                      y = erf_sentiment[training_index],
                      ntree = X$trees[i],
                      mtry = X$mtry[i],
                      nodesize = X$node_size[i],
                      numThreads = 3)
  valid_prediction <- predict(model, erf_training_data[-training_index,])
  X$accuracy[i] <- accuracy(valid_prediction, erf_sentiment[-training_index])
  X$seed[i] <- seed.number
}
V <-data.frame(
  trees = rep(0, p_num),
  mtry = rep(0, p_num),
  node_size = rep(0, p_num)
)
Pb <- data.frame(
  trees = rep(0, p_num),
  mtry = rep(0, p_num),
  node_size = rep(0, p_num),
  seed = rep(0, p_num),
  accuracy = rep(0, p_num)
)
Gb <- data.frame(
  trees = 0,
  mtry = 0,
  node_size = 0,
  seed = 0,
  accuracy = 0
)

for(i in 1:itr_num){
  cat("iteration:", i)
  for(j in 1:p_num){
    if(Pb$accuracy[j]<X$accuracy[j]) Pb[j,]<-X[j,]
  }
  if(Gb$accuracy < max(Pb$accuracy)) Gb <- Pb[grep(max(Pb$accuracy), Pb$accuracy)[1],]
  for(j in 1:p_num){
    for(k in 1:3){
      V[j,k] <- as.integer(w * V[j,k] + c1 * runif(1) * (Pb[j,k]-X[j,k]) + c2 * runif(1) * (Gb[k]-X[j,k]))
      X[j,k] <- X[j,k] + V[j,k]
      if(X[j,k]<1) X[j,k] <- 1
      if(X[j,k]>1000) X[j,k] <- 1000
    }
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    X$seed[j] <- seed.number
    model <- extraTrees(x = erf_training_data[training_index,],
                        y = erf_sentiment[training_index],
                        ntree = X$trees[j],
                        mtry = X$mtry[j],
                        nodesize = X$node_size[j])
    valid_prediction <- predict(model, erf_training_data[-training_index,])
    X$accuracy[j] <- accuracy(valid_prediction, erf_sentiment[-training_index])
  }
  if(i == itr_num){
    for(j in 1:p_num){
      if(Pb$accuracy[j]<X$accuracy[j]) Pb[j,]<-X[j,]
    }
    if(Gb$accuracy < max(Pb$accuracy)) Gb <- Pb[grep(max(Pb$accuracy), Pb$accuracy)[1],]
  }
  print(Gb$accuracy)
}
ert_Gb <- Gb
set.seed(Gb$seed)
erf_model <- extraTrees(x = erf_training_data,
                        y = erf_sentiment,
                        ntree = ert_Gb$trees[j],
                        mtry = ert_Gb$mtry[j],
                        nodesize = ert_Gb$node_size[j]) ##使用最佳參數去訓練模型

erf_testing_data <- test_count_table
names(erf_testing_data) <- make.names(names(erf_testing_data))
erf_prediction <- predict(erf_model, erf_testing_data)
confusion_matrix(erf_prediction, as.factor(test_count_table$sentiment[-grep(0, test_count_table$sentiment)])) ##混淆矩陣

#####XGBoost with PSO#####
library(gbm)
library(xgboost)
library(Matrix)
library(glmnetUtils)

xgb_training_data <- count_table[,-1]
xgb_training_label <- count_table$sentiment
xgb_training_label <- (xgb_training_label + 1)/2
m <- nlevels(as.factor(xgb_training_label))

split_line <- c(0.8, 1) * dim(xgb_training_data)[1]
new_train <- model.matrix(~.+0, data = xgb_training_data[1:split_line[1], -1], with = F)
new_valid <- model.matrix(~.+0, data = xgb_training_data[(split_line[1]+1):split_line[2], -1], with = F) ##one-hot encoding
dtrain <- xgb.DMatrix(data = new_train, label = xgb_training_label[1:split_line[1]])
dvalid <- xgb.DMatrix(data = new_valid, label = xgb_training_label[(split_line[1]+1):split_line[2]]) ##轉化成xgboost需要的格式

p_num <- 5
itr_num <- 5
w <- 0.5
c1 <- 2
c2<- 2
X <- data.frame(
  eta = runif(p_num, .01, .3),
  max_depth = sample(1:10, p_num, replace = TRUE),
  gamma=sample(0:10, p_num, replace = TRUE),
  subsample=runif(p_num, .6, .9),
  colsample_bytree=runif(p_num, .5, .8),
  seed = vector("integer", p_num),
  loss = vector("double", p_num)
)
for(i in 1:p_num){
  params = list(
    booster="gbtree",
    eta = X$eta[i],
    max_depth=X$max_depth[i],
    gamma=X$gamma[i],
    subsample=X$subsample[i],
    colsample_bytree=X$colsample_bytree[i],
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=m,
    nthread = 3
  )
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  xgb.fit=xgb.train(
    params=params,
    data=dtrain,
    nrounds=50,
    nthreads=1,
    early_stopping_rounds=10,
    watchlist=list(val1=dtrain,val2=dvalid),
    verbose=0
  )
  X$loss[i] <- min(xgb.fit$evaluation_log[,val2_mlogloss])
  X$seed[i] <- seed.number
}
V <-data.frame(
  eta = rep(0, p_num),
  max_depth = rep(0, p_num),
  gamma=rep(0, p_num),
  subsample=rep(0, p_num),
  colsample_bytree=rep(0, p_num)
)
Pb <- data.frame(
  eta = rep(0, p_num),
  max_depth = rep(0, p_num),
  gamma=rep(0, p_num),
  subsample=rep(0, p_num),
  colsample_bytree=rep(0, p_num),
  seed = rep(0, p_num),
  loss = rep(Inf, p_num)
)
Gb <- data.frame(
  eta = 0,
  max_depth = 0,
  gamma = 0,
  subsample = 0,
  colsample_bytree = 0,
  seed = 0,
  loss = Inf
)

for(i in 1:itr_num){
  cat("iteration:", i)
  for(j in 1:p_num){
    if(Pb$loss[j]>X$loss[j]) Pb[j,]<-X[j,]
  }
  if(Gb$loss > min(Pb$loss)) Gb <- Pb[grep(min(Pb$loss), Pb$loss)[1],]
  for(j in 1:p_num){
    for(k in 1:5){
      V[j,k] <- w * V[j,k] + c1 * runif(1) * (Pb[j,k]-X[j,k]) + c2 * runif(1) * (Gb[k]-X[j,k])
      X[j,k] <- X[j,k] + V[j,k]
    }
    if(X$eta[j] > 1 | X$eta[j] < 0) X$eta[j] <- runif(1)
    X$max_depth[j] <- as.integer(X$max_depth[j])
    if(X$max_depth[j] < 1) X$max_depth[j] <- 1
    X$gamma[j] <- as.integer(X$gamma[j])
    if(X$gamma[j] < 0) X$gamma[j] <- 0
    if(X$subsample[j] > 1 | X$subsample[j] < 0) X$subsample[j] <- runif(1)
    if(X$colsample_bytree[j] > 1 | X$colsample_bytree[j] < 0) X$colsample_bytree[j] <- runif(1)
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    X$seed[j] <- seed.number
    params = list(
      booster="gbtree",
      eta = X$eta[j],
      max_depth=X$max_depth[j],
      gamma=X$gamma[j],
      subsample=X$subsample[j],
      colsample_bytree=X$colsample_bytree[j],
      objective="multi:softprob",
      eval_metric="mlogloss",
      num_class=m
    )
    xgb.fit=xgb.train(
      params=params,
      data=dtrain,
      nrounds=50,
      nthreads=1,
      early_stopping_rounds=10,
      watchlist=list(val1=dtrain,val2=dvalid),
      verbose=0
    )
    X$loss[j] <- min(xgb.fit$evaluation_log[,val2_mlogloss])
  }
  if(i == itr_num){
    for(j in 1:p_num){
      if(Pb$loss[j]>X$loss[j]) Pb[j,]<-X[j,]
    }
    if(Gb$loss > min(Pb$loss)) Gb <- Pb[grep(min(Pb$loss), Pb$loss)[1],]
  }
  print(Gb$loss)
}

xgb_Gb <- Gb

best_params = list(
  booster="gbtree",
  eta = xgb_Gb$eta,
  max_depth = xgb_Gb$max_depth,
  gamma = xgb_Gb$gamma,
  subsample = xgb_Gb$subsample,
  colsample_bytree = xgb_Gb$colsample_bytree,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = m
)
set.seed(xgb_Gb$seed)
xgb.fit=xgb.train(
  params=best_params,
  data=dtrain,
  nrounds=500,
  nthreads=3,
  early_stopping_rounds=20,
  watchlist=list(val1=dtrain, val2 = dvalid),
  verbose=2
)
plot(1:length(xgb.fit$evaluation_log[,val2_mlogloss]), xgb.fit$evaluation_log[,val2_mlogloss], type = "l")

best_iteration <- grep(min(xgb.fit$evaluation_log[,val2_mlogloss]), xgb.fit$evaluation_log[,val2_mlogloss]) ##找到最佳的迭代次數

new_train <- model.matrix(~.+0, data = xgb_training_data[, -1], with = F)
dtrain <- xgb.DMatrix(data = new_train, label = xgb_training_label)

xgb.fit=xgb.train(
  params=best_params,
  data=dtrain,
  nrounds=best_iteration,
  nthreads=3,
  early_stopping_rounds=10,
  watchlist=list(val1=dtrain),
  verbose=2
) ##使用整筆資料進行訓練

test_count_matrix <- model.matrix(~.+0, data = test_count_table[,-c(1,2)], with = F)
prediction <- as.data.frame(predict(xgb.fit, test_count_matrix, reshape=T))
colnames(prediction) <- levels(as.factor(c(-1,1)))
xgb_test_predict <- apply(prediction,1,function(x) colnames(prediction)[which.max(x)])
confusion_matrix(xgb_test_predict, test_count_table$sentiment)

#####Catboost#####
library(catboost)
cat_training_data <- data.frame(lapply(count_table[,-c(1,2)], as.numeric))
cat_training_label <- (count_table$sentiment + 1)/2
split_line <- c(0.8, 1) * length(cat_training_label)
train_pool <- catboost.load_pool(data = cat_training_data[1:split_line[1],], label = cat_training_label[1:split_line[1]])
valid_pool <- catboost.load_pool(data = cat_training_data[(split_line[1]+1):split_line[2],], label = cat_training_label[(split_line[1]+1):split_line[2]]) ##轉化成catboost需要的格式
model <- catboost.train(train_pool,  valid_pool,
                        params = list(loss_function = 'MultiClass',
                                      iterations = 10000, metric_period=50, thread_count = 4)) ##找到最好的迭代次數
train_pool <- catboost.load_pool(data = cat_training_data, label = cat_training_label)
model <- catboost.train(train_pool,  NULL,
                        params = list(loss_function = 'MultiClass',
                                      iterations = model$tree_count, metric_period=10, thread_count = 4)) ##使用整筆資料訓練

real_pool <- catboost.load_pool(data.frame(lapply(test_count_table[,-c(1,2)], as.numeric)))
prediction <- catboost.predict(model, real_pool, prediction_type = "Probability")
colnames(prediction) <- levels(as.factor(c(-1,1)))
cat_test_prediction <- apply(prediction,1,function(x) colnames(prediction)[which.max(x)]) ##預測
confusion_matrix(cat_test_prediction, test_count_table$sentiment) ##混淆矩陣


#####vote#####
vote <- function(predict1, predict2, predict3, predict4, predict5){
  final_score <- 3*as.integer(predict1) + 1*as.integer(predict2) + 1*as.integer(predict3) + 1*as.integer(predict4) + 1*as.integer(predict5)
  final_prediction <- as.integer(final_score>=0)*2-1
  return(final_prediction)
} ##投票
final_prediction <- vote(test_prediction_sentiment, 
                         as.character(rf_prediction_sentiment$predictions),
                         as.character(erf_prediction),
                         xgb_test_predict,
                         cat_test_prediction) ##vote ensemble
confusion_matrix(final_prediction, test_count_table$sentiment) ##混淆矩陣
