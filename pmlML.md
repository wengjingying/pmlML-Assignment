Practical Machine Learning Assignment

==============================================



## Background

<br>

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

<br>

## Project Abstract:

<br>

The goal of this project is to predict the manner in which they did the exercise using any of the other variables. This is the "classe" variable in the training set. A report is created describing how the model is built, cross validation, what is the expected out of sample error, and the choice is made. The prediction model is also used to predict 20 different test cases.

<br>

## About "classe"" variables

exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)


### 1. Obtain datasets and preprocess

```r
setwd("F:/Mooc/Data Science/8 Machine Learning/w4")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./pml-training.csv")
download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./pml-testing.csv")

trainingData <- read.csv("pml-training.csv", stringsAsFactors=F, na.string = c("", "NA", "Null"))
testingData <- read.csv("pml-testing.csv", stringsAsFactors=F, na.string = c("", "NA", "Null"))

trainingData$classe<-as.factor(trainingData$classe)
dim(trainingData)
```

```
## [1] 19622   160
```

```r
dim(testingData)
```

```
## [1]  20 160
```
* Training data has 159 variables, which is too many, we need to narrow down the variables for prediction.

<br>

### 2. Problems in the data - Too many NAs

<br>

Count every columns' with NA

```r
NAcount<-function(l){sum(is.na(trainingData[,l]))}
NAC<-sapply(c(1:160),function(x)NAcount(x))
noNA<-sum(NAC==0)
firstNA<-which(NAC>0)[1]
```
There are only 60 columns without NAs,begins at column 12.

Omit those columns.

```r
train1<-trainingData[, -which(NAC>0)]
```

### 3. Preprocess

* Also remove unrelevant variables like username, timestamp(testing is not based on time serials)

```r
remove = c("X", 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
train1<-train1[,-which(names(train1) %in% remove)]
```

* Separate training set for cross validation:

```r
library(caret)
inTrain<-createDataPartition(y=train1$classe,p=0.7,list=F)
training<-train1[inTrain,]
Validating<-train1[-inTrain,]
```

<br>

* Look at the correlation between variables:

```r
M <-abs(cor(training[,-53]))
diag(M)<-0
Cor <-which(M>0.8, arr.ind=T)
NumCor <- length(unique(Cor[,2]))
```

* There are 22 variables highly correlated. We should pre-process data using PCA (Principle component analysis).

### 4.1 First Model fit using caret library, rpart method.

```r
set.seed(12345)
modFitRP<-train(classe~ .,method="rpart",data=training,preProcess="pca")
print(modFitRP$finalModel)
```

```
## n= 13737 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
##   2) PC14>=-0.8596774 12264 8404 A (0.31 0.18 0.19 0.17 0.14) *
##   3) PC14< -0.8596774 1473  713 E (0.031 0.27 0.041 0.15 0.52) *
```
* We can see the problem is there is no prediction rules for Classe B & C.

### 4.2 Second Model fit using random forest method.

```r
require(randomForest)
set.seed(12345)
modFitRF=randomForest(classe~.,data=training,ntree=100, importance=TRUE, preProcess="pca")
modFitRF
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, ntree = 100,      importance = TRUE, preProcess = "pca") 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.64%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B   13 2638    6    1    0 0.0075244545
## C    0   19 2372    5    0 0.0100166945
## D    0    0   27 2222    3 0.0133214920
## E    0    0    4    9 2512 0.0051485149
```

```r
varImpPlot(modFitRF,)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png) 

* We can see random forest predicted all five classes, used 7 variables. The OOB estimate of error rate is low. From the variable importance plot, we can see top 7 variables important and been used in the prediction.

<br>

### 5. Evaluate Rpart and Random Forest prediction results on Validating Data.
* Rpart

```r
RpartPred = predict(modFitRP, Validating)
predMatrix1 = with(Validating, table(RpartPred, classe))
a1<-sum(diag(predMatrix1))/sum(as.vector(predMatrix1))
a1
```

```
## [1] 0.337808
```
* The 0.337808 accuracy is very low for Rpart method from caret package. 
* Out of sample error  = 0.662192

<br>

* Random Forest

```r
RFpred = predict(modFitRF, Validating)
predMatrix2 = with(Validating, table(RFpred, classe))
a2<-sum(diag(predMatrix2))/sum(as.vector(predMatrix2))
a2
```

```
## [1] 0.993373
```
* Apparantly, Random Forest have much higher and good accuracy 0.993373 than Rpart.
* Out of sample error = 0.006627

<br>

### 6. Predict Testing Data

* Now we can use our random forest model to predict 20 observation in testing data.

```r
predTest <- predict(modFitRF, testingData)
predTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```



