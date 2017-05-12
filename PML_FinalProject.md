# Practical Machine Learning Final Project
Dick Cooman  
May 9, 2017  



NOTE: The class requirements say to upload HTML and R Markdown files to yor Github 
account.  I did that and those are available
at https://github.com/dcooman/Practical_Machine_Learning.  I also created this RPubs 
document which shows the full document with all code and output.  I think this is
a nicer format to review.

### Introduction

This project uses a dataset of exercise data collected from various personal devices.  The 
purpose is to develop a model to predict the class of the dumbell curl movement from the other
variables in the dataset.  The training and validation datasets will each be loaded from a URL.  

The training dataset will be split into an actual training set and a test set.  All three datasets will 
be cleaned up to remove variables which will have little or no effect on
the prediction.  Three models will be created using different methods. I chose Random Forest, Gradient 
Boosted, and Linear Discriminant Analysis because these were used often in the class and they seem to be 
reliable for this type of data.  Models will be created from all three and then will be 
tested for accuracy.  The the best will be applied to the validation set.

### Load LIbraries


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

### Prepare data files

Download the two CVS files and create the complete training and validation data frames.


```r
Train_URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Test_URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(Train_URL,destfile="Training.csv")
download.file(Test_URL, destfile="Testing.csv")
dfWholeFile <- read.csv("Training.csv")
dfValidate <- read.csv("Testing.csv")
dim(dfWholeFile)
```

```
## [1] 19622   160
```

```r
dim(dfValidate)
```

```
## [1]  20 160
```

Split the whole data file into a training set with 70% of the values and a test set with the remaining 30% of the values.


```r
set.seed(1331)
inTrain <- createDataPartition(y=dfWholeFile$classe, p=0.7, list=FALSE)
dfTrain <- dfWholeFile[inTrain,]
dfTest <- dfWholeFile[-inTrain,]
```

Remove troublesome columns such as thw row number and timestamps which will not help in the
analysis.


```r
UselessCols <- c(1, 3:7)
dfTrain <- dfTrain[,-UselessCols]
dfTest <- dfTest[,-UselessCols]
dfValidate <- dfValidate[,-UselessCols]
```

Find columns in the training set that contain any NAs and remove these columns
from ALL data sets to keep the data frames consistent. 


```r
goodCols <- colSums(is.na(dfTrain)) == 0
dfTrain <- dfTrain[,goodCols]
dfTest <- dfTest[,goodCols]
dfValidate <- dfValidate[,goodCols]
```

Now remove columns that have a variance near zero, ie. mostly the same value.  These 
columns will not help much in fitting the model.


```r
badCols <- nearZeroVar(dfTrain)
dfTrain <- dfTrain[,-badCols]
dfTest <- dfTest[,-badCols]
dfValidate <- dfValidate[,-badCols]
dim(dfTrain)
```

```
## [1] 13737    54
```

```r
dim(dfTest)
```

```
## [1] 5885   54
```

```r
dim(dfValidate)
```

```
## [1] 20 54
```

From the output of the initial and final dim statements, the number of columns has 
been reduced from 160 to 54.

### Create a Random Forest model

Create a Random Forest model using cross-validation with three iterations.  Determine 
the accuracy against the test set and the out-of-sample error rate.


```r
tc <- trainControl(method="cv",3)
rfModel <- train(classe~.,data=dfTrain, method="rf", trControl=tc, verbose=FALSE)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
rfPredict <- predict(rfModel,dfTest)
confusionMatrix(dfTest$classe, rfPredict)$overall[1]
```

```
##  Accuracy 
## 0.9947324
```

```r
accuracy <- as.numeric(confusionMatrix(dfTest$classe, rfPredict)$overall[1])
err <- 1 - accuracy
```

For this model, the accuracy is 99.473% and the out-of-sample error 
rate is 0.527%.

### Create a Gradient Boosted model

Create a gradient boosted model (gbm) against the training set and determine the accuracy and out-of-sample 
error rate against the test set.


```r
gbmModel <- train(classe~.,data=dfTrain, method="gbm", trControl=tc, verbose=FALSE)
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.3.3
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.3
```

```
## Loading required package: plyr
```

```r
gbmPredict <- predict(gbmModel,dfTest)
confusionMatrix(dfTest$classe, gbmPredict)$overall[1]
```

```
##  Accuracy 
## 0.9634664
```

```r
accuracy <- as.numeric(confusionMatrix(dfTest$classe, gbmPredict)$overall[1])
err <- 1 - accuracy
```

For this model, the accuracy is 96.347% and the out-of-sample error 
rate is 3.653%.

### Create a Linear Discriminant Model

Create a third model by using linear discriminant analysis (ldm) against the training set and determine the accuracy and out-of-sample error rate against the test set.


```r
ldaModel <- train(classe~.,data=dfTrain, method="lda", trControl=tc, verbose=FALSE)
```

```
## Loading required package: MASS
```

```r
ldaPredict <- predict(ldaModel,dfTest)
confusionMatrix(dfTest$classe, ldaPredict)$overall[1]
```

```
##  Accuracy 
## 0.7262532
```

```r
accuracy <- as.numeric(confusionMatrix(dfTest$classe, ldaPredict)$overall[1])
err <- 1 - accuracy
```

For this model, the accuracy is 72.625% and the out-of-sample error 
rate is 27.375%.

### Use the best model on the validation set

From the above results, it appears that the Random Forest model is the most accurate.  Use this
model to predict the results in the validation set.


```r
finalPredict <- predict(rfModel, dfValidate)
finalPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Conclusions

Of the three models chosen, Random Forest appears to be the best.  Gradient Boosted comes in second and Linear Discriminant performed very poorly. Perhaps increasing the number of cross-validation interations could improve
this model.
