---
title: "Machine Learning Project"
author: "Sean Fitzpatrick"
date: "21-7-2015"
output: html_document
---



## Synopsis
This paper will attempt to build a machine learning model using the 'Weight Lifting Dataset' in order to predict the efficacy with which a subject performs a weight lifting exercise.

The interested reader is referred to http://groupware.les.inf.puc-rio.br/har for further information on this dataset and it's characteristics.

## Data Input
We define here a generic function which will download a web based resource optionally unzipping if it required


```r
dataLoader <- function(vDataFile, vSourceURL, unzip = FALSE){    
    if(!file.exists(vDataFile))
        {
        vDestFile = basename(vSourceURL)
        message("Downloading data file - please wait ... ", appendLF = FALSE)
        download.file(url = vSourceURL, destfile = vDestFile, method = "curl", quiet = TRUE)
        message("Done", appendLF = TRUE)
        if(unzip == TRUE)
            {
            message("Unzipping data file - please wait ... ", appendLF = FALSE)
            unzip(vDestFile, overwrite = TRUE)
            }
        message("Done", appendLF = TRUE)
        }
    }
```

The above defined function is used to download the activity.csv data from 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv


```r
dataLoader("pml-training.csv", "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", unzip = FALSE)
df.rawData <- read.csv("pml-training.csv", header = TRUE, stringsAsFactors = TRUE, na.strings = "NA")

# Throw away the dataLoader function - it's not needed any longer
rm("dataLoader")
```

Load the libraries required for this anaysis


```r
require(caret)
require(randomForest)
```

## Exploratory Analysis
An inspection of the data frame dimensions indicates that the available data contains nearly 20,000 rows of 160 columns. This is a substantial number of predictors with which to build a model, so some reduction techniques will be applied to select a smaller set of predictors.

The various types and numbers of column classes are as follows.


```r
with(df.rawData, {
    print(dim(df.rawData))
    for(v.type in unique(unname(sapply(df.rawData, class)))) {
        print(paste(v.type, sum(sum(unname(sapply(df.rawData, class)) %in% v.type))))
    }
})
```

```
## [1] 19622   160
## [1] "integer 35"
## [1] "factor 37"
## [1] "numeric 88"
```

## Cross Validation Technique
The cross validation technique selected is to split the available data into train and test sets. A 60/40 split is used. K-fold cross validation will be applied to the training dataset and the actual out-of-sample error will be estimated with the test set.


```r
set.seed(1)
with(df.rawData, {
    v.trainingIndices <- createDataPartition(y = df.rawData$classe, p = 0.6, list = FALSE)
    df.trainData <<- df.rawData[v.trainingIndices,]
    df.testData <<- df.rawData[-v.trainingIndices,]
})

# Throw away the original raw data frame - it's not needed any longer
rm("df.rawData")
```

## Covariate Selection
A visual inspection of the **train** data set with head and str commands (outputs not shown for brevity) reveal some areas of concern in the data which will impair the effectiveness of any training algorithm.

- There are several columns where the datum type is clearly *not* useful for training purposes - e.g. 'user_name' or 'raw_timestamp_part_n'
- There are several columns with a high number of NA values
- There are columns with a high number of '#DIV/0!' or empty values

Other areas of concern are

- There *may* be columns with low overall variance in the data - limiting their usefulness for training purposes
- There *may* be columns that are highly correlated to each other -  rendering some columns redundant for training purposes

The reader might question the wisdom of removing the raw_timestamp_part_n columns. The model is ultimately intended to evaluate the efficacy of a given action performed by a subject. The very fact that there are many time-adjacent data samples available each second, for each subject, would lend weight to the idea that analysing a time-sequenced group of samples could be of service in building an accurate and useful model. However the available data is classified on **every available row** - lending weight to the idea that single point-in-time samples are also of value when training.

Additionally, since the final test set contains *no* time adjacent samples and contains only single point-in-time measurements, the potential benefits of time sequence based analysis are rendered null and void. Am I cheating by having an insight into the final testing methodology and using this to guide my covariate selection process ? - I'll leave that up to you.


```r
# Clean the column names to enforce tidy names under R's column naming conventions
names(df.trainData) <- make.names(names(df.trainData), unique = TRUE)
names(df.testData) <- make.names(names(df.testData), unique = TRUE)
```



```r
# Drop this columns where 'common sense' indicates that the datum type is simply not useful for training a model.
# Train and test sets receive the same treatment
with (df.trainData, {
    v.columnNamesToDrop <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
    print(paste("The following training set column will be dropped : ", v.columnNamesToDrop))
    df.trainData <<- df.trainData[,!(names(df.trainData) %in% v.columnNamesToDrop)]
    df.testData <<- df.testData[,!(names(df.testData) %in% v.columnNamesToDrop)]
})
```

```
## [1] "The following training set column will be dropped :  X"                   
## [2] "The following training set column will be dropped :  user_name"           
## [3] "The following training set column will be dropped :  raw_timestamp_part_1"
## [4] "The following training set column will be dropped :  raw_timestamp_part_2"
## [5] "The following training set column will be dropped :  cvtd_timestamp"      
## [6] "The following training set column will be dropped :  new_window"          
## [7] "The following training set column will be dropped :  num_window"
```

```r
# Those columns from the *train* set where more than 50% of values are NA. The columns selected here are also removed from the test set
with (df.trainData, {
    v.naThreshold <- dim(df.trainData)[1] * 0.5    
    v.naCounts <- sapply(df.trainData, function(column) {
        sum(length(which(is.na(column))))
    })
    v.columnNamesToDrop <- names(which(v.naCounts > v.naThreshold))
    print(paste("The following training set column will be dropped : ", v.columnNamesToDrop))
    df.trainData <<- df.trainData[,!(names(df.trainData) %in% v.columnNamesToDrop)]
    df.testData <<- df.testData[,!(names(df.testData) %in% v.columnNamesToDrop)]
})
```

```
##  [1] "The following training set column will be dropped :  max_roll_belt"           
##  [2] "The following training set column will be dropped :  max_picth_belt"          
##  [3] "The following training set column will be dropped :  min_roll_belt"           
##  [4] "The following training set column will be dropped :  min_pitch_belt"          
##  [5] "The following training set column will be dropped :  amplitude_roll_belt"     
##  [6] "The following training set column will be dropped :  amplitude_pitch_belt"    
##  [7] "The following training set column will be dropped :  var_total_accel_belt"    
##  [8] "The following training set column will be dropped :  avg_roll_belt"           
##  [9] "The following training set column will be dropped :  stddev_roll_belt"        
## [10] "The following training set column will be dropped :  var_roll_belt"           
## [11] "The following training set column will be dropped :  avg_pitch_belt"          
## [12] "The following training set column will be dropped :  stddev_pitch_belt"       
## [13] "The following training set column will be dropped :  var_pitch_belt"          
## [14] "The following training set column will be dropped :  avg_yaw_belt"            
## [15] "The following training set column will be dropped :  stddev_yaw_belt"         
## [16] "The following training set column will be dropped :  var_yaw_belt"            
## [17] "The following training set column will be dropped :  var_accel_arm"           
## [18] "The following training set column will be dropped :  avg_roll_arm"            
## [19] "The following training set column will be dropped :  stddev_roll_arm"         
## [20] "The following training set column will be dropped :  var_roll_arm"            
## [21] "The following training set column will be dropped :  avg_pitch_arm"           
## [22] "The following training set column will be dropped :  stddev_pitch_arm"        
## [23] "The following training set column will be dropped :  var_pitch_arm"           
## [24] "The following training set column will be dropped :  avg_yaw_arm"             
## [25] "The following training set column will be dropped :  stddev_yaw_arm"          
## [26] "The following training set column will be dropped :  var_yaw_arm"             
## [27] "The following training set column will be dropped :  max_roll_arm"            
## [28] "The following training set column will be dropped :  max_picth_arm"           
## [29] "The following training set column will be dropped :  max_yaw_arm"             
## [30] "The following training set column will be dropped :  min_roll_arm"            
## [31] "The following training set column will be dropped :  min_pitch_arm"           
## [32] "The following training set column will be dropped :  min_yaw_arm"             
## [33] "The following training set column will be dropped :  amplitude_roll_arm"      
## [34] "The following training set column will be dropped :  amplitude_pitch_arm"     
## [35] "The following training set column will be dropped :  amplitude_yaw_arm"       
## [36] "The following training set column will be dropped :  max_roll_dumbbell"       
## [37] "The following training set column will be dropped :  max_picth_dumbbell"      
## [38] "The following training set column will be dropped :  min_roll_dumbbell"       
## [39] "The following training set column will be dropped :  min_pitch_dumbbell"      
## [40] "The following training set column will be dropped :  amplitude_roll_dumbbell" 
## [41] "The following training set column will be dropped :  amplitude_pitch_dumbbell"
## [42] "The following training set column will be dropped :  var_accel_dumbbell"      
## [43] "The following training set column will be dropped :  avg_roll_dumbbell"       
## [44] "The following training set column will be dropped :  stddev_roll_dumbbell"    
## [45] "The following training set column will be dropped :  var_roll_dumbbell"       
## [46] "The following training set column will be dropped :  avg_pitch_dumbbell"      
## [47] "The following training set column will be dropped :  stddev_pitch_dumbbell"   
## [48] "The following training set column will be dropped :  var_pitch_dumbbell"      
## [49] "The following training set column will be dropped :  avg_yaw_dumbbell"        
## [50] "The following training set column will be dropped :  stddev_yaw_dumbbell"     
## [51] "The following training set column will be dropped :  var_yaw_dumbbell"        
## [52] "The following training set column will be dropped :  max_roll_forearm"        
## [53] "The following training set column will be dropped :  max_picth_forearm"       
## [54] "The following training set column will be dropped :  min_roll_forearm"        
## [55] "The following training set column will be dropped :  min_pitch_forearm"       
## [56] "The following training set column will be dropped :  amplitude_roll_forearm"  
## [57] "The following training set column will be dropped :  amplitude_pitch_forearm" 
## [58] "The following training set column will be dropped :  var_accel_forearm"       
## [59] "The following training set column will be dropped :  avg_roll_forearm"        
## [60] "The following training set column will be dropped :  stddev_roll_forearm"     
## [61] "The following training set column will be dropped :  var_roll_forearm"        
## [62] "The following training set column will be dropped :  avg_pitch_forearm"       
## [63] "The following training set column will be dropped :  stddev_pitch_forearm"    
## [64] "The following training set column will be dropped :  var_pitch_forearm"       
## [65] "The following training set column will be dropped :  avg_yaw_forearm"         
## [66] "The following training set column will be dropped :  stddev_yaw_forearm"      
## [67] "The following training set column will be dropped :  var_yaw_forearm"
```

```r
# Those columns from the *train*  set where more than 50% of values are empty or contain '#DIV/0!'
with (df.trainData, {
    v.naThreshold <- dim(df.trainData[1]) * 0.5
    v.naCounts <- sapply(df.trainData, function(column) {
        length(grep("^$|#DIV/0!", column))
    })
    v.columnNamesToDrop <- names(which(v.naCounts > v.naThreshold))
    print(paste("The following training set column will be dropped : ", v.columnNamesToDrop))
    df.trainData <<- df.trainData[,!(names(df.trainData) %in% v.columnNamesToDrop)]
    df.testData <<- df.testData[,!(names(df.testData) %in% v.columnNamesToDrop)]
})
```

```
##  [1] "The following training set column will be dropped :  kurtosis_roll_belt"     
##  [2] "The following training set column will be dropped :  kurtosis_picth_belt"    
##  [3] "The following training set column will be dropped :  kurtosis_yaw_belt"      
##  [4] "The following training set column will be dropped :  skewness_roll_belt"     
##  [5] "The following training set column will be dropped :  skewness_roll_belt.1"   
##  [6] "The following training set column will be dropped :  skewness_yaw_belt"      
##  [7] "The following training set column will be dropped :  max_yaw_belt"           
##  [8] "The following training set column will be dropped :  min_yaw_belt"           
##  [9] "The following training set column will be dropped :  amplitude_yaw_belt"     
## [10] "The following training set column will be dropped :  kurtosis_roll_arm"      
## [11] "The following training set column will be dropped :  kurtosis_picth_arm"     
## [12] "The following training set column will be dropped :  kurtosis_yaw_arm"       
## [13] "The following training set column will be dropped :  skewness_roll_arm"      
## [14] "The following training set column will be dropped :  skewness_pitch_arm"     
## [15] "The following training set column will be dropped :  skewness_yaw_arm"       
## [16] "The following training set column will be dropped :  kurtosis_roll_dumbbell" 
## [17] "The following training set column will be dropped :  kurtosis_picth_dumbbell"
## [18] "The following training set column will be dropped :  kurtosis_yaw_dumbbell"  
## [19] "The following training set column will be dropped :  skewness_roll_dumbbell" 
## [20] "The following training set column will be dropped :  skewness_pitch_dumbbell"
## [21] "The following training set column will be dropped :  skewness_yaw_dumbbell"  
## [22] "The following training set column will be dropped :  max_yaw_dumbbell"       
## [23] "The following training set column will be dropped :  min_yaw_dumbbell"       
## [24] "The following training set column will be dropped :  amplitude_yaw_dumbbell" 
## [25] "The following training set column will be dropped :  kurtosis_roll_forearm"  
## [26] "The following training set column will be dropped :  kurtosis_picth_forearm" 
## [27] "The following training set column will be dropped :  kurtosis_yaw_forearm"   
## [28] "The following training set column will be dropped :  skewness_roll_forearm"  
## [29] "The following training set column will be dropped :  skewness_pitch_forearm" 
## [30] "The following training set column will be dropped :  skewness_yaw_forearm"   
## [31] "The following training set column will be dropped :  max_yaw_forearm"        
## [32] "The following training set column will be dropped :  min_yaw_forearm"        
## [33] "The following training set column will be dropped :  amplitude_yaw_forearm"
```

Those columns in the **train** data set that exhibit a high correlation with each other indicate that some columns may be removed without compromising the usefulness of the training data for modelling purposes. The goal is to capture as much information as possible in as few columns as possible. The code below will determine the correlation between all columns of the training data and save this in a matrix. Printing those column / row indices of the matrix entries of interest will yield a set of row indices from the training data set which may be removed. An (arbitrary) threshold of 0.8 has been chosen to indicate 'high' correlation. 


```r
# Those columns from the training data set which correlation is high (> 0.8)
with(df.trainData, {
    # Build the matrix of relative correlations between all columns (except the 'classe' column)
    correlationMatrix <- abs(cor(df.trainData[, -dim(df.trainData)[2]]))
    # Zero the diagonal 'self correlation' entries (always 1)
    diag(correlationMatrix) <- 0
    # Capture the matrix indices in a vector and split into colum and row sets
    # We only need remove *one* of these sets, since the column / row values are transposed for the other sets of co-ordinates
    v.indices <- as.numeric(unlist(strsplit(as.character(which(correlationMatrix > 0.8, arr.ind = T)), '\\s+')))
    v.columnIndicesToDrop <- sort(unique(v.indices[1:(length(v.indices) / 2)]), decreasing = FALSE)
    print(paste("The following training set column will be dropped : ", names(df.trainData)[v.columnIndicesToDrop]))
    # Get the names of the train (and test) columns to drop
    v.columnNamesToDrop <- names(df.trainData)[v.columnIndicesToDrop]
    df.trainData <<- df.trainData[,!(names(df.trainData) %in% v.columnNamesToDrop)]
    df.testData <<- df.testData[,!(names(df.testData) %in% v.columnNamesToDrop)]
})
```

```
##  [1] "The following training set column will be dropped :  roll_belt"       
##  [2] "The following training set column will be dropped :  pitch_belt"      
##  [3] "The following training set column will be dropped :  yaw_belt"        
##  [4] "The following training set column will be dropped :  total_accel_belt"
##  [5] "The following training set column will be dropped :  accel_belt_x"    
##  [6] "The following training set column will be dropped :  accel_belt_y"    
##  [7] "The following training set column will be dropped :  accel_belt_z"    
##  [8] "The following training set column will be dropped :  magnet_belt_x"   
##  [9] "The following training set column will be dropped :  gyros_arm_x"     
## [10] "The following training set column will be dropped :  gyros_arm_y"     
## [11] "The following training set column will be dropped :  accel_arm_x"     
## [12] "The following training set column will be dropped :  magnet_arm_x"    
## [13] "The following training set column will be dropped :  magnet_arm_y"    
## [14] "The following training set column will be dropped :  magnet_arm_z"    
## [15] "The following training set column will be dropped :  pitch_dumbbell"  
## [16] "The following training set column will be dropped :  yaw_dumbbell"    
## [17] "The following training set column will be dropped :  gyros_dumbbell_x"
## [18] "The following training set column will be dropped :  gyros_dumbbell_z"
## [19] "The following training set column will be dropped :  accel_dumbbell_x"
## [20] "The following training set column will be dropped :  accel_dumbbell_z"
## [21] "The following training set column will be dropped :  gyros_forearm_y" 
## [22] "The following training set column will be dropped :  gyros_forearm_z"
```

Finally those columns of the **train** data sets which exhibit minimal variance many be excluded - since these contain less useful information for model building than a high variance column. The variance of each of the training set columns is computed and the 20th percentile is calculated.  Columns whose total variance is less than the 20th percentile are deemed not variable enough and are removed from the training and test sets.


```r
# Those columns from the *train* data set where there is minimal variance in the data
with(df.trainData, {
    v.variance <- vector(mode = "numeric", length = 0)
    # v.threshold <- vector(mode = "numeric", length = 0)
    for(index in 1:dim(df.trainData)[2] - 1) {
      v.variance <- c(v.variance, var(df.trainData[,index]))          
  }
  v.threshold <- (quantile(v.variance, 0.2))
  v.columnNamesToDrop <- names(df.trainData[which(v.variance < v.threshold)])
  print(paste("The following training set column will be dropped : ", v.columnNamesToDrop))
  df.trainData <<- df.trainData[,!(names(df.trainData) %in% v.columnNamesToDrop)]
  df.testData <<- df.testData[,!(names(df.testData) %in% v.columnNamesToDrop)]
})
```

```
## [1] "The following training set column will be dropped :  gyros_belt_x"    
## [2] "The following training set column will be dropped :  gyros_belt_y"    
## [3] "The following training set column will be dropped :  gyros_belt_z"    
## [4] "The following training set column will be dropped :  gyros_arm_z"     
## [5] "The following training set column will be dropped :  gyros_dumbbell_y"
## [6] "The following training set column will be dropped :  gyros_forearm_x"
```

Now the final set of columns which are selected as the basis for building the model are available - these are :


```r
print(paste("The following training set column will be included in the model : ", names(df.trainData)))
```

```
##  [1] "The following training set column will be included in the model :  magnet_belt_y"       
##  [2] "The following training set column will be included in the model :  magnet_belt_z"       
##  [3] "The following training set column will be included in the model :  roll_arm"            
##  [4] "The following training set column will be included in the model :  pitch_arm"           
##  [5] "The following training set column will be included in the model :  yaw_arm"             
##  [6] "The following training set column will be included in the model :  total_accel_arm"     
##  [7] "The following training set column will be included in the model :  accel_arm_y"         
##  [8] "The following training set column will be included in the model :  accel_arm_z"         
##  [9] "The following training set column will be included in the model :  roll_dumbbell"       
## [10] "The following training set column will be included in the model :  total_accel_dumbbell"
## [11] "The following training set column will be included in the model :  accel_dumbbell_y"    
## [12] "The following training set column will be included in the model :  magnet_dumbbell_x"   
## [13] "The following training set column will be included in the model :  magnet_dumbbell_y"   
## [14] "The following training set column will be included in the model :  magnet_dumbbell_z"   
## [15] "The following training set column will be included in the model :  roll_forearm"        
## [16] "The following training set column will be included in the model :  pitch_forearm"       
## [17] "The following training set column will be included in the model :  yaw_forearm"         
## [18] "The following training set column will be included in the model :  total_accel_forearm" 
## [19] "The following training set column will be included in the model :  accel_forearm_x"     
## [20] "The following training set column will be included in the model :  accel_forearm_y"     
## [21] "The following training set column will be included in the model :  accel_forearm_z"     
## [22] "The following training set column will be included in the model :  magnet_forearm_x"    
## [23] "The following training set column will be included in the model :  magnet_forearm_y"    
## [24] "The following training set column will be included in the model :  magnet_forearm_z"    
## [25] "The following training set column will be included in the model :  classe"
```

## Missing Values Imputation
The data set which will be used to generate the model is sufficiently clean that **no** missing values need be imputed.
A check of the total number of NA values in the training set confirms this.


```r
sum(is.na(df.trainData))
```

```
## [1] 0
```

## Model Training
The technique of repeated K-fold cross validation has been selected for this exercise.  A value of 3 has been arbitrarily chosen for K. For the actual model building a random forest technique has been chosen.


```r
trainOptions <- trainControl(method = "repeatedcv", number = 3, repeats = 3, verboseIter = TRUE)
```


```r
set.seed(1)
# model <- train(classe ~ ., data = df.trainData, method = "rf", trControl = trainOptions)
```

## Results

A confusion matrix of the model applied to the test data gives an estimate of 97.8%  for the out of sample accuracy of the final model


```r
confusionMatrix(df.testData$classe, predict(model, df.testData))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    3    2    1    0
##          B   32 1464   22    0    0
##          C    0   16 1350    2    0
##          D    0    1   51 1230    4
##          E    0    8    8   19 1407
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9785         
##                  95% CI : (0.975, 0.9816)
##     No Information Rate : 0.2878         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9727         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9858   0.9812   0.9421   0.9824   0.9972
## Specificity            0.9989   0.9915   0.9972   0.9915   0.9946
## Pos Pred Value         0.9973   0.9644   0.9868   0.9565   0.9757
## Neg Pred Value         0.9943   0.9956   0.9872   0.9966   0.9994
## Prevalence             0.2878   0.1902   0.1826   0.1596   0.1798
## Detection Rate         0.2837   0.1866   0.1721   0.1568   0.1793
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9924   0.9864   0.9696   0.9870   0.9959
```

The output of 'varImp' shows the relative importance of the variables used in the model


```r
varImp(model)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 24)
## 
##                      Overall
## magnet_dumbbell_z     100.00
## magnet_dumbbell_y      89.49
## pitch_forearm          81.55
## magnet_belt_y          79.31
## magnet_belt_z          79.25
## magnet_dumbbell_x      73.61
## roll_forearm           68.36
## roll_dumbbell          56.43
## accel_dumbbell_y       54.79
## roll_arm               44.33
## yaw_arm                35.62
## total_accel_dumbbell   35.28
## accel_forearm_x        32.98
## magnet_forearm_z       29.79
## accel_forearm_z        27.84
## magnet_forearm_y       22.02
## magnet_forearm_x       21.93
## accel_arm_y            20.30
## pitch_arm              19.92
## yaw_forearm            13.00
```

The finalModel is shown for reference purposes


```r
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 2.01%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3339    7    1    0    1 0.002688172
## B   51 2198   20    4    6 0.035541904
## C    0   31 2016    5    2 0.018500487
## D    0    0   72 1856    2 0.038341969
## E    2    7    2   24 2130 0.016166282
```


## Acknowledgements

The author gratefully acknowlegdes the use of the 'Weight Lifting Exercise' dataset
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3gz0Re6YS
