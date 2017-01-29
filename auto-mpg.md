auto-mpg documentation
================
Kirk Imamura
January 29, 2017

``` r
# Making a prediction on the Auto-MPG dataset

# This dataset has 398 observations and 8 attirbutes plus the label.  The label is the expected outcome, which is mpg of an automobile when given the values of the given attributes.

# Attribute 1: mpg (expected outcome)
# Attribute 2: cylinders
# Attribute 3: displacement
# Attribute 4: horsepower
# Attribute 5: weight
# Attribute 6: acceleration
# Attribute 7: model modelYear
# Attribute 8: origin
# Attribute 9: car name

# Get dataset from UCI machine-learning repository
autos <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", header=FALSE, sep="", as.is=TRUE)

head(autos, 5)
```

    ##   V1 V2  V3    V4   V5   V6 V7 V8                        V9
    ## 1 18  8 307 130.0 3504 12.0 70  1 chevrolet chevelle malibu
    ## 2 15  8 350 165.0 3693 11.5 70  1         buick skylark 320
    ## 3 18  8 318 150.0 3436 11.0 70  1        plymouth satellite
    ## 4 16  8 304 150.0 3433 12.0 70  1             amc rebel sst
    ## 5 17  8 302 140.0 3449 10.5 70  1               ford torino

``` r
tail(autos, 5)
```

    ##     V1 V2  V3    V4   V5   V6 V7 V8              V9
    ## 394 27  4 140 86.00 2790 15.6 82  1 ford mustang gl
    ## 395 44  4  97 52.00 2130 24.6 82  2       vw pickup
    ## 396 32  4 135 84.00 2295 11.6 82  1   dodge rampage
    ## 397 28  4 120 79.00 2625 18.6 82  1     ford ranger
    ## 398 31  4 119 82.00 2720 19.4 82  1      chevy s-10

``` r
summary(autos)
```

    ##        V1              V2              V3             V4           
    ##  Min.   : 9.00   Min.   :3.000   Min.   : 68.0   Length:398        
    ##  1st Qu.:17.50   1st Qu.:4.000   1st Qu.:104.2   Class :character  
    ##  Median :23.00   Median :4.000   Median :148.5   Mode  :character  
    ##  Mean   :23.51   Mean   :5.455   Mean   :193.4                     
    ##  3rd Qu.:29.00   3rd Qu.:8.000   3rd Qu.:262.0                     
    ##  Max.   :46.60   Max.   :8.000   Max.   :455.0                     
    ##        V5             V6              V7              V8       
    ##  Min.   :1613   Min.   : 8.00   Min.   :70.00   Min.   :1.000  
    ##  1st Qu.:2224   1st Qu.:13.82   1st Qu.:73.00   1st Qu.:1.000  
    ##  Median :2804   Median :15.50   Median :76.00   Median :1.000  
    ##  Mean   :2970   Mean   :15.57   Mean   :76.01   Mean   :1.573  
    ##  3rd Qu.:3608   3rd Qu.:17.18   3rd Qu.:79.00   3rd Qu.:2.000  
    ##  Max.   :5140   Max.   :24.80   Max.   :82.00   Max.   :3.000  
    ##       V9           
    ##  Length:398        
    ##  Class :character  
    ##  Mode  :character  
    ##                    
    ##                    
    ## 

``` r
str(autos)
```

    ## 'data.frame':    398 obs. of  9 variables:
    ##  $ V1: num  18 15 18 16 17 15 14 14 14 15 ...
    ##  $ V2: int  8 8 8 8 8 8 8 8 8 8 ...
    ##  $ V3: num  307 350 318 304 302 429 454 440 455 390 ...
    ##  $ V4: chr  "130.0" "165.0" "150.0" "150.0" ...
    ##  $ V5: num  3504 3693 3436 3433 3449 ...
    ##  $ V6: num  12 11.5 11 12 10.5 10 9 8.5 10 8.5 ...
    ##  $ V7: int  70 70 70 70 70 70 70 70 70 70 ...
    ##  $ V8: int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ V9: chr  "chevrolet chevelle malibu" "buick skylark 320" "plymouth satellite" "amc rebel sst" ...

``` r
# Preparing the data for the modeling process

# Rename columns type
colnames(autos) <- c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "modelYear", "origin", "carName")

# Change data type of horsepower to numeric
autos$horsepower <- as.numeric(autos$horsepower)
```

    ## Warning: NAs introduced by coercion

``` r
# Replace missing values with the mean of the entire column
autos$horsepower[is.na(autos$horsepower)] <- mean(autos$horsepower, na.rm=TRUE)

# Change the following attributes with discrete values to factors
autos$origin <- factor (autos$origin)
autos$modelYear <- factor (autos$modelYear)
autos$cylinders <- factor (autos$cylinders)

# Discard car name attribute from the data frame
autos$carName <- NULL

str(autos)
```

    ## 'data.frame':    398 obs. of  8 variables:
    ##  $ mpg         : num  18 15 18 16 17 15 14 14 14 15 ...
    ##  $ cylinders   : Factor w/ 5 levels "3","4","5","6",..: 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ displacement: num  307 350 318 304 302 429 454 440 455 390 ...
    ##  $ horsepower  : num  130 165 150 150 140 198 220 215 225 190 ...
    ##  $ weight      : num  3504 3693 3436 3433 3449 ...
    ##  $ acceleration: num  12 11.5 11 12 10.5 10 9 8.5 10 8.5 ...
    ##  $ modelYear   : Factor w/ 13 levels "70","71","72",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ origin      : Factor w/ 3 levels "1","2","3": 1 1 1 1 1 1 1 1 1 1 ...

``` r
# Split autos dataset into two sets
# One for training the model
# One for testing the model
# Using 70/30 split between training and testing datasets
trainSize <- round(nrow(autos) * 0.7)
testSize <- nrow(autos) -  trainSize

trainSize
```

    ## [1] 279

``` r
testSize
```

    ## [1] 119

``` r
# Create training set from a random selection of the entire dataset
set.seed(123)
training_indices <- sample(seq_len(nrow(autos)), size=trainSize)
trainSet <- autos[training_indices, ]
testSet <- autos[-training_indices, ]

# Create a linear regression model that uses mpg as a resposne variable and all other variables as predictor variables
model <- lm(formula=trainSet$mpg ~., data=trainSet)

summary(model)
```

    ## 
    ## Call:
    ## lm(formula = trainSet$mpg ~ ., data = trainSet)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8.2335 -1.8353 -0.1777  1.4944 10.3628 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  32.3982324  2.7869918  11.625  < 2e-16 ***
    ## cylinders4    7.5566601  1.8268522   4.136 4.79e-05 ***
    ## cylinders5    5.1525411  2.8756087   1.792 0.074345 .  
    ## cylinders6    5.2780827  2.0666565   2.554 0.011232 *  
    ## cylinders8    8.7074339  2.4367101   3.573 0.000421 ***
    ## displacement  0.0108861  0.0088901   1.225 0.221885    
    ## horsepower   -0.0398462  0.0151571  -2.629 0.009085 ** 
    ## weight       -0.0060124  0.0007913  -7.598 5.65e-13 ***
    ## acceleration  0.0306668  0.1022069   0.300 0.764385    
    ## modelYear71   0.8418349  1.0300922   0.817 0.414550    
    ## modelYear72  -0.8234493  1.0689036  -0.770 0.441792    
    ## modelYear73  -0.8431837  0.9786156  -0.862 0.389709    
    ## modelYear74   0.9293372  1.0884433   0.854 0.394002    
    ## modelYear75   1.0591511  1.0711153   0.989 0.323680    
    ## modelYear76   0.6425808  1.0306412   0.623 0.533526    
    ## modelYear77   3.0892335  1.0909155   2.832 0.004997 ** 
    ## modelYear78   3.1158390  1.0086769   3.089 0.002229 ** 
    ## modelYear79   4.6223012  1.0579031   4.369 1.81e-05 ***
    ## modelYear80   8.3292306  1.1110222   7.497 1.07e-12 ***
    ## modelYear81   6.9562536  1.1318069   6.146 3.03e-09 ***
    ## modelYear82   7.1177850  1.0889953   6.536 3.40e-10 ***
    ## origin2       1.8747254  0.5958604   3.146 0.001849 ** 
    ## origin3       2.0368634  0.5949548   3.424 0.000720 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2.903 on 256 degrees of freedom
    ## Multiple R-squared:  0.8741, Adjusted R-squared:  0.8633 
    ## F-statistic: 80.82 on 22 and 256 DF,  p-value: < 2.2e-16

``` r
# With the model created, we can make predictions against it with the test data
predictions <- predict(model, testSet, interval="predict", level=.95)

head(predictions)
```

    ##        fit       lwr      upr
    ## 2 16.48993 10.530223 22.44964
    ## 4 18.16543 12.204615 24.12625
    ## 5 18.39992 12.402524 24.39732
    ## 6 12.09295  6.023341 18.16257
    ## 7 11.37966  5.186428 17.57289
    ## 8 11.66368  5.527497 17.79985

``` r
comparison <- cbind(testSet$mpg, predictions[,1])
colnames(comparison) <- c("actual", "predicted")

head(comparison)
```

    ##   actual predicted
    ## 2     15  16.48993
    ## 4     16  18.16543
    ## 5     17  18.39992
    ## 6     15  12.09295
    ## 7     14  11.37966
    ## 8     14  11.66368

``` r
summary(comparison)
```

    ##      actual        predicted     
    ##  Min.   :10.00   Min.   : 8.849  
    ##  1st Qu.:16.00   1st Qu.:17.070  
    ##  Median :21.50   Median :22.912  
    ##  Mean   :22.79   Mean   :23.048  
    ##  3rd Qu.:28.00   3rd Qu.:29.519  
    ##  Max.   :44.30   Max.   :37.643

``` r
# Examine the mean absolute percent error (mape) to measure the accuracy of our regression model
mape <- (sum(abs(comparison[,1]-comparison[,2]) / abs(comparison[,1]))/nrow(comparison))*100
mape
```

    ## [1] 10.93689

``` r
# view the results and errors in a table view
mapeTable <- cbind(comparison, abs(comparison[,1]-comparison[,2]) / comparison[,1]*100)
colnames(mapeTable)[3] <- "absolute percent error"

head(mapeTable)
```

    ##   actual predicted absolute percent error
    ## 2     15  16.48993               9.932889
    ## 4     16  18.16543              13.533952
    ## 5     17  18.39992               8.234840
    ## 6     15  12.09295              19.380309
    ## 7     14  11.37966              18.716708
    ## 8     14  11.66368              16.688031

``` r
sum(mapeTable[,3]) / nrow(comparison)
```

    ## [1] 10.93689

``` r
# Make new predictions with a list of 7 attributes
newPrediction <- predict(model, list(cylinders=factor(4), displacement=370, horsepower=150, weight=3904, acceleration=12, modelYear=factor(70), origin=factor(1)), interval="predict", level=.95)
newPrediction
```

    ##        fit     lwr      upr
    ## 1 14.90128 8.12795 21.67462
