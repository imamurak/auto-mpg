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
tail(autos, 5)
summary(autos)

str(autos)

# Preparing the data for the modeling process

# Rename columns type
colnames(autos) <- c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "modelYear", "origin", "carName")

# Change data type of horsepower to numeric
autos$horsepower <- as.numeric(autos$horsepower)

# Replace missing values with the mean of the entire column
autos$horsepower[is.na(autos$horsepower)] <- mean(autos$horsepower, na.rm=TRUE)

# Change the following attributes with discrete values to factors
autos$origin <- factor (autos$origin)
autos$modelYear <- factor (autos$modelYear)
autos$cylinders <- factor (autos$cylinders)

# Discard car name attribute from the data frame
autos$carName <- NULL

str(autos)

# Split autos dataset into two sets
# One for training the model
# One for testing the model
# Using 70/30 split between training and testing datasets
trainSize <- round(nrow(autos) * 0.7)
testSize <- nrow(autos) -  trainSize

trainSize
testSize

# Create training set from a random selection of the entire dataset
set.seed(123)
training_indices <- sample(seq_len(nrow(autos)), size=trainSize)
trainSet <- autos[training_indices, ]
testSet <- autos[-training_indices, ]

# Create a linear regression model that uses mpg as a resposne variable and all other variables as predictor variables
model <- lm(formula=trainSet$mpg ~., data=trainSet)

summary(model)

# With the model created, we can make predictions against it with the test data
predictions <- predict(model, testSet, interval="predict", level=.95)

head(predictions)

comparison <- cbind(testSet$mpg, predictions[,1])
colnames(comparison) <- c("actual", "predicted")

head(comparison)
summary(comparison)

# Examine the mean absolute percent error (mape) to measure the accuracy of our regression model
mape <- (sum(abs(comparison[,1]-comparison[,2]) / abs(comparison[,1]))/nrow(comparison))*100
mape

# view the results and errors in a table view
mapeTable <- cbind(comparison, abs(comparison[,1]-comparison[,2]) / comparison[,1]*100)
colnames(mapeTable)[3] <- "absolute percent error"

head(mapeTable)

sum(mapeTable[,3]) / nrow(comparison)

# Make new predictions with a list of 7 attributes
newPrediction <- predict(model, list(cylinders=factor(4), displacement=370, horsepower=150, weight=3904, acceleration=12, modelYear=factor(70), origin=factor(1)), interval="predict", level=.95)
newPrediction