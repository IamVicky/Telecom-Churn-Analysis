################################################################################################################
##                                                                                                            ## 
## Domain     :  Telecom                                                                                      ##
## Project    :  Telecom Churn Analysis                                                                       ##
## Data       :  Data for Customer Usage Pattern has been provided by UpX Academy in txt format               ## 
##               named as churn_data.txt.                                                                     ##
## Objective  :  1) To predict customer churn                                                                 ##  
##            :  2) Develop an algorithm to predict the churn score based on usage pattern along              ##
##            :     with accuracy of the model.                                                               ##
##                                                                                                            ##
################################################################################################################
##
## Install Required Packages
##
## install.packages("Amelia")
## install.packages("class")
## install.packages("gmodels")
## install.packages("GMD")
## install.packages("sjPlot")
## install.packages("survival") 
##
## Declare the installed needed packages so that we can use their functions in our R workbook
library(ggplot2)                          
library(dplyr)
library(Amelia)      #To use MISSMAP function to find out any missing Values
library(tree)        #To use tree modelling algorithms/functions
library(caret)       #To use Classification and regression Mehtods alogorithms/functions
library(class)
library(gmodels)     #To use validation the model performance of knn model
library(GMD)   
library(sjPlot)      #To plot the Elbow curve for the calculation of optimum value of k in knn
library(randomForest)
library(survival)    # To perform survival analysis 
library(e1071)       # To use svm method
##
##----------------------------------------------------------------------------------------------------------##
##  Step 1 - Data Collection                                                                                ##
##  In the this step we will be loading our input data telecom_churn_data from our system. As the provided  ##
##  input data doesn't contain header/column names, where as we have been provided separately the column    ##
##  details. Hence in the below code Columns <- we will be passing our column details while reading the data into ##
##  R dataframe, so that we can perform unctionality on the basis of the columns.                           ##                                                           
##                                                                                                          ##

columns <- c('State','Account_Len','Area','Ph_No.','Int_Plan','Vmail_Plan','messgs',
            'tot_day_mins','tot_day_calls','tot_day_chrgs','tot_evening_mins',
            'tot_evening_calls','tot_evening_chrgs','tot_ngt_mins','tot_ngt_calls',
            'tot_ngt_chrgs','tot_int_mins','tot_int_calls','tot_int_chrgs',
            'cust_calls_made','churn_status')
str(columns)

churn.data <- read.table('telecom_churn_data.txt',sep=',',col.names = columns)

##----------------------------------------------------------------------------------------------------------##
## Step 2 - Data Exploration & Data Preparation.                                                                      ##
## In this Step we will  :                                                                                  ##
##        - Perform basic analysis to understand the spread,dimensions & volume of the input data           ## 
##        - Will use various EDA tools like Pie chart, Box plot and Bar charts to identify the patterns &   ##
##          relations among input variables                                                                 ##
##        - Will try to identify the variables which are impacting our predicting variable churn_status     ##
##        - Will find out the missing values and clean the data so that it can be feeded to ML algorithms   ##
##          in subsequent steps.                                                                            ##
##                                                                                                          ##  

head(churn.data,10)           
tail(churn.data,20)             
dim(churn.data)               # Columns = 21, Observation = 4617, Predicted Variable Column name- churn_status
str(churn.data)                
summary(churn.data)             

churn.eda <-churn.data           # Copying our input Dataframe into another df churn.eda so that any changes we do for
                                 # EDA is not impacted on the input dataset.

str(churn.eda$churn_status)

# Lets plot a Pie Chart to show the proportions percentage of Customers churned & Not Churned

tab <- as.data.frame(table(churn.eda$churn_status))
slices <- c(tab[1,2], tab[2,2]) 
lbls <- c("Churned-False", "Churned-True")
pct <- round(slices/sum(slices)*100,digits = 2)  # calculating % rounded to 2 digits
lbls <- paste(lbls, pct)                         # add percents to labels 
lbls <- paste(lbls,"%",sep="")                   # ad % to labels 
pie(slices,labels = lbls, col=rainbow(length(lbls)),angle = 90,
    main="Percentage of Customer Churned")

# Lets plot varios Box plots to understand the various attributes of input variables for Churned and not Churned customers

# Boxplot to see the pattern of Total Minutes (Day/Evening/Night/International) for Churned / Not Churned Customers
boxplot(churn.eda$tot_day_mins ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Day mins")
boxplot(churn.eda$tot_evening_mins ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Evening mins")
boxplot(churn.eda$tot_ngt_mins ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Night mins")
boxplot(churn.eda$tot_int_mins ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Int. mins")
# 
#  Observation from Above Plots 
#  - The Q1,Q3, IQR & the area of the box for Total Day mins,Total Evening mins & Total International mins            
#     for Churned Customer is higher than Not Churned Customers.Hence, usage pattern of Chruned customers is high         
#    as compared to Not Churned.  
#  - There is no significant difference reg the usuage of Total Night Calls for Churned and Not Churned Customers.  
#  - Outliers are present in all the cases especially significant in Total Int mins, Total Evening Mins & Total Night mins
#               

# Boxplot to see the pattern of Total Cals (Day/Evening/Night/International) for Churned / Not Churned Customers

boxplot(churn.eda$tot_day_calls ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Day Calls")      
boxplot(churn.eda$tot_evening_calls ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Evening Calls")
boxplot(churn.eda$tot_ngt_calls ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Night Calls")
boxplot(churn.eda$tot_int_calls~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Int. Calls")   

#  Observation from Above Plots 
#  - As such no significant pattern difference for Churned and Not Churned Customers across the diff scenarios except
#    Total Night Calls where spread of IQR for Chruned Customers is relatively higher. 
#  - The Q1, Q3 and upper whiskers reg Total International Calls for Not Churned Customers is relatively higher than
#    the Churned Customers. So, can we say that Customers Churned, call less relatively but the duration of 
#    their calls are relatively high (mins used from the previous box plot)??   
#  - Outliers are present in all the cases are significantly high.
#               

# Boxplot to see the pattern of Total Charges (Day/Evening/Night/International) for Churned / Not Churned Customers

boxplot(churn.eda$tot_day_chrgs ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Day Charges")
boxplot(churn.eda$tot_evening_chrgs ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Evening Charges")
boxplot(churn.eda$tot_ngt_chrgs ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Night Charges")
boxplot(churn.eda$tot_int_chrgs ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Total Int. Charges")

# 
#  Observation from Above Plots 
#  - The Q1,Q3, IQR & the area of the box for Total Day Charges for Churned Customer is significantly higher than 
#    Not Churned Customers.And relatively higher for Churned Customers for all other cases as well.
#  - Outliers are present in all the cases are significantly high.
#           

# Boxplot to see the pattern of Total Customer Service Calls made for Churned / Not Churned Customers

boxplot(churn.eda$cust_calls_made ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Customer Calls Made")
# 
#  Observation from Above Plot
#  One can see from the Box Plot that the spread for Customer Calls made by Chruned Customer is significantly
#  more than tha Not churned ones. It seems Customers going to be churned call Customer Service a lot with their 
#  issues reg service.

# Boxplot to see the pattern of their stay for Churned / Not Churned Customers

boxplot(churn.eda$Account_Len ~ churn.eda$churn_status, data = churn.eda, col = "red",
        xlab = "Customer Churned",ylab = "Length of the Service(in days)")

# 
#  Observation from Above Plot
#  One can see from the above Box plot for the Length of the Service, we are not able to find out as such 
#  anything meaningful.So, lets try to plot any other kind of plots rather than box plots for analysing the
#  behaviour of Length of the Service.


sapply(churn.eda, class)

# Lets plot varios Bar plots now to see if we can find more insights apart from what we found via Box plots.
# As in the input data the obervations for Customer churned are significantly higher than the customer not churned which
# is obvious, lets start with finding out the average value for the needed columns before plotting them.

churn.eda.true <- filter(churn.eda,churn.eda$churn_status==" True.")      # df with observations for Churned Customers 
churn.eda.false <- filter(churn.eda,churn.eda$churn_status==" False.")    # df with observations for Not-Churned Customers
dim(churn.eda.true)
str(churn.eda.true)
dim(churn.eda.false)
str(churn.eda.false)

churn.eda.inttrue <- filter(churn.eda.true,churn.eda.true$Int_Plan==" yes")
dim(churn.eda.inttrue)
churn.eda.intfalse <- filter(churn.eda.false,churn.eda.false$Int_Plan==" yes")

# Finding mean among the various columns like minutes, calls & charges for Day/Evening/Night seperataly for Churned and 
# Not churned customers.

# Mean calculation for Mins Columns for Chruned Customer.
tot_day_mins_true = mean(churn.eda.true$tot_day_mins)
tot_day_mins_true
tot_evening_mins_true = mean(churn.eda.true$tot_evening_mins)
tot_evening_mins_true
tot_ngt_mins_true = mean(churn.eda.true$tot_ngt_mins)
tot_ngt_mins_true
tot_int_mins_true = mean(churn.eda.inttrue$tot_int_mins)
tot_int_mins_true


# Mean calculation for Mins Columns for Not-Chruned Customers
tot_day_mins_false = mean(churn.eda.false$tot_day_mins)
tot_day_mins_false
tot_evening_mins_false = mean(churn.eda.false$tot_evening_mins)
tot_evening_mins_false
tot_ngt_mins_false = mean(churn.eda.false$tot_ngt_mins)
tot_ngt_mins_false
tot_int_mins_false = mean(churn.eda.intfalse$tot_int_mins)
tot_int_mins_false

# Storing the above calculated values of various Mins columns for both Churned and Not-churned in dataframe, so that we can 
# use the same to plot bars against each other.

data.mins <- structure(list(W=c(tot_day_mins_true,tot_day_mins_false),Q=c(tot_evening_mins_true,tot_evening_mins_false),
                       Y=c(tot_ngt_mins_true,tot_ngt_mins_false),Z=c(tot_int_mins_true,tot_int_mins_false)),
                       .Names = c("Day","Evening","Night","Int"),
                       class="data.frame",row.names=c(NA,-2L))
data.mins
# Plotting Bar plots.
colours <- c("red","blue")
barplot(as.matrix(data.mins), main="Customer's Avg Total Mins Usage",ylab = "Avg. Total Mins used",
        cex.lab = 1.2, cex.main = 1, beside=TRUE, col=colours,ylim = c(0,250),width = .5)

legend("topright", c("Churned->True","Churned->False"), cex=.9,
       bty="n", fill=colours)

#Like we calculated above similar caliculation is done for Mean calculation for Calls columns for Chruned Customer.

tot_day_calls_true = mean(churn.eda.true$tot_day_calls)
tot_day_calls_true
tot_evening_calls_true = mean(churn.eda.true$tot_evening_calls)
tot_evening_calls_true
tot_ngt_calls_true = mean(churn.eda.true$tot_ngt_calls)
tot_ngt_calls_true
tot_int_calls_true = mean(churn.eda.inttrue$tot_int_calls)
tot_int_calls_true

# Mean calculation for Calls Columns for Not-Chruned Customers
tot_day_calls_false = mean(churn.eda.false$tot_day_calls)
tot_day_calls_false
tot_evening_calls_false = mean(churn.eda.false$tot_evening_calls)
tot_evening_calls_false
tot_ngt_calls_false = mean(churn.eda.false$tot_ngt_calls)
tot_ngt_calls_false
tot_int_calls_false = mean(churn.eda.intfalse$tot_int_calls)
tot_int_calls_false

# Plotting Bar plots for Calls made for cutomers Chruned vs Not churned
data.calls <- structure(list(W=c(tot_day_calls_true,tot_day_calls_false),Q=c(tot_evening_calls_true,tot_evening_calls_false),
                            Y=c(tot_ngt_calls_true,tot_ngt_calls_false),Z=c(tot_int_calls_true,tot_int_calls_false)),
                       .Names = c("Day","Evening","Night","Int"),
                       class="data.frame",row.names=c(NA,-2L))
data.mins
colours <- c("red","blue")
barplot(as.matrix(data.calls), main="Avg Total Customer's Calls",ylab = "Avg. Total Calls",
        cex.lab = 1.2, cex.main = 1, beside=TRUE, col=colours,ylim = c(0,150),width = .5)
legend("topright", c("Churned->True","Churned->False"), cex=.9,
       bty="n", fill=colours)

#Like we calculated above similar caliculation is done for Mean calculation for Charges columns for Chruned Customer.
tot_day_chrgs_true = mean(churn.eda.true$tot_day_chrgs)
tot_day_chrgs_true
tot_evening_chrgs_true = mean(churn.eda.true$tot_evening_chrgs)
tot_evening_chrgs_true
tot_ngt_chrgs_true = mean(churn.eda.true$tot_ngt_chrgs)
tot_ngt_chrgs_true
tot_int_chrgs_true = mean(churn.eda.inttrue$tot_int_chrgs)
tot_int_chrgs_true

# Mean calculation for Charges Columns for Not-Chruned Customers
tot_day_chrgs_false = mean(churn.eda.false$tot_day_chrgs)
tot_day_chrgs_false
tot_evening_chrgs_false = mean(churn.eda.false$tot_evening_chrgs)
tot_evening_chrgs_false
tot_ngt_chrgs_false = mean(churn.eda.false$tot_ngt_chrgs)
tot_ngt_chrgs_false
tot_int_chrgs_false = mean(churn.eda.intfalse$tot_int_chrgs)
tot_int_chrgs_false

# Plotting Bar plots for Charges incurred for Chruned vs Not churned Customers
data.chrgs <- structure(list(W=c(tot_day_chrgs_true,tot_day_chrgs_false),Q=c(tot_evening_chrgs_true,tot_evening_chrgs_false),
                            Y=c(tot_ngt_chrgs_true,tot_ngt_chrgs_false),Z=c(tot_int_chrgs_true,tot_int_chrgs_false)),
                       .Names = c("Day","Evening","Night","Int"),
                       class="data.frame",row.names=c(NA,-2L))
data.chrgs
colours <- c("red","blue")
barplot(as.matrix(data.chrgs), main="Avg Total Charges of a Customer",ylab = "Avg Total Chrgs",
        cex.lab = 1.2, cex.main = 1, beside=TRUE, col=colours,ylim = c(0,40),width = .5)
legend("topright", c("Churned->True","Churned->False"), cex=.9,
       bty="n", fill=colours)

#Now lets find out if there are any Missing values in our dataset.
missmap(churn.eda)        # No missing values as the map shows no white cells which signifies for missing values

cor(churn.eda[7:20])

## Survival Analysis (Cox Proportional Hazard)

train_survival <- train
test_survival <- test
pred_survival <- pred

str(train_survival$churn_status)
set.seed(1000)
train_survival$churn_status = as.integer(train_survival$churn_status)
train_survival$survival <- Surv(train_survival$Account_Len, train_survival$churn_status == 2)
head(train_survival,5)
str(train_survival$survival)


results <- coxph(survival ~ Int_Plan + Vmail_Plan + messgs + tot_day_calls + tot_day_chrgs 
                 + tot_day_mins +  
                   + tot_int_calls + tot_int_chrgs + tot_ngt_chrgs  
                 + tot_int_mins + cust_calls_made, data = train_survival)

results
# From the above displayed results one can see that most statistically significant coefficients are for
# Int_Plan (yes) with co-efficient(z) 11.53, Cust_Calls_made  with co-efficient(z) 9.72. Further by
# analysing the exp(coef), we can interpret the magnitude of the effects, like for cust_calls_made
# value of 1.36 signifies that cust_calls_made users churn 1.36 times more (or 36 % ) than the baseline
# survival rate.

# Survival plot

plot(survfit(results), ylim = c(0,1), xlab = "Account length in Days", 
     ylab = "Percentage not churned")

# So, from the figure we can see that for the first 50-70 days we have no customer churn, that means we
# are able to retain all of our customers in the first 50-70 days. For next 50-70 days the percentage of
# not churned reduces almost from 100% to 80% approx. And decreases considerably in such a way that, 
# after 200 days of tenure the percentage of not churned comes down almost  to 40 %. That means by the 
# end of completing 200 days tenure, 60% of our customers are churned who joined us on day 1.

cox.zph(results)

# 
# Obervation from above EDA  
#  -  The usage pattern for Chruned Customers which mainly includes minutes(Day/Evening/Night/Int) and their charges incurred 
#     is higher/larger than the Not Churned Customers. 
#  -  Both Box and Bar plots are complementing each other with their findings.
#  -  The box plot for cutomer calls made gives us enough proofs that customers going to be churned call customer
#     care many times before getting churned.
#  -  With the help of Box Plot for Account Length, we were not able to identify any specific findings.
#  -  Lucky enough to have no missing values in the input dataset.
#  -  From Survival Analysis it is confirmed that Customer Calls Made & International Plan are most influential features. 

##----------------------------------------------------------------------------------------------------------##
##  Now lets create training and test Data sets. For this project although our objective is to predict churn 
##  customers, we are not provided with predict dataset.Hence here we will seggregate the input dataset
##  into 3 datasets i.e. train, test and pred in proportions 60,20 & 20% respectively. So, we can train our model
##  on train dataset, validate/test on test dataset and finally by improving the final performance we give
##  our final prediction on predict dataset.

set.seed(1000)
idx <- sample(seq(1, 3), size = nrow(churn.data), replace = TRUE, prob = c(.6, .2, .2))
train <- churn.data[idx == 1,]
test <-  churn.data[idx == 2,]
pred <-  churn.data[idx == 3,]
dim(train)
dim(test)
dim(pred)

##----------------------------------------------------------------------------------------------------------##
## Step 3 - Selection, Building, Training, Testing & Fine tuning/Improving performance of different ML models                                                                      ##
##  We will seggregate this step into majorly 3 further sub-steps.for each Selected ML Model/Algorithm
##   -> Building/Training Model on train dataset
##   -> Testing the Model on test dataset, Calculating various Model performances
##   -> Improving further model performance by various Finetuning techniques & predicting the final outcomes 
##      on pred dataset.
##
##---------------------------------------------------------------------------------------------------------
##
## 1st Model -- LOGISTIC REGRESSION
## 
## Building/Training Model on train dataset.
## Our dependant variable(variable to be predicted) in this model will be churn_status,further the independant
## variables on which dependent variables will depend can be as below :-
## tot_day_mins, tot_day_chrgs ,tot_evening_mins, tot_evening_chrgs,tot_ngt_mins
## tot_ngt_chrgs, tot_int_mins, tot_int_chrgs, cust_calls_made
## The selection of these variables are on the basis of our EDA findings were we found that for the above
## were relatively/tending to be high for Chruned Customers.

attach(train)
set.seed(1000)
logit.model = glm(churn_status ~ tot_day_mins + tot_day_chrgs + tot_evening_mins + 
                  tot_evening_chrgs + tot_ngt_mins + tot_ngt_chrgs + tot_int_mins + 
                  tot_int_chrgs + cust_calls_made, family = "binomial", data = train)
summary(logit.model)
coef(logit.model)

## From the summary of the logit model, we can see the significant codes are only there for 2 variables
## i.e. for Intercept and for Cust_calls_made. Identification of Cust_calls_made is as expected but it is
## missing to signify other variables which it should. Lets try o plot again by dropping few independent not so relevant 
## variables from the formula. Lets drop the minutes used and keep only the charges in the formula.

attach(train)
set.seed(1000)
logit.model = glm(churn_status ~ Account_Len + tot_day_chrgs + 
                    tot_evening_chrgs + tot_ngt_chrgs + 
                    tot_int_chrgs + cust_calls_made, family = "binomial", data = train)
summary(logit.model)
coef(logit.model)

## Dropping minutes columns seems to have a positive effect, we can see the significant codes have been
## populated for almost all the independent used input variables which is good for our model.Lets proceed now and
## test out model on the test dataset.

## Testing the Model on test dataset, predict & Calculating various Model performances
probs = predict(logit.model,test, type = "response")          # predict the logit.model on test dataset 
contrasts(test$churn_status)
logit.pred = rep (" False." ,length(test$churn_status))       # replacing all values of logit.pred with default as False
logit.pred[probs > 0.5] = " True."                            # Considering 0.5 is the threshold
t <- table(logit.pred, test$churn_status)                     # Confusion Matrix  
t
mean(logit.pred == test$churn_status)                         # Accuracy of model = 85.48
Specificity = 741 /(741 + 13)                                 # Specificity = 98.27
Specificity
Sensitivity = 13/(13 + 115)                                   # Sensitivity = 10.15 
Sensitivity                                                    

## Analysing Model Performance
## The predicton of above model on dataset set is with Acc 85.48, Specificity = 98.27 & Sensitivity of the 
## model is only 10.15 % which is very less. That means out of 128 True churned Customers we were able to 
## predict correctly for only 13 which is very very low.Lets try to improvise the selected model.

## Improvising the Model Performance
## Lets try to variate the .5 threshold value to .4 & .3 and then further check the model performance.

contrasts(test$churn_status)
logit.pred = rep (" False." ,length(test$churn_status))        # replacing all values of logit.pred with default as False
logit.pred[probs > 0.4] = " True."                             # Considering 0.4 is the threshold
t <- table(logit.pred, test$churn_status)                      # Confusion Matrix  
t
mean(logit.pred == test$churn_status)                          # Accuracy 84.58
Specificity = 725 / (725 + 29)                                 # Specificity 96.15    
Specificity
Sensitivity = 23 / (23 + 105)
Sensitivity                                                    # Sensitivity 17.96

contrasts(test$churn_status)
logit.pred = rep (" False." ,length(test$churn_status))        # replacing all values of logit.pred with default as False
logit.pred[probs > 0.3] = " True."                             # Considering 0.3 is the threshold
t <- table(logit.pred, test$churn_status)                      # Confusion Matrix  
t
mean(logit.pred == test$churn_status)                          # Accuracy 83.33
Specificity = 694 / (694 + 60)                                 # Specificity 92.04    
Specificity
Sensitivity = 41 / (41 + 87)                                   # Sensitivity 32.03
Sensitivity                                
str(probs)

## Seems the best out of above models we can select is the latest one with threshold value 0.3, although
## the Sensitivity is 32.03 % only but further our specificity is also decreased. It will not be fruitful
## to decrease the threshold value more because that will increase in our Fall Positive Ratio. 
## Lets try to use the latest model to predict the outcomes for pred dataset with threshold value of 0.3 and 
## identify the model performances
## 
set.seed(1000)
probs = predict(logit.model,pred, type = "response")          # predict the logit.model on test dataset 
contrasts(pred$churn_status)
logit.pred = rep (" False." ,length(pred$churn_status))       # replacing all values of logit.pred with default as False
logit.pred[probs > 0.3] = " True."                            # Considering 0.3 is the threshold
t <- table(logit.pred, pred$churn_status)                     # Confusion Matrix  
t
mean(logit.pred == pred$churn_status)                         # Accuracy of model = 84.21
Specificity = 756 /(756 + 67)                                 # Specificity = 91.85
Specificity
Sensitivity = 41/(41 + 86)                                    # Sensitivity = 32.28 
Sensitivity                   

## Conclusion:- It seems we were able to predict the outcomes with ACC 84.21, Specificity 91.85 & Sensiti
## -vity as 32.28. In business sense the model produced by Logistic regression in this particular case is 
## quite poor. Because of the fact business is able to predict only 32% customer correctly which are going
## to be churned. Additonally in this case False Positive Ratio is increasing, resulting in suggesting of incorrect
## customer as going to be chruned, where as in reality they are not going to be churned.
## Hence, it seems Logistic Regression cannot be used for predicting the churn status, let proceed with 
## further ML models.

## 2nd Model k-nearest neighbour
## Building/Training Model on train dataset.

set.seed(1000)
knn_train <- train[c(3,7:20)]     # As in knn the model takes into account only numerics(bec of Distance calculation) removing the factors
knn_test <- test[c(3,7:20)]       # Same as above for test dataset 
knn_pred <- pred[c(3,7:20)]       # Same as above for pred dataset
k <- sqrt(dim(knn_train)[1])      # Using k sqrt of number of observations    
k
dim(knn_train)
knn_model <- knn(train = knn_train, test = knn_test,cl = train$churn_status, k=53)
summary(knn_train)

## Testing the Model performance on test dataset.
p <- CrossTable(x = test$churn_status,y = knn_model,prop.chisq = FALSE )   # Performance of the model
t <- table(knn_model,test$churn_status)
t
mean(knn_model==test$churn_status)               # Accuracy 87.55
Specificity = 752/(752 + 2)                      # Specificity 99.73   
Specificity
Sensitivity = 17/(17 + 111)                      # Sensitivity 13.28
Sensitivity

## Improving further model performance as the above obained model performance is very low.
## Use Elbow method to find the optimum value of k and also normalise the data so that scale irregularities
## can be suppressed.

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
knn_train_n <- as.data.frame(lapply(knn_train, normalize))
knn_test_n <- as.data.frame(lapply(knn_test, normalize))
knn_pred_n <- as.data.frame(lapply(knn_pred, normalize))

sjc.elbow(knn_train_n)    # Seems to be the optimum value of k as 4, lets go ahead and use the same.

set.seed(1000)
knn_model <- knn(train = knn_train_n, test = knn_test_n,cl = train$churn_status, k=4)
c <- CrossTable(x = test$churn_status,y = knn_model,prop.chisq = TRUE )

table(knn_model,test$churn_status)               
mean(knn_model==test$churn_status)              # Accuracy 88.32
Specificity = 735/(735 + 19)                    # Specificity 97.48 
Specificity
Sensitivity = 44/(44 + 84)                      # Sensitivity 34.37
Sensitivity

# Now we can see from the above parameters that the performance of the model has been increased a lot 
# as compared to previous one after the use of optimum value of k=4.
# Lets try to use the latest model to predict the outcomes for pred dataset

set.seed(1000)
knn_model <- knn(train = knn_train_n, test = knn_pred_n,cl = train$churn_status, k=4)
table(knn_model,pred$churn_status)
mean(knn_model==pred$churn_status)              # Accuracy 89.26
Specificity = 805/(805 + 18)                    # Specificity 97.81   
Specificity
Sensitivity = 43/(43 + 84)                      # Sensitivity 33.85
Sensitivity

## Conclusion:-Further degradation of k results in having a negative impact on the model, further if one compares the model
## performance with Logistic Regression this model looks much better for all the calculated model performance parameters.
## But still stats from knn.model doesnt looks convincing w.r.t business, so lets proceed further with another
## ML algorithm.

## 3rd Model Decision Trees algorithm
## Building/Training Model on train dataset.

attach(train)
set.seed(1000)
tree_model <- tree(churn_status ~.-State-Area-Ph_No., data = train)
summary(tree_model)       # No of term nodes = 15, also provides the variables which were taken into consideration
                          # for creation of this tree model
plot(tree_model)
text(tree_model)

## Testing the Model performance on test dataset.
set.seed(1000)
predict_tree <- predict(tree_model, test,type = "class")
str(predict_tree)
confusionMatrix(predict_tree,test$churn_status)    #  Accuracy 93.99
Specificity = 739/(739 + 15)                       #  Specificity 98.01  
Specificity
Sensitivity = 90 / (90 + 38)                       #  Sensitivity 70.31 
Sensitivity

## Improvising the model further by using various techniques 
## Like Pruning, so that it doesnt overfit the trainig dataset.
## Lets try to apply the prune method and see if we can further improve the model or not.

set.seed(1000)
tree_validate <- cv.tree(object = tree_model,FUN = prune.misclass )  # Use of cv.tree function to calculate the determine the optimal no. of tree levels
tree_validate                 # Took 7 tree levels into consideration
plot(x=tree_validate$size, y=tree_validate$dev, type="b")

# From the above plot one can see that tree_validate$dev diff is same from for tree levels 11-14, so can
# we assume the best tree level size to be 12 (rather than original number 15) at the cost of some bias. 
# Lets go ahead check the model performance.

tree_model_prun <- prune.misclass(tree_model, best = 12)
plot(tree_model_prun)
text(tree_model_prun, pretty=0)
summary(tree_model_prun)
str(test)
predict_tree_prun <- predict(tree_model_prun, test,type = "class")
str(predict_tree_prun)
confusionMatrix(predict_tree_prun,test$churn_status)      # Accuracy 93.54
Specificity = 733 /(733+21)                               # Specificity 97.2 
Specificity
Sensitivity = 92 / (92 + 36)                              # Sensitivity 71.85
Sensitivity

## The above performance indicators looks way better than the earlier models.
## Lets predict in our final dataset pred and log the performance indicators
set.seed(1000)
predict_tree_prun_pred <- predict(tree_model_prun, pred,type = "class")
confusionMatrix(predict_tree_prun_pred,pred$churn_status)      # Accuracy 93.05
Specificity = 801 /(801+22)                               # Specificity 97.32 
Specificity
Sensitivity = 83 / (83 + 44)                              # Sensitivity 65.35
Sensitivity

## Conclusion:- The latest CART model performance is way better than the earlier models. Here in this model
## we are everytime predicting correctly more than 60% which customers are going to be churned, at the same time keeping
## the fall out ratio to be very less that means the prediction of incorrect customers to chruned is significantly
## low which is only 2.6 %. It seems to be a very convincing model for business, but before that lets proceed
## further with some more ML algorithms.
##

## 4th Model Random Forest 
set.seed(1000)
random_model <- randomForest(as.factor(churn_status)~ Account_Len + Int_Plan + Vmail_Plan
                             + tot_day_mins + tot_evening_mins + tot_ngt_mins
                             + tot_int_mins + cust_calls_made,data = train,ntree = 2000, 
                             importance = TRUE)

#random_model <- randomForest(train$churn_status~ train$Len_Area + train$Int_Plan + train$Vmail_Plan
#                            + train$tot_day_mins + train$tot_evening_mins + train$tot_ngt_mins
#                            + train$tot_int_mins + train$cust_calls_made,data = train,ntree = 1000, 
#                             importance = TRUE)
summary(random_model)
varImpPlot(random_model)        # Variaable Importance plot to show which variables were considered important
                                # for prediction of Customer churned variable. The output shows that the 
                                # variables tot_day_mins, cust_calls_made & Int_Plan are the most important variables.

# Testing the Model on test dataset, Calculating various Model performances
predict_random_model <- predict(random_model, test)
str(predict_random_model)
confusionMatrix(predict_random_model,test$churn_status)      # Accuracy 94.10
Specificity = 746 / (746 + 8)
Specificity                                                  # 98.93
Sensitivity = 84 / (84 + 44)                             
Sensitivity                                                  # 64.62

## Performance Boosting by Cross Validation with folds k=10
#set.seed(1000)
train_control <- trainControl(method="cv", number=10)
random_model_train <- train(as.factor(churn_status)~ Account_Len + Int_Plan + Vmail_Plan
                            + tot_day_mins + tot_evening_mins + tot_ngt_mins
                            + tot_int_mins + cust_calls_made, data=train, method="rf", 
                              metric="Accuracy", trControl=train_control) 
pred_train_model <- predict(random_model_train, test)
str(pred_train_model)
confusionMatrix(pred_train_model,test$churn_status)          # Accuracy 94.67
Specificity = 741 / (741 + 13)             
Specificity                                                  # Specificty 98.27
Sensitivity = 94 / (94 + 34)
Sensitivity                                                  # Sensitivity 73.43

# We can see from the above stats that the model performance after using k cross validation method has
# shooted up the model performances in a better way.

# Lets predict for our final values on pred dataset.
pred_test_model <- predict(random_model_train, pred)
str(pred_test_model)
confusionMatrix(pred_test_model,pred$churn_status)        # Accuracy  93.47
Specificity = 808 / (808 + 15)
Specificity                                               # Specificity 98.17                                     
Sensitivity = 80 / (80 + 47)
Sensitivity                                               # Sensitivity 62.99 

## Conclusion:- The RF model performance looks almost similar to the model performance by CART
## Here as well we are everytime predicting correctly more than 60% which customers are going to be churned, at the same time keeping
## the fall out ratio to be very less that means the prediction of incorrect customers to chruned is significantly
## low which is only 1.83 %. It seems to be a very convincing model again for business, but can any other ML algorithm can do better 
## than this, lets try with some other ML agorithm.
##

## 5th Model SVM Analysis
set.seed(100)
svm_model <- svm(churn_status ~.,data=train,kernel='radial',gamma=1,cost=100)
svm_model
summary(svm_model)

# Testing the Model on test dataset, Calculating various Model performances
test_svm_model <- predict(svm_model,test)
confusionMatrix(test_svm_model,test$churn_status)      # Accuracy  85.49
Specificity                                            # Specicicity 100 %
Sensitivity                                            # Sensitivity 0 %

# One can see from the above stats that the model output is incorrect w.r.t used given comma & beta
# parameters, as the Sensivitiy is 0 % , that means it was not able to predict any customer to be predicted
# correctly.Lets try to fine tune the model now.

## Improving further model performance by various Finetuning techniques
## Using Cross Validation technique to optimise the best values for gamma and cost.
set.seed(1000)
svm.tune <- tune(svm,churn_status ~ Account_Len + Int_Plan + Vmail_Plan
                 + tot_day_mins + tot_evening_mins + tot_ngt_mins
                 + tot_int_mins + cust_calls_made,data=train,kernel='radial',ranges = list(cost = c(0.1,1,10,100,1000),
                                                                  gamma = c(0.5,1,2,3,4)))
summary(svm.tune)
test_svm_model2 <- predict(svm.tune$best.model,test)
confusionMatrix(test_svm_model2,test$churn_status)         # Accuracy 92.74
Specificity =  747 / (747 + 7)                          
Specificity                                                # Specificty 99.07
Sensitivity = 71 / (71 + 57)
Sensitivity                                                # Sensitivity 55.26

# Predicting the final values on pred dataset

pred_svm_model <- predict(svm.tune$best.model,pred)
confusionMatrix(pred_svm_model,pred$churn_status)         # Accuracy 92.74
Specificity =  813 / (813 + 10)                          
Specificity                                                # Specificty 98.78
Sensitivity = 68 / (68 + 59)
Sensitivity                                                # Sensitivity 55.54

## Conclusion:- One can see from the above stats that model performance is not that great
## as compared to earlier models.

## Final Conclusion :-  Out of various models developed for Prediction of Customers going to be churned, 
## Decision Trees Model (CART) and Random Forest suits best for our business case. Both the models are 
## almost similar w.r.t each other. The accuracy for both the models is almost same i.e. 93.05 %. The 
## only difference between their performance is, RF is able to correctly predict 62.99% of customers to
## be churned as true but also with the false out ratio of 1.83%. That means apart from predicting 62.99% 
## correctly it is also predicting incorrectly for some 1.83% customers as churned(which is not true).
## Where as Decision Tree(CART) model is able to predict correctly 65.35% customers to be churned as true
## but also predicts incorrectly for some 2.68% customers as churned(which also is not true). So it depends
## upon Business which model it would like to use if it would like to go for higher predictive power at cost
## of some Fall Out ratio/error deviation Decision Tree is best. ELSE if company wants to stress more on 
## minimizing error rate than RF can be good one.
## Where as Decision Tree(CART) model is able to predict correctly 65.35% customers to be churned as true
## but also predicts incorrectly for some 2.68% customers as churned(which also is not true). So it depends 
## upon Business which model it would like to use if it would like to go for higher predictive power at
## cost of some Fall Out ratio/error deviation Decision Tree is best. ELSE if company wants to stress more 
## on minimizing error rate than RF can be good one.


