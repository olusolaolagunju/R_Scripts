#I USED LOGISTIC REGRESSION ANALYSIS TO PREDICT THE PROBABILITY OF AN INDIVIDUAL TO
# BE FOOD SECURE OR INSECURE

library(readr)
Food_insecurity<- read.csv("C:/Users/DELL/OneDrive/COURSERA COURSES/GOOGLE ANALYTIC/R/Google Data Analytic/Food_security/2_Food_insecurity_combined.csv")
Food_insecurity <- Food_insecurity[, c(3:10, 24)]
view(Food_insecurity)
str(Food_insecurity)
colnames(Food_insecurity)

# grouping household variable


Food_insecurity$household_size <-  case_when(Food_insecurity$household_size <= 2 ~ "1-2",
                                             Food_insecurity$household_size <=4 ~ "3-4",
                                             Food_insecurity$household_size <= 6 ~ "5-6",
                                             TRUE ~ "7-10")

# regrouping the age
Food_insecurity$age[Food_insecurity$age == "51-60"|
                      Food_insecurity$age == "61 and above"] <-  "51 and above"

Food_insecurity %>% 
  distinct(education)
# converting characters to factors 


Food_insecurity$gender <- as.factor(Food_insecurity$gender)
Food_insecurity$age <-  as.factor(Food_insecurity$age)
Food_insecurity$education <- as.factor(Food_insecurity$education)
Food_insecurity$employment_status <-  as.factor(Food_insecurity$employment_status)
Food_insecurity$income <-  as.factor(Food_insecurity$income)
Food_insecurity$children <-  as.factor(Food_insecurity$children)
Food_insecurity$lg <-  as.factor(Food_insecurity$lg)
Food_insecurity$status  <-  as.factor(Food_insecurity$status)
Food_insecurity$household_size <-  as.factor(Food_insecurity$household_size)

# removng any NAs
Food_insecurity <-  na.omit(Food_insecurity)

# Household size food insecurity distribution 

ggplot(Food_insecurity)+
  geom_bar(aes(household_size, fill = status))

# Descriptive stattistics

Food_insecurity %>% 
  group_by(status) %>% 
  summarise(status_n = n()) %>% 
  mutate(status_freq = paste0(round((100 * status_n/sum(status_n)), 0), '%'))

# gender
Food_insecurity %>% 
  group_by(gender) %>% 
  summarise(gender_n = n()) %>%
  mutate(gender_freq = paste0(round((100 * gender_n/sum(gender_n)), 0), '%'))            

# Food_insecurity %>% 
#   group_by(gender, status) %>% 
#   summarise(gender_n = n()) %>%
#   mutate(gender_freq = paste0(round((100 * gender_n/sum(gender_n)), 0), '%'))

# age
Food_insecurity %>% 
  group_by(age) %>% 
  summarise(age_n = n()) %>%
  mutate(age_freq = paste0(round((100 * age_n/sum(age_n)), 0), '%'))

# education
Food_insecurity %>% 
  group_by(education) %>% 
  summarise(education_n = n()) %>%
  mutate(education_freq = paste0(round((100 * education_n/sum(education_n)), 0), '%'))

# employment status
Food_insecurity %>% 
  group_by(employment_status) %>% 
  summarise(employment_status_n = n()) %>%
  mutate(employment_status_freq = paste0(round((100 * employment_status_n/sum(employment_status_n)), 0), '%'))

# income
Food_insecurity %>% 
  group_by(income) %>% 
  summarise(income_n = n()) %>%
  mutate(income_freq = paste0(round((100 * income_n/sum(income_n)), 0), '%'))


# children
Food_insecurity %>% 
  group_by(children) %>% 
  summarise(children_n = n()) %>%
  mutate(children_freq = paste0(round((100 * children_n/sum(children_n)), 0), '%'))

# Lg
Food_insecurity %>% 
  group_by(lg) %>% 
  summarise(lg_n = n()) %>%
  mutate(lg_freq = paste0(round((100 * lg_n/sum(lg_n)), 0), '%'))

# household
Food_insecurity %>% 
  group_by(household_size) %>% 
  summarise(household_n = n()) %>%
  mutate(household_freq = paste0(round((100 * household_n/sum(household_n)), 0), '%'))


# BUILDING THE MODEL
#'We will be splitting the data into the test and train using the createDataPartition() 
#'function from the caret package in R. 
#'We will train the model using the training dataset and predict the values on the test dataset.
#'To train the logistic model, we will be using glm() function.
#'



# checking the contrast of the STATUS

Food_insecurity$status <-  relevel(Food_insecurity$status, ref = "Food_secure")
Food_insecurity$status <-  relevel(Food_insecurity$income, ref = "#31,000 - #70,000")
contrasts(Food_insecurity$status)




# Loading caret library
#install.packages("caret")
require(caret)
library(caret)
rm(Food_insecurity)

# Splitting the data into train and test
index2 <- createDataPartition(Food_insecurity$status, p = .70, list = FALSE)
train2 <- Food_insecurity[index2, ]
food_test <- Food_insecurity[-index2, ]

# Training the model to include all the predictors for the ful model and adjusted model (significant predictors)
food_model <- glm(status ~ ., family = binomial(), train2)
food_model2 <- glm(status ~ education + employment_status + income + household_size, family = binomial(), train2)

# Checking the model
summary(food_model)
tab_model(food_model) # ODD RATIO
summary(food_model2)
tab_model(food_model2)



# RUNNING ANOVA CHISQUARE
anova(food_model2, test ="Chisq") # to check the significance of each variable

# Overall Model:  Evaluation and Goodness-of-Fit Statistics
#1. Wald Test to determine if any of the predictors are significant. 
# if p is significant with the full model, use the full model, otherwise, use the adjusted nested model
# Ward test chi square
#install.packages("aod")
library(aod)
wald.test(Sigma = vcov(food_model), b = coef(food_model), Terms = 1:8)
wald.test(Sigma = vcov(food_model2), b = coef(food_model2), Terms = 1:4)

#2. Likelihood test: to compare the full and the adjusted model
# if p is significant with the full model, use the full model, otherwise, use the adjusted nested model
#install.packages("lmtest")
library(lmtest)
lrtest(food_model, food_model2)



# 3 Goodness of fit model a significant p value means the model does not fit the data well
library(devtools)
#install_github("gnattino/largesamplehl") # for  hltest goodness of fit
library(largesamplehl)
#install.packages("ResourceSelection") #  hoslem.test goodness of fit test
# install.packages("rms")
#install.packages("performance")
#install.packages("PredictABEL")
library(ResourceSelection)
library(performance)         # for performance_hosmer goodness of fit test
library(PredictABEL)
library(rms)


### Goodnes of fit test: Method 1:
hoslem.test(train2$status, fitted(food_model2), g = 10) ### Use the unfactored data for the outcomes


### Goodnes of fit test: Method 2
performance_hosmer(food_model2, n_bins = 10) # model does fit well 
performance_hosmer(food_model, n_bins = 10) # model does not fit well 

## method 3

hltest(food_model2, G = 10) 

## method 4 omnibus goodness of fit test THE PREFERED FIT TEST
logit1.res <- lrm(status ~., data = train2, y = TRUE, x = TRUE)

residuals(logit1.res, type = "gof")


# Predicting in the test dataset

pred_prob <- predict(food_model2, food_test, type = "response")


## Converting from probability to actual output
train2$pred_class <- ifelse(food_model2$fitted.values >= 0.5, "Food_secure", "Food_insecure")

# Generating the classification table for the trained dataset (.70)
ctab_train <- table(train2$status, train2$pred_class)
ctab_train


# Training dataset converting from probability to class values
# Converting probability to class values in the training dataset
# Converting from probability to actual output
food_test$pred_class <- ifelse(pred_prob >= 0.5, "Food_secure ", "Food_insecure ")

# Generating the classification table for the test  dataset (.30)
ctab_test <- table(food_test$status, food_test$pred_class)
ctab_test

#' Training dataset converting from probability to class values

# # Accuracy
# Accuracy = (TP + TN)/(TN + FP + FN + TP)
# Accuracy in Training dataset
accuracy_train <- sum(diag(ctab_train))/sum(ctab_train)*100
accuracy_train

#the logistics model is able to classify 80.54% of all the observations correctly in the test dataset.
# Accuracy in Test dataset. This shows that the model perfomed well
accuracy_test <- sum(diag(ctab_test))/sum(ctab_test)*100
accuracy_test


# MISCLASSIFICATION FOR TRUE POSITIVE AND NEGATIVE
Accuracy = (TP + TN)/(TN + FP + FN + TP)

#Misclassification Rate = (FP+FN)/(TN + FP + FN + TP)
# True Positive Rate – Recall or Sensitivity
# Recall or TPR indicates how often does our model predicts actual TRUE from the 
#overall TRUE events.
# Also called Specificity : 88.75%
Recall <- (ctab_train[2, 2]/sum(ctab_train[2, ]))*100
Recall

# True Negative Rate
# TNR indicates how often does our model predicts actual nonevents from the overall nonevents.

TNR = TN/(TN + FP)
# TNR in Train dataset
TNR <- (ctab_train[1, 1]/sum(ctab_train[1, ]))*100
TNR

# Precision
# Precision indicates how often does your predicted TRUE values are actually TRUE.

Precision = TP/FP + TP
# Precision in Train dataset
Precision <- (ctab_train[2, 2]/sum(ctab_train[, 2]))*100
Precision

# Calculating F-Score
# F-Score is a harmonic mean of recall and precision. The score value lies between 0 and 1. The value of 1 represents perfect precision & recall. The value 0 represents the worst case.

F_Score <- (2 * Precision * Recall / (Precision + Recall))/100
F_Score

# ROC Curve
# The area under the curve(AUC) is the measure that represents ROC(Receiver Operating Characteristic) curve. This ROC curve is a line plot that is drawn between the Sensitivity and (1 – Specificity) Or between TPR and TNR. This graph is then used to generate the AUC value. An AUC value of greater than .70 indicates a good model.
library(pROC)
roc <- roc(train2$status, food_model$fitted.values)
auc(roc)
plot(roc)el2$y,food_model2$fitted.values)

# odds
#' categorical odds ratio = 1.81 means the odds for females are 81% higher than the odds for males
#' #' categorical odds ratio = 0.81 means the odds for females are 81% lower than the odds for males
#' continous odds ratio = 1.66922 means we exepect to see 17% increase in the odss of being in the honor 
#' for a one unit increase in math score
#' continous odds ratio = 0.66922 means we exepect to see 17% decrease in the odss of being in the honor 
#' for a one unit increase in math score

