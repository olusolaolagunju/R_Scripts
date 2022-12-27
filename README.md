# Graduate_School_R_Scripts

## RScript1: Linear Regression
* The script contains codes I used to evaluate the Covid-19 dataset. 
* I performed a correlation test to find the relationship between the dependent IFR, CFR, and VR variables and the independent variables, followed by a Regression analysis modeling to determine the significant effect of the independent variables on the dependent variables. 
* The model I built for the Infection rate had adjusted R2   = 68%. Indicating, other independent factors that could have influenced the infection rate were not captured in the dataset.
* I used boxplot for the descriptive statistical analysis 

## RScript2: Logistic Regression
* The script contains codes I used to evaluate Food Insecurity [survey data](https://docs.google.com/forms/d/e/1FAIpQLSfnENEsf881JKhhjzWn0hcUC3KQo8snrPBI6qSfSVPEq6Nklw/viewform?usp=sf_link) collected by me. 
* I coded the data as binary outcomes ( food secure and food insecure) using the criteria set by the USDA
* I split the dataset into train and test (70:30) and then used the trained dataset to predict the outcome of the  model test
* I evaluated the fitness of the data using the Likelihood test, Wald test, and the Goodness of fit test. The model I built fits the data well.
* The accuracy of the model to classify the test observations into Food secure and Food insecure was 78.9%.
* True Positive Rate – Recall or Sensitivity was 95.7%, while the Precision of the model was 82.5%
* The F-score and the Area under the score were 0.9 and 0.8 respectively, which indicates a good model


## RScript3: XGBoost Regression 
