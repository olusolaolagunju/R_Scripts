library(tidyverse)
library(skimr)
library(forcats) 
library(dplyr)
#install.packages("ggpmisc") # to create table inside ggplot
library(ggpmisc)
library(readxl)
#install.packages("hablar")
library(hablar) # to tackle Inf and NA
library(ggplot2)
#install.packages("corrtable")
library(corrtable)
# install.packages("sjPlot")
# install.packages("sjmisc")
# install.packages("sjlabelled")
library(sjPlot) # models
library(sjmisc)# regression model
library(sjlabelled) # regression model
#install.packages("gtExtras")
library(gtExtras)    # GT TABLES AND THEMES
#install.packages("gt")
library(gt)
#install.packages("multcompView") # to compare anova and turkey test
library(multcompView)

# I USED ONE-WAY ANOVA, COREELATION AND LINEAR REGRESSION TO ANALYZE COVID-19 DATASET

# read data
covid_ethiopia <- read_excel("C:/Users/DELL/OneDrive/COURSERA COURSES/GOOGLE ANALYTIC/R/Google Data Analytic/dataset/covid_ethiopia.xlsx")


# view data
colnames(covid_ethiopia)
skim_without_charts(covid_ethiopia)


#removing outliers
clean_data <- covid_ethiopia %>% 
  select(iso_code, continent, location, date, population,  new_cases,total_cases,
         new_deaths, total_deaths, people_vaccinated, aged_65_older, gdp_per_capita, 
         people_fully_vaccinated, median_age, extreme_poverty, cardiovasc_death_rate, 
         diabetes_prevalence,hospital_beds_per_thousand,life_expectancy, human_development_index, aged_70_older) %>% 
  filter(continent !=c("World", "Asia", "Lower middle income, Upper middle income", 
                       "High income", "Low income"), location != "North Korea")

Summarised_data <- clean_data %>% 
  group_by(continent,location, population) %>% 
  summarise(total_case = sum(new_cases, na.rm = TRUE), 
            case = sum(new_cases, na.rm = TRUE), 
            total_death = sum(new_deaths, na.rm = TRUE), 
            vacc = max_(people_vaccinated),
            Median_age = mean(median_age, na.rm= TRUE), 
            Older_than_65 = mean(aged_65_older, na.rm= TRUE),
            Gdp_capita= mean(gdp_per_capita, na.rm= TRUE),
            diabetes_prev= mean(diabetes_prevalence, na.rm= TRUE),
            card_death = mean(cardiovasc_death_rate, na.rm= TRUE),
            age_70 = mean(aged_70_older, na.rm =TRUE),
            bed_per_thousand = mean (hospital_beds_per_thousand, na.rm =TRUE)) %>% 
  mutate(Infection_rate = total_case/population * 100,
         Case_fatality_rate = total_death/case *100,
         Vaccination_rate = vacc/population * 100)



# CORRELATION

corr <- correlation_matrix(Summarised_data[, 8:17],digits = 2, use = "lower", replace_diagonal = TRUE)
corr




write.table(corr, file = "covidcorr.txt", sep = ",", quote = FALSE, row.names = F) # save as txt file

corrtable <- read_excel("C:/Users/DELL/OneDrive/COURSERA COURSES/GOOGLE ANALYTIC/R/Google Data Analytic/dataset/correlation_covid_table.xlsx")

corrtable2 <- corrtable[8:10, c(-7, -9, -10, -11) ]

corrtable2%>% 
  gt() %>% 
  sub_missing(
    columns = everything(),
    rows = everything(),
    missing_text = "---"
  ) %>% 
  tab_header(title = "Table 1: Correlation of Covid-19 Independent factors and IFR, CFR and VR",
             subtitle = "Infection rate: Age has strong correlation with IFR |
             Case fatality rate: Gdp per capita and Bed per thousand have weak correlations with CFR \n | Vacination rate: Age and Gdp per capita have moderate correlations with VR"
  ) %>% 
  tab_footnote(footnote = " * = level of significance, IFR = Infection rate, 
               CFR = Case fatality rate, VR = Vaccination rate")

#REGRESSION MODEL

IFR_Model <- lm(Infection_rate ~ Median_age + Gdp_capita, data = Summarised_data)

summary(IFR_Model)


CFR_Model <- lm(Case_fatality_rate ~ bed_per_thousand + Gdp_capita, data = Summarised_data)
summary(CFR_Model)

VR_Model <- lm(Vaccination_rate~  Median_age + Gdp_capita, data = Summarised_data)

summary(VR_Model)


tab_model(IFR_Model)

tab_model(CFR_Model)
tab_model(VR_Model)
tab_model(IFR_Model, CFR_Model, VR_Model)

# ANOVA
# Infection rate One way anova, turkey and boxplot
#install.packages("multcompView") # to compare anova and turkey test
library(multcompView)
IFR_anova <- aov(Infection_rate ~ continent, data = Summarised_data)
summary(IFR_anova)

# turkey test
tukey <- TukeyHSD(IFR_anova)
print(tukey)

# Compact letter display to indicate significant differences
cld <- multcompLetters4(IFR_anova, tukey)
print(cld)

# Creating a table with the summarised data and the compact letter display
# table with factors and 3rd quantile
Tk <- group_by(Summarised_data, continent) %>%
  summarise(mean=mean(Infection_rate, na.rm = TRUE), quant = quantile(Infection_rate , probs = 0.75, na.rm = TRUE)) %>%
  arrange(desc(mean))

# extracting the compact letter display and adding to the Tk table
cld <- as.data.frame.list(cld$continent)
Tk$cld <- cld$Letters

print(Tk)

TM <- Tk %>% mutate_at(vars(mean, quant), funs(round(., 1)))



# Box plot
# boxplot



ggplot(Summarised_data, aes(continent, Infection_rate)) + 
  geom_boxplot(aes(fill = continent), outlier.colour = "red", outlier.shape = 1, show.legend = FALSE)+
  labs(title = "Figure 1: Global Infection Rate by Sept 2022",
       subtitle = " Africa had the lowest average infection rate (3.2%) across all continents",
       x="Continent", y="Infection rate (%)")+
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  geom_text(data = TM, aes(label = mean, y = mean + 0.008), size = 3)+
  geom_text(data = TM, aes(x = continent, y = quant, label = cld), size = 4, vjust=-1, hjust =-1)+
  #scale_fill_brewer(palette = "Pastel1")
  scale_fill_manual(values = c('lightgreen', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey' ))




ggsave("boxplot_IFR.png", width = 6, height = 4, dpi = 1000)
# 


# CaseFatality rate One way anova, turkey and boxplot
# CFR
CFR_anova <-  aov(Case_fatality_rate ~ continent, data = Summarised_data)
summary(CFR_anova)

# turkey test
tukey_CFR <- TukeyHSD(CFR_anova)
print(tukey_CFR)

# Compact letter display to indicate significant differences
cld_CFR <- multcompLetters4(CFR_anova, tukey_CFR)
print(cld_CFR)

# Creating a table with the summarised data and the compact letter display
# table with factors and 3rd quantile
Tk_CFR <- group_by(Summarised_data, continent) %>%
  summarise(mean=mean(Case_fatality_rate, na.rm = TRUE), quant = quantile(Case_fatality_rate , probs = 0.75, na.rm = TRUE)) %>%
  arrange(desc(mean))

# extracting the compact letter display and adding to the Tk table
cld_CFR <- as.data.frame.list(cld_CFR$continent)
Tk_CFR$cld_CFR <- cld_CFR$Letters

print(Tk_CFR)

TM_CFR <- Tk_CFR %>% mutate_at(vars(mean, quant), funs(round(., 1)))

TM_CFR
# Box plot



# boxplot
ggplot(Summarised_data, aes(continent, Case_fatality_rate)) + 
  geom_boxplot(aes(fill = continent), outlier.colour = "red", outlier.shape = 1, show.legend = FALSE)+
  labs(title = "Figure 2: Global Case Fatality Rate for Sept 2022",
       subtitle = " Africa  recorded the second highest average deaths per cases (1.8%) despite\n having the lowest infection rate (3.2%)", 
       x="Continent", y="Case fatality rate (logarithmic scale) (%)")+
  scale_y_log10()+
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  geom_text(data = TM_CFR, aes(label = mean, y = mean + 0.1), size = 3)+
  geom_text(data = TM_CFR, aes(x = continent, y = quant, label = cld_CFR), size = 4, vjust=-1, hjust =-1)+
  scale_fill_manual(values = c('lightgreen', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgreen' ))
#scale_fill_brewer(palette = "Pastel1")


ggsave("boxplot_CFR.png", width = 6, height = 4, dpi = 1000)
# 



# VR 
VR_anova <- aov(Vaccination_rate ~ continent, data = Summarised_data)
summary(VR_anova)


# turkey test
tukey_VR <- TukeyHSD(VR_anova)
print(tukey_VR)

# Compact letter display to indicate significant differences
cld_VR <- multcompLetters4(VR_anova, tukey_VR)
print(cld_VR)

# Creating a table with the summarised data and the compact letter display
# table with factors and 3rd quantile
Tk_VR <- group_by(Summarised_data, continent) %>%
  summarise(mean=mean(Vaccination_rate, na.rm = TRUE), quant = quantile(Vaccination_rate , probs = 0.75, na.rm = TRUE)) %>%
  arrange(desc(mean))

# extracting the compact letter display and adding to the Tk table
cld_VR <- as.data.frame.list(cld_VR$continent)
Tk_VR$cld_VR <- cld_VR$Letters

print(Tk_VR)

TM_VR <- Tk_VR %>% mutate_at(vars(mean, quant), funs(round(., 1)))

TM_VR
# Box plot



# boxplot
ggplot(Summarised_data, aes(continent, Vaccination_rate)) + 
  geom_boxplot(aes(fill = continent), outlier.colour = "red", outlier.shape = 1, show.legend = FALSE)+
  labs(title = " Figure 3: Global Vaccination Rate of Covid-19 Vaccine by Sept 2022",
       subtitle = " Percentage of the number of people that have recieved at least a dose of Covid-19 \n vaccine per population. Africa had the least vaccination rate (34.3%)",
       x="Continent", y="Vaccination rate (%)")+
  #scale_y_log10()+
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  geom_text(data = TM_VR, aes(label = mean, y = mean + 0.1), size = 3)+
  geom_text(data = TM_VR, aes(x = continent, y = quant, label = cld_VR), size = 4, vjust=-1, hjust =-1)+
  scale_fill_manual(values = c('lightgreen', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey' ))
#scale_fill_brewer(palette = "Pastel1")

ggsave("boxplot_VR.png", width = 7, height = 4, dpi = 1000)

