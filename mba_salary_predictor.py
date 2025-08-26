
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=4, linewidth=100)
#loading the dataset
mba_salary_df=pd.read_csv("MBA Salary.csv")

#print(mba_salary_df)

import statsmodels.api as sm

#creating feature set(X) and outcome variable(Y)
x=sm.add_constant(mba_salary_df["Percentage in Grade 10"])
#print(mba_salary_df.head())
Y=mba_salary_df["Salary"]

# splitting the dataset into training and validation set
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,Y,test_size=0.8,random_state=100)

#fitting the model
mba_salary_lm=sm.OLS(train_y,train_x).fit()

#printing the estimated parameters and interpreting train

print(mba_salary_lm.params)

print(mba_salary_lm.summary2())

#checking the normal distribution of residual
mba_salary_resid =mba_salary_lm.resid
residuals = mba_salary_lm.resid
residuals_std = (residuals - residuals.mean()) / residuals.std()
probplot=sm.ProbPlot(residuals_std)
plt.figure(figsize=(12,8))
probplot.ppplot(line='45')
plt.title('Normal P-P plot of regression standardised residuals')
plt.show()

from scipy.stats import shapiro
print(shapiro(mba_salary_resid))

#checking for homoscedasticity
sns.residplot(x=mba_salary_lm.fittedvalues,y=mba_salary_resid,lowess=True, line_kws={'color':'red','lw':1,'alpha':0.8})
plt.xlabel("fitted values")
plt.ylabel("residuals")
plt.show()