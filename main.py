import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# load the data
customers = pd.read_csv("../Ecommerce Customers")

#check info on data
data_top_head=customers.head()
data_top_describe= customers.describe()

################################
#       Data Exploration       #
################################

#compare the Time on Website and Yearly Amount Spent columns
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

#compare the Time on App and Yearly Amount Spent columns
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

#show pairplot of the whole data
sns.pairplot(customers)

#show heatmap of the whole data
sns.heatmap(customers.corr())



###############################
#       Train/Test Split      #
###############################

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#create a model
lm = LinearRegression()
lm.fit(X_train,y_train)


#make predictions
predictions = lm.predict( X_test)



plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# The coefficients
print('Coefficients: \n', lm.coef_)


#Evaluate the model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


#CHeck the residual so that it is normally distributed
sns.distplot((y_test-predictions),bins=50)



coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)


plt.show()


