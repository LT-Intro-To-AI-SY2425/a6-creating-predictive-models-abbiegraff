import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age", "year"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)
#create linear regression model
model = LinearRegression().fit(xtrain, ytrain)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)

print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {intercept}")
print("R Squared value:", r_squared)

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")
predict = model.predict(xtest)
predict = np.around(predict, 2)

for index in range(len(xtest)):
    actual = ytest[index]  
    predicted_y = predict[index] 
    x_coord = xtest[index]  
    print(f"Miles: {x_coord[0]} Age: {x_coord[1]} Year: {x_coord[2]} Actual Price: {actual} Predicted Price: {predicted_y}")

x_1 = data["miles"]
x_2 = data["age"]
x_3 = data["year"]
y = data["Price"]

fig, graph = plt.subplots(3)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("miles")
graph[0].set_ylabel("price")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("age")
graph[1].set_ylabel("price")

graph[2].scatter(x_3, y)
graph[2].set_xlabel("year")
graph[2].set_ylabel("price")

print("Correlation between miles and price:",round(x_1.corr(y),2))
print("Correlation between age and price:",round(x_2.corr(y),2))
print("Correlation between year and price:",round(x_3.corr(y),2))

plt.tight_layout()
plt.show()