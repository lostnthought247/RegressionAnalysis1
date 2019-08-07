import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("MockSchoolToIncome.csv")
print(df.head())

salaries = df["Salary"].tolist()
yeas_in_edu = df["Years Higher Education"].tolist()
yeas_in_edu = np.array(yeas_in_edu)
yeas_in_edu = yeas_in_edu.reshape(-1, 1)

line_fitter = LinearRegression()
line_fitter.fit(yeas_in_edu, salaries)

salary_predict = line_fitter.predict(yeas_in_edu)

input_years = input("Please input years of education...")
salary_est = line_fitter.predict(int(input_years))
print("The estimated salary for %s is %s" % (input_years, salary_est))

plt.scatter(df["Years Higher Education"], df["Salary"])
plt.plot(yeas_in_edu, salary_predict)
plt.title("Mock Salary/Education Regression")
plt.xlabel("Years of Education")
plt.ylabel("Salary")
labels = ["Salary", "Prediction"]
plt.plot(int(input_years), salary_est, 'rX')
plt.show()
