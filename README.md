# d12
# d12
#pandas
import pandas as pd
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
print(data.head()) =>       TV  radio  newspaper  sales
                      1  230.1   37.8       69.2   22.1
                      2   44.5   39.3       45.1   10.4
                      3   17.2   45.9       69.3    9.3
                      4  151.5   41.3       58.5   18.5
                      5  180.8   10.8       58.4   12.9
print(data.tail()) =>       TV  radio  newspaper  sales
                    196   38.2    3.7       13.8    7.6
                    197   94.2    4.9        8.1    9.7
                    198  177.0    9.3        6.4   12.8
                    199  283.6   42.0       66.2   25.5
                    200  232.1    8.6        8.7   13.4
print(data.shape) => (200,4)

#seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
sns.pairplot(data,x_vars=['TV','radio','newspaper'],y_vars='sales',size=7,aspect=0.7,kind='reg')
plt.show()

#linear-regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
feature_cols = ['TV','radio','newspaper']
X = data[['TV','radio','newspaper']]
y = data['sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
linear = LinearRegression()
linear.fit(X_train,y_train)
y_pred = linear_predict(X_test)
print(linear.intercept_) => 2.8769666223179353      #截距
print(linear.coef_) => [0.04656457 0.17915812 0.00345046]        #系数
print(list(zip(feature_cols,linear.coef_))) => [('TV', 0.04656456787415026), ('radio', 0.1791581224508884), ('newspaper', 0.0034504647111804065)]
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred))) => 1.4046514230328948

#cross-validation
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=5)
score = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print(score) => [1.         0.93333333 1.         1.         0.86666667 0.93333333
                 0.93333333 1.         1.         1.        ]
print(score.mean()) => 0.9666666666666668

k_range = range[1,31]
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
