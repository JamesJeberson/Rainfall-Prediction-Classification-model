<p style="text-align:center">
    <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
    </a>
</p>

<h1 align="center"><font size="5">Final Project: Classification with Python</font></h1>


<h2>Table of Contents</h2>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ul>
    <li><a href="https://#Section_1">Instructions</a></li>
    <li><a href="https://#Section_2">About the Data</a></li>
    <li><a href="https://#Section_3">Importing Data </a></li>
    <li><a href="https://#Section_4">Data Preprocessing</a> </li>
    <li><a href="https://#Section_5">One Hot Encoding </a></li>
    <li><a href="https://#Section_6">Train and Test Data Split </a></li>
    <li><a href="https://#Section_7">Train Logistic Regression, KNN, Decision Tree, SVM, and Linear Regression models and return their appropriate accuracy scores</a></li>
</a></li>
<hr>


# Instructions


In this notebook, you will  practice all the classification algorithms that we have learned in this course.


Below, is where we are going to use the classification algorithms to create a model based on our training data and evaluate our testing data using evaluation metrics learned in the course.

We will use some of the algorithms taught in the course, specifically:

1. Linear Regression
2. KNN
3. Decision Trees
4. Logistic Regression
5. SVM

We will evaluate our models using:

1.  Accuracy Score
2.  Jaccard Index
3.  F1-Score
4.  LogLoss
5.  Mean Absolute Error
6.  Mean Squared Error
7.  R2-Score

Finally, you will use your models to generate the report at the end. 


# About The Dataset


The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from [http://www.bom.gov.au/climate/dwo/](http://www.bom.gov.au/climate/dwo/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01).

The dataset to be used has extra columns like 'RainToday' and our target is 'RainTomorrow', which was gathered from the Rattle at [https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData](https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)




This dataset contains observations of weather metrics for each day from 2008 to 2017. The **weatherAUS.csv** dataset includes the following fields:

| Field         | Description                                           | Unit            | Type   |
| ------------- | ----------------------------------------------------- | --------------- | ------ |
| Date          | Date of the Observation in YYYY-MM-DD                 | Date            | object |
| Location      | Location of the Observation                           | Location        | object |
| MinTemp       | Minimum temperature                                   | Celsius         | float  |
| MaxTemp       | Maximum temperature                                   | Celsius         | float  |
| Rainfall      | Amount of rainfall                                    | Millimeters     | float  |
| Evaporation   | Amount of evaporation                                 | Millimeters     | float  |
| Sunshine      | Amount of bright sunshine                             | hours           | float  |
| WindGustDir   | Direction of the strongest gust                       | Compass Points  | object |
| WindGustSpeed | Speed of the strongest gust                           | Kilometers/Hour | object |
| WindDir9am    | Wind direction averaged of 10 minutes prior to 9am    | Compass Points  | object |
| WindDir3pm    | Wind direction averaged of 10 minutes prior to 3pm    | Compass Points  | object |
| WindSpeed9am  | Wind speed averaged of 10 minutes prior to 9am        | Kilometers/Hour | float  |
| WindSpeed3pm  | Wind speed averaged of 10 minutes prior to 3pm        | Kilometers/Hour | float  |
| Humidity9am   | Humidity at 9am                                       | Percent         | float  |
| Humidity3pm   | Humidity at 3pm                                       | Percent         | float  |
| Pressure9am   | Atmospheric pressure reduced to mean sea level at 9am | Hectopascal     | float  |
| Pressure3pm   | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascal     | float  |
| Cloud9am      | Fraction of the sky obscured by cloud at 9am          | Eights          | float  |
| Cloud3pm      | Fraction of the sky obscured by cloud at 3pm          | Eights          | float  |
| Temp9am       | Temperature at 9am                                    | Celsius         | float  |
| Temp3pm       | Temperature at 3pm                                    | Celsius         | float  |
| RainToday     | If there was rain today                               | Yes/No          | object |
| RainTomorrow  | If there is rain tomorrow                             | Yes/No          | float  |

Column definitions were gathered from [http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)



## **Import the required libraries**



```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
```

### Importing the Dataset



```python
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv"
df = pd.read_csv(filepath)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/1/2008</td>
      <td>19.5</td>
      <td>22.4</td>
      <td>15.6</td>
      <td>6.2</td>
      <td>0.0</td>
      <td>W</td>
      <td>41</td>
      <td>S</td>
      <td>SSW</td>
      <td>...</td>
      <td>92</td>
      <td>84</td>
      <td>1017.6</td>
      <td>1017.4</td>
      <td>8</td>
      <td>8</td>
      <td>20.7</td>
      <td>20.9</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/2/2008</td>
      <td>19.5</td>
      <td>25.6</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>2.7</td>
      <td>W</td>
      <td>41</td>
      <td>W</td>
      <td>E</td>
      <td>...</td>
      <td>83</td>
      <td>73</td>
      <td>1017.9</td>
      <td>1016.4</td>
      <td>7</td>
      <td>7</td>
      <td>22.4</td>
      <td>24.8</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/3/2008</td>
      <td>21.6</td>
      <td>24.5</td>
      <td>6.6</td>
      <td>2.4</td>
      <td>0.1</td>
      <td>W</td>
      <td>41</td>
      <td>ESE</td>
      <td>ESE</td>
      <td>...</td>
      <td>88</td>
      <td>86</td>
      <td>1016.7</td>
      <td>1015.6</td>
      <td>7</td>
      <td>8</td>
      <td>23.5</td>
      <td>23.0</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/4/2008</td>
      <td>20.2</td>
      <td>22.8</td>
      <td>18.8</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>W</td>
      <td>41</td>
      <td>NNE</td>
      <td>E</td>
      <td>...</td>
      <td>83</td>
      <td>90</td>
      <td>1014.2</td>
      <td>1011.8</td>
      <td>8</td>
      <td>8</td>
      <td>21.4</td>
      <td>20.9</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/5/2008</td>
      <td>19.7</td>
      <td>25.7</td>
      <td>77.4</td>
      <td>4.8</td>
      <td>0.0</td>
      <td>W</td>
      <td>41</td>
      <td>NNE</td>
      <td>W</td>
      <td>...</td>
      <td>88</td>
      <td>74</td>
      <td>1008.3</td>
      <td>1004.8</td>
      <td>8</td>
      <td>8</td>
      <td>22.5</td>
      <td>25.5</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



### Data Preprocessing


#### One Hot Encoding


First, we need to perform one hot encoding to convert categorical variables to binary variables.



```python
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
```


```python
df_sydney_processed.columns
```




    Index(['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
           'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
           'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
           'Temp9am', 'Temp3pm', 'RainTomorrow', 'RainToday_No', 'RainToday_Yes',
           'WindGustDir_E', 'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N',
           'WindGustDir_NE', 'WindGustDir_NNE', 'WindGustDir_NNW',
           'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE', 'WindGustDir_SSE',
           'WindGustDir_SSW', 'WindGustDir_SW', 'WindGustDir_W', 'WindGustDir_WNW',
           'WindGustDir_WSW', 'WindDir9am_E', 'WindDir9am_ENE', 'WindDir9am_ESE',
           'WindDir9am_N', 'WindDir9am_NE', 'WindDir9am_NNE', 'WindDir9am_NNW',
           'WindDir9am_NW', 'WindDir9am_S', 'WindDir9am_SE', 'WindDir9am_SSE',
           'WindDir9am_SSW', 'WindDir9am_SW', 'WindDir9am_W', 'WindDir9am_WNW',
           'WindDir9am_WSW', 'WindDir3pm_E', 'WindDir3pm_ENE', 'WindDir3pm_ESE',
           'WindDir3pm_N', 'WindDir3pm_NE', 'WindDir3pm_NNE', 'WindDir3pm_NNW',
           'WindDir3pm_NW', 'WindDir3pm_S', 'WindDir3pm_SE', 'WindDir3pm_SSE',
           'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W', 'WindDir3pm_WNW',
           'WindDir3pm_WSW'],
          dtype='object')




```python
df_sydney_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>...</th>
      <th>WindDir3pm_NNW</th>
      <th>WindDir3pm_NW</th>
      <th>WindDir3pm_S</th>
      <th>WindDir3pm_SE</th>
      <th>WindDir3pm_SSE</th>
      <th>WindDir3pm_SSW</th>
      <th>WindDir3pm_SW</th>
      <th>WindDir3pm_W</th>
      <th>WindDir3pm_WNW</th>
      <th>WindDir3pm_WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/1/2008</td>
      <td>19.5</td>
      <td>22.4</td>
      <td>15.6</td>
      <td>6.2</td>
      <td>0.0</td>
      <td>41</td>
      <td>17</td>
      <td>20</td>
      <td>92</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/2/2008</td>
      <td>19.5</td>
      <td>25.6</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>2.7</td>
      <td>41</td>
      <td>9</td>
      <td>13</td>
      <td>83</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/3/2008</td>
      <td>21.6</td>
      <td>24.5</td>
      <td>6.6</td>
      <td>2.4</td>
      <td>0.1</td>
      <td>41</td>
      <td>17</td>
      <td>2</td>
      <td>88</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/4/2008</td>
      <td>20.2</td>
      <td>22.8</td>
      <td>18.8</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>41</td>
      <td>22</td>
      <td>20</td>
      <td>83</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/5/2008</td>
      <td>19.7</td>
      <td>25.7</td>
      <td>77.4</td>
      <td>4.8</td>
      <td>0.0</td>
      <td>41</td>
      <td>11</td>
      <td>6</td>
      <td>88</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>




```python
df_sydney_processed['RainTomorrow']
```




    0       Yes
    1       Yes
    2       Yes
    3       Yes
    4       Yes
           ... 
    3266     No
    3267     No
    3268     No
    3269     No
    3270     No
    Name: RainTomorrow, Length: 3271, dtype: object



Next, we replace the values of the 'RainTomorrow' column changing them from a categorical column to a binary column. We do not use the `get_dummies` method because we would end up with two columns for 'RainTomorrow' and we do not want, since 'RainTomorrow' is our target.



```python
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)
```

    C:\Users\james\AppData\Local\Temp\ipykernel_25564\288546165.py:1: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)
    


```python
df_sydney_processed['RainTomorrow']
```




    0       1
    1       1
    2       1
    3       1
    4       1
           ..
    3266    0
    3267    0
    3268    0
    3269    0
    3270    0
    Name: RainTomorrow, Length: 3271, dtype: int64



### Training Data and Test Data


Now, we set our 'features' or x values and our Y or target variable.



```python
df_sydney_processed.drop('Date',axis=1,inplace=True)
```


```python
df_sydney_processed = df_sydney_processed.astype(float)
```


```python
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
```

### Linear Regression


#### Q1) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `10`.



```python
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size = 0.2, random_state = 10)
```

#### Q2) Create and train a Linear Regression model called LinearReg using the training data (`x_train`, `y_train`).



```python
LinearReg = LinearRegression().fit(X_train, y_train)
```

#### Q3) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.



```python
predictions = LinearReg.predict(X_test)
```

#### Q4) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.



```python
LinearRegression_MAE = metrics.mean_absolute_error(y_test, predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test, predictions)
LinearRegression_R2 = metrics.r2_score(y_test, predictions)
```

#### Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.



```python
metrics_dict = {
    'Metrics': ['MAE', 'MSE', 'R2'],
    'Values': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
}

Report = pd.DataFrame(metrics_dict)
Report
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MAE</td>
      <td>0.256316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.115723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R2</td>
      <td>0.427121</td>
    </tr>
  </tbody>
</table>
</div>



### KNN


#### Q6) Create and train a KNN model called KNN using the training data (`x_train`, `y_train`) with the `n_neighbors` parameter set to `4`.



```python
k = 4
KNN = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
```

#### Q7) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.



```python
predictions = KNN.predict(X_test)
```

#### Q8) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.



```python
KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

metrics_dict = {
    'Metrics': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Values': [KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score]
}

Report = pd.DataFrame(metrics_dict)
Report
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy Score</td>
      <td>0.818321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jaccard Index</td>
      <td>0.425121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 Score</td>
      <td>0.596610</td>
    </tr>
  </tbody>
</table>
</div>



### Decision Tree


#### Q9) Create and train a Decision Tree model called Tree using the training data (`x_train`, `y_train`).



```python
Tree = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
```

#### Q10) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.



```python
predictions = Tree.predict(X_test)
```

#### Q11) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.



```python
Tree_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions) 
Tree_F1_Score = f1_score(y_test, predictions)

metrics_dict = {
    'Metrics': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Values': [Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score]
}

Report = pd.DataFrame(metrics_dict)
Report
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy Score</td>
      <td>0.763359</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jaccard Index</td>
      <td>0.421642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 Score</td>
      <td>0.593176</td>
    </tr>
  </tbody>
</table>
</div>



### Logistic Regression


#### Q12) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `1`.



```python
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size= 0.2, random_state= 1)
```

#### Q13) Create and train a LogisticRegression model called LR using the training data (`x_train`, `y_train`) with the `solver` parameter set to `liblinear`.



```python
LR = LogisticRegression(C= 0.01, solver='liblinear').fit(X_train, y_train)
```

#### Q14) Now, use the `predict` and `predict_proba` methods on the testing data (`x_test`) and save it as 2 arrays `predictions` and `predict_proba`.



```python
predictions = LR.predict(X_test)
```


```python
predict_proba = LR.predict_proba(X_test)
```

#### Q15) Using the `predictions`, `predict_proba` and the `y_test` dataframe calculate the value for each metric using the appropriate function.



```python
LR_Accuracy_Score = metrics.accuracy_score (y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)

metrics_dict = {
    'Metrics': ['Accuracy Score', 'Jaccard Index', 'F1 Score', 'Log Loss'],
    'Values': [LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score, LR_Log_Loss]
}

Report = pd.DataFrame(metrics_dict)
Report
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy Score</td>
      <td>0.827481</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jaccard Index</td>
      <td>0.484018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 Score</td>
      <td>0.652308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Log Loss</td>
      <td>0.380085</td>
    </tr>
  </tbody>
</table>
</div>



### SVM


#### Q16) Create and train a SVM model called SVM using the training data (`x_train`, `y_train`).



```python
SVM = svm.SVC(kernel='rbf').fit(X_train, y_train)
```

#### Q17) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.



```python
predictions = SVM.predict(X_test)
```

#### Q18) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.



```python
SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

metrics_dict = {
    'Metrics': ['Accuracy Score', 'Jaccard Index', 'F1 Score',],
    'Values': [SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]
}

Report = pd.DataFrame(metrics_dict)
Report
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy Score</td>
      <td>0.722137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jaccard Index</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 Score</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Report


#### Q19) Show the Accuracy,Jaccard Index,F1-Score and LogLoss in a tabular format using data frame for all of the above models.

\*LogLoss is only for Logistic Regression Model



```python
Report = pd.DataFrame({
    'Metrics': [ 'Mean Absolute Error', 'Mean Squared Error', 'R2 Score', 'Accuracy', 'Jaccard Index', 'F1-Score', 'LogLoss'],
    'Linear Regression': [ LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2, '-', '-', '-', '-'],
    'K-Nearest Neighbours': [ '-', '-', '-', KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score, '-'],
    'Decision Tree': [ '-', '-', '-', Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score, '-'],
    'Logistic Regression': [ '-', '-', '-', LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score, LR_Log_Loss],
    'Support Vector Machine': [ '-', '-', '-', SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score, '-'],
})

Report
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
      <th>Linear Regression</th>
      <th>K-Nearest Neighbours</th>
      <th>Decision Tree</th>
      <th>Logistic Regression</th>
      <th>Support Vector Machine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean Absolute Error</td>
      <td>0.256316</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mean Squared Error</td>
      <td>0.115723</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R2 Score</td>
      <td>0.427121</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Accuracy</td>
      <td>-</td>
      <td>0.818321</td>
      <td>0.763359</td>
      <td>0.827481</td>
      <td>0.722137</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jaccard Index</td>
      <td>-</td>
      <td>0.425121</td>
      <td>0.421642</td>
      <td>0.484018</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>F1-Score</td>
      <td>-</td>
      <td>0.59661</td>
      <td>0.593176</td>
      <td>0.652308</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LogLoss</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.380085</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>


