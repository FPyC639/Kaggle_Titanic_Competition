<h1> Titanics Dataset Predictions </h1>
<h2> Jose M. Serra Jr. </h2>
Mounted Google Drive to import data from Kaggle Train, and Test data sets.


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
import matplotlib.pyplot as mplplt
import numpy as np
import pandas as pd
from scipy.stats import reciprocal
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
import tensorflow.keras.callbacks as tkc
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.manifold import Isomap
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.utils import resample
```

Here I included standard packages which include TensorFlow machine learning library, as well as the Sci-Kit Learn library for data processing. As well as standard imports such as Pandas for reading data, and numpy for numerial analysis.


```python
file1_train, file2_test = pd.read_csv(r"/content/drive/MyDrive/Titanic/train.csv", delimiter=","),\
pd.read_csv(r"/content/drive/MyDrive/Titanic/test.csv", delimiter= ",")
```

<p> Now I am going to encode the Sex column into a binary 1s, and 0s output Sex_Binary column.</p>


```python
file1_train = file1_train.drop(columns = ["PassengerId","Name"])
X, y = file1_train.drop("Survived", axis=1), file1_train["Survived"]
```


```python
file1_train.head()
```





  <div id="df-c7ce2438-9901-4a85-9a07-d516575df762">
    <div class="colab-df-container">
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c7ce2438-9901-4a85-9a07-d516575df762')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c7ce2438-9901-4a85-9a07-d516575df762 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c7ce2438-9901-4a85-9a07-d516575df762');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
objs = X.select_dtypes(["object"])
num  = X.select_dtypes(["number"])
```


```python
objs.isnull().sum().head()
```




    Sex           0
    Ticket        0
    Cabin       687
    Embarked      2
    dtype: int64




```python
num.isnull().sum().head()
```




    Pclass      0
    Age       177
    SibSp       0
    Parch       0
    Fare        0
    dtype: int64




```python
numerical_features = num.columns
numerical_pipeline = Pipeline(
    steps=[
            ("imputer", SimpleImputer(strategy = 'mean')),
            ("scaler", StandardScaler())
])
```


```python
for i in range(len(objs.columns)):
    print(objs.columns[i], objs.iloc[:,i].value_counts().unique())
```

    Sex [577 314]
    Ticket [7 6 5 4 3 2 1]
    Cabin [4 3 2 1]
    Embarked [644 168  77]



```python
binary = ["Sex"]
binary_pipeline = Pipeline(steps=[("binary", OneHotEncoder())])
```


```python
cat1 = ["Ticket", "Cabin", "Embarked"]
catergorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy = 'most_frequent')), ("ordinal_encoder", OrdinalEncoder()),("scaler", StandardScaler())])
```


```python
data_preprocessor = ColumnTransformer( [('numerical', numerical_pipeline, numerical_features),
                                      ('binary', binary_pipeline, binary),
                                      ('categorical', catergorical_pipeline, cat1)])
```


```python
X.head()
```





  <div id="df-5df701c6-220c-4342-86ea-1fcf2b48ab14">
    <div class="colab-df-container">
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
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5df701c6-220c-4342-86ea-1fcf2b48ab14')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5df701c6-220c-4342-86ea-1fcf2b48ab14 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5df701c6-220c-4342-86ea-1fcf2b48ab14');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
X = Pipeline(steps=[('processing',data_preprocessor)]).fit_transform(X)
```


```python
param_distribs = {'learn_rate' : np.array(np.linspace(.1,.9))}
```


```python
def base_model1(learn_rate = .1):
    input_dim = X.shape[1]
    model =Sequential([
    Dense(200 , input_dim = input_dim, activation= "relu"),
    Dropout(rate=.10),
    Dense(100, activation= "tanh"),
    Dense(1,activation = "sigmoid"),
    ])
    lr_schedule = ExponentialDecay(
    learn_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=["accuracy"])
    return model
```


```python
checkpoint = [ModelCheckpoint("Titanic.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')]
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
```


```python
NN_clf = KerasClassifier(build_fn=base_model1, epochs=100, verbose=1, callbacks =[checkpoint,early] )
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead.
      """Entry point for launching an IPython kernel.



```python
%%capture
random_trainor = GridSearchCV(estimator=NN_clf,param_grid=param_distribs, cv=None)
random_trainor.fit(X,(y.values.reshape(-1,1)))
```


```python
best = random_trainor.best_estimator_.model
```


```python
best.summary()
```

    Model: "sequential_413"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_1239 (Dense)          (None, 200)               2200      
                                                                     
     dropout_413 (Dropout)       (None, 200)               0         
                                                                     
     dense_1240 (Dense)          (None, 100)               20100     
                                                                     
     dense_1241 (Dense)          (None, 1)                 101       
                                                                     
    =================================================================
    Total params: 22,401
    Trainable params: 22,401
    Non-trainable params: 0
    _________________________________________________________________



```python
best.save("Titanic.h5")
```


```python
X_test = data_preprocessor.fit_transform(file2_test)
```


```python
PassengerId = file2_test["PassengerId"].to_list()
```


```python
final_pred = (best.predict(X_test) > 0.5).astype("int32").flatten()
```


```python
#d = {"PassengerId":PassengerId, "Survived":final_pred}
#pd.DataFrame(data=d,index=None, columns= ["PassengerId","Survived"]).to_csv("12232021.csv",index=False, header=1)
```


```python
!jupyter nbconvert "/content/drive/MyDrive/Colab Notebooks (1)/Characteristic_Path_Length.ipynb" --to markdown  --output-dir markdown
```


```python

```
