<H3>ENTER YOUR NAME : K S Vinay Suhirthan</H3>
<H3>ENTER YOUR REGISTER NO. 212224230305</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 27-07-2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```Python
# NAME : K S VINAY SUHIRTHAN
# REG.NO: 212224230305

#importing libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

df = pd.read_csv(r"C:\Users\ksvin\Downloads\crop_yield.csv")

#-------------------------------------------------------------------------------------
print(df.info())
#-------------------------------------------------------------------------------------
#checking for null value and duplicate values
print(df.isnull().sum())
print(df.duplicated().sum())
#-------------------------------------------------------------------------------------
#outlier detection
Q1 = df["Annual_Rainfall"].quantile(0.25)
Q3 = df["Annual_Rainfall"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Annual_Rainfall"] < lower) | (df["Annual_Rainfall"] > upper)]

print("Number of outliers:", len(outliers))
#-------------------------------------------------------------------------------------
#outlier removal
Q1 = df["Annual_Rainfall"].quantile(0.25)
Q3 = df["Annual_Rainfall"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Annual_Rainfall"] >= lower) & (df["Annual_Rainfall"] <= upper)]
print(df.head(2))
#-------------------------------------------------------------------------------------
#scanlling data


scaler = MinMaxScaler()

df[["Production_Scaled", "Area_Scaled", "Annual_Rainfall_Scaled"]] = scaler.fit_transform(
    df[["Production", "Area", "Annual_Rainfall"]]
)
#-------------------------------------------------------------------------------------
print(df[["Production", "Production_Scaled",
          "Area", "Area_Scaled",
          "Annual_Rainfall", "Annual_Rainfall_Scaled"]].head())
#-------------------------------------------------------------------------------------
#finding the best encoding class
print(df["Crop"].unique())
print(df["Season"].unique())
print(df["State"].unique())
#-------------------------------------------------------------------------------------

#label encoding

le = LabelEncoder()

df["Season"] = le.fit_transform(df["Season"])
#-------------------------------------------------------------------------------------
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)
#-------------------------------------------------------------------------------------
print(df.head())
#-------------------------------------------------------------------------------------
# normalising pesticide column
df["Pesticide_Normalized"] = MinMaxScaler().fit_transform(df[["Pesticide"]])

print(df[["Pesticide", "Pesticide_Normalized"]].head())
#-------------------------------------------------------------------------------------
#Splitting the data into test and train
X = df.drop("Yield", axis=1)
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
```


## OUTPUT:
### DATASET:
<img width="571" height="512" alt="image" src="https://github.com/user-attachments/assets/ba48789e-deef-4202-9d97-194d878c0607" />


### NULL VALUES: 
<img width="537" height="408" alt="image" src="https://github.com/user-attachments/assets/244d3b8a-4cc5-4dc1-a684-d7bcbea14eba" />


### NORMALIZED DATA:
<img width="1107" height="530" alt="image" src="https://github.com/user-attachments/assets/e4cddde0-bd8d-45b3-a2e4-ef8d2b3789f5" />

### LABEL ENCODING:

<img width="976" height="570" alt="image" src="https://github.com/user-attachments/assets/ef6aa1a6-7aaf-4023-a122-1536bca58dfe" />


### TRAIN AND TEST DATA:
<img width="698" height="367" alt="image" src="https://github.com/user-attachments/assets/0fadef62-a1fe-4fea-8d48-c6989f802ca9" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python using a data set downloaded from Kaggle.


