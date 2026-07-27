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
![image](https://github.com/user-attachments/assets/4e75215a-6909-47e8-be89-f1b0dcf64069)

### NULL VALUES: 
![image](https://github.com/user-attachments/assets/1a4ab591-3115-42e5-88c6-c2356d1175b5)

### NORMALIZED DATA:
![image](https://github.com/user-attachments/assets/27a5d162-c488-42b5-b5b9-fe563cf3062c)
### DATA SPLITTING:
![image](https://github.com/user-attachments/assets/e5154e4d-b3f4-444e-9d20-5a3d908c4797)
![image](https://github.com/user-attachments/assets/6d65b734-6632-4fa7-aa91-39c6feb88e5e)

### TRAIN AND TEST DATA:
![image](https://github.com/user-attachments/assets/8716f8a6-4f08-42ea-9c6a-1b86bb38ae00)
![image](https://github.com/user-attachments/assets/4214bcfc-f2c1-4f69-b7b0-6ead12703483)
![image](https://github.com/user-attachments/assets/f219a314-14c7-4921-a631-df59b3c46301)
![image](https://github.com/user-attachments/assets/025abc61-60a1-4f7d-b8ca-40210b03f7f4)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python using a data set downloaded from Kaggle.


