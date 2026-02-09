#EX.NO:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
   # Step 1: Import Required Libraries

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Read the Dataset

# Replace with your actual CSV file
df1 = pd.read_csv('Data_set.csv')
df1.head()

# Step 3: Dataset Information
df1.info()
df1.describe()

# Step 4: Handling Missing Values
# Check Null Values
df1.isnull()
df1.isnull().sum()

# Fill Missing Values with 0
df1_fill_0 = df1.fillna(0)
df1_fill_0

# Forward Fill
df1_ffill = df1.ffill()
df1_ffill

# Backward Fill
df1_bfill = df1.bfill()
df1_bfill

# Fill with Mean (Numerical Column Example)

df1['num_episodes'] = df1['num_episodes'].fillna(df1['num_episodes'].mean())
df1

# Drop Missing Values
df1_dropna = df1.dropna()
df1_dropna

#Step 5: Save Cleaned Data
df1_dropna.to_csv('Data_set_new.csv', index=False)

# OUTLIER DETECTION
# Step 6: IQR Method (Using  Data_set_new)
da = pd.read_csv('Data_set_new.csv')
da.head()
da.info()
da.describe() 

#Boxplot for Outlier Detection
sns.boxplot(x=da['num_episodes'])
plt.show()

# Calculate IQR
Q1 = da['num_episodes'].quantile(0.25)
Q3 = da['num_episodes'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:", IQR)

# Detect Outliers
outliers_iqr = da[
    (da['num_episodes'] < (Q1 - 1.5 * IQR)) |
    (da['num_episodes'] > (Q3 + 1.5 * IQR))
]
outliers_iqr

# Remove Outliers
da_cleaned = da[
    ~((da['num_episodes'] < (Q1 - 1.5 * IQR)) |
      (da['num_episodes'] > (Q3 + 1.5 * IQR)))
]
da_cleaned

# Step 7: Z-Score Method

data = [1,12,15,18,21,24,27,30,33,36,39,42,45,48,51,
        54,57,60,63,66,69,72,75,78,81,84,87,90,93]

df_z = pd.DataFrame(data, columns=['values'])
df_z

# Calculate Z-Scores
z_scores = np.abs(stats.zscore(df_z))
z_scores

# Detect Outliers
threshold = 3
outliers_z = df_z[z_scores > threshold]
print("Outliers:")
outliers_z

# Remove Outliers
df_z_cleaned = df_z[z_scores <= threshold]
df_z_cleaned 
![WhatsApp Image 2026-02-09 at 11 51 40 AM](https://github.com/user-attachments/assets/6520fe73-503e-4745-96bb-27f9f77f8633)
![WhatsApp Image 2026-02-09 at 11 51 41 AM](https://github.com/user-attachments/assets/8e69124c-05de-433b-89b2-67cf41cc7bd5)
![WhatsApp Image 2026-02-09 at 11 51 41 AM (1)](https://github.com/user-attachments/assets/b0b00077-3c5b-430e-975d-de4134caa4e9)


# Result
Thus the data cleaning process using python is successfully completed
