
# ✈️ Flight Delay Prediction using AWS Data Lake and Machine Learning

This project builds a fully scalable **Data Lake + ML pipeline on AWS** to predict flight delays based on historical flight data from 2019–2023.  
The pipeline includes ingestion, transformation, querying, machine learning model training, and dashboard visualization.

---

## 📦 Dataset

- **Source:** [Flight Delay and Cancellation Data (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
- **Format:** CSV
- **Size:** ~2.5 GB

---

## 🏗️ Architecture & Workflow Overview

1. ✅ **Amazon S3** for raw/processed/result storage  
2. ✅ **AWS Glue** for schema inference and cataloging  
3. ✅ **Athena** for serverless querying  
4. ✅ **Lambda** for ETL (triggered on upload)  
5. ✅ **SageMaker Studio Lab** for model training  
6. ✅ **QuickSight** for ML result dashboards  

---

## 🧱 Step-by-Step Pipeline

### 🔹 Step 1: Set Up S3 Data Lake

- **S3 Bucket:** `flight-data-lake`
- **Folder Structure:**
  ```
  flight-data-lake/
  ├── raw/
  ├── processed/
  └── results/
  ```
- Uploaded raw CSV files into the `raw/` directory

---

### 🔹 Step 2: Catalog with AWS Glue

- Created a **Glue database**: `flight_data_catalog`
- Set up a **Crawler** to scan `raw/` and infer schema
- Ran crawler → Made data queryable in Athena

---

### 🔹 Step 3: Query with Amazon Athena

- Set Athena output to: `s3://flight-data-lake/results/`
- Example Query:
  ```sql
  SELECT carrier, COUNT(*) AS total_flights
  FROM flight_data_catalog.flight_data
  GROUP BY carrier;
  ```

---

### 🔹 Step 4: Data Processing with AWS Lambda

- **Trigger:** New files in `raw/`
- **Function:** Clean + transform → Save to `processed/`

```python
import boto3, pandas as pd, io

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    df_clean = df.dropna()
    out_buffer = io.BytesIO()
    df_clean.to_csv(out_buffer, index=False)
    
    s3.put_object(Bucket=bucket, Key=key.replace('raw/', 'processed/'), Body=out_buffer.getvalue())
```

---

### 🔹 Step 5: Train ML Model in SageMaker Studio Lab

- Downloaded cleaned dataset from `processed/`
- Uploaded to **SageMaker Studio Lab**
- Trained **Random Forest Classifier** on binary target:
  - `arr_delay > 15` ➝ `1 (Delayed)` else `0 (On Time)`

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("flight_data_cleaned.csv")
X = df[['crs_dep_time', 'dep_delay', 'air_time', 'distance']]
y = (df['arr_delay'] > 15).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

✅ Model trained with >90% accuracy  
📁 Exported results: `predictions.csv`

---

### 🔹 Step 6: Visualize in Amazon QuickSight

- Connected QuickSight to `flight-data-lake/results/`
- Uploaded `predictions.csv`
- Built dashboards to show:
  - Model performance
  - Flight delay breakdowns by carrier, airport, etc.

---

## 🎯 Outcomes

- Fully automated **data ingestion to model prediction**
- End-to-end pipeline uses **only serverless/free-tier AWS tools**
- Dashboard gives real-time insight into prediction results





## 🔮 Future Improvements

- Move model training to AWS SageMaker pipelines
- Automate dashboard refresh using Glue job triggers
- Integrate real-time streaming using Kinesis

---
