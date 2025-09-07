parsaenergy_repo/
│── README.md
│── main.py
│── requirements.txt
└── data/
    └── sample_data.csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load sample dataset
data = pd.read_csv("data/sample_data.csv")

# Features and target
X = data.drop("Safe_Water", axis=1)
y = data["Safe_Water"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
pandas
scikit-learn
numpy
pH,Turbidity,TDS,BOD,COD,Safe_Water
7.0,3,500,2,20,1
6.5,5,800,4,40,0
7.2,2,450,1,15,1
8.0,6,1000,5,50,0

