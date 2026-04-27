
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

# Regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# Time Series
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



# PART 1: MACHINE LEARNING (Breast Cancer Dataset)


print("\n=========== MACHINE LEARNING PART ===========")

df = pd.read_csv("Breast_Cancer.csv")
df.columns = df.columns.str.strip()

# Preprocessing
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df.drop_duplicates(inplace=True)

# Encoding
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Normalization
for col in df.select_dtypes(include=np.number).columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())


# REGRESSION

print("\n--- REGRESSION ---")

X = df[['Age','Tumor Size','Survival Months']]
y = df['Reginol Node Positive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models_reg = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor()
}

for name, model in models_reg.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MAE:", mean_absolute_error(y_test, pred))



# CLASSIFICATION

print("\n--- CLASSIFICATION ---")

X = df[['Age','Tumor Size','Survival Months']]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models_clf = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "NaiveBayes": GaussianNB()
}

for name, model in models_clf.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, pred))



# CLUSTERING

print("\n--- CLUSTERING ---")

X = df[['Age','Tumor Size','Survival Months']]

kmeans = KMeans(n_clusters=3)
df['KMeans'] = kmeans.fit_predict(X)

agg = AgglomerativeClustering(n_clusters=3)
df['Agglomerative'] = agg.fit_predict(X)

plt.scatter(X['Age'], X['Tumor Size'], c=df['KMeans'])
plt.title("KMeans Clustering")
plt.show()



#  TIME SERIES (Gold Price Dataset)


print("\n=========== TIME SERIES PART ===========")

ts_df = pd.read_csv("GoldPrice.csv")

ts_df['Date'] = pd.to_datetime(ts_df['Date'], dayfirst=True)
ts_df = ts_df.sort_values('Date')
ts_df.set_index('Date', inplace=True)

# Plot
plt.plot(ts_df['Price'])
plt.title("Gold Price Trend")
plt.show()

# ---- AutoReg ----
model_ar = AutoReg(ts_df['Price'], lags=3)
model_ar_fit = model_ar.fit()
pred_ar = model_ar_fit.predict(start=len(ts_df), end=len(ts_df)+5)

# ---- ARIMA ----
model_arima = ARIMA(ts_df['Price'], order=(1,1,1))
model_arima_fit = model_arima.fit()
pred_arima = model_arima_fit.forecast(steps=5)

# ---- SARIMA ----
model_sarima = SARIMAX(ts_df['Price'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_sarima_fit = model_sarima.fit()
pred_sarima = model_sarima_fit.forecast(steps=5)

print("\nAutoReg:", pred_ar)
print("\nARIMA:", pred_arima)
print("\nSARIMA:", pred_sarima)



# PART 3: NLP (Text Modeling + Classification)


print("\n=========== NLP PART ===========")

# Create text column
df['Text'] = df['Grade'].astype(str) + " " + df['Status'].astype(str)

X_text = df['Text']
y = df['Status']

vectorizer = TfidfVectorizer()
X_text_vec = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X_text_vec, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("NLP Accuracy:", accuracy_score(y_test, pred))