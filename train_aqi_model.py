import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # Added 2nd Algorithm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "Air_quality_bigdata_1GB.csv"
MODEL_FILE = "aqi_model.pkl"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
df = df.dropna()

features = ["PM2.5","PM10","NO2","CO","SO2","O3"]
target = "AQI"

X = df[features]
y = df[target]

# Split data to evaluate both models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training and Comparing Models...")

# --- Algorithm 1: Random Forest ---
model_rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
model_rf.fit(X_train, y_train)
score_rf = r2_score(y_test, model_rf.predict(X_test))

# --- Algorithm 2: Linear Regression ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
score_lr = r2_score(y_test, model_lr.predict(X_test))

# --- COMPARISON LOGIC (Defining 'best_model') ---
print(f"Random Forest R2: {score_rf:.4f}")
print(f"Linear Regression R2: {score_lr:.4f}")

if score_rf > score_lr:
    print("Selecting Random Forest as the best model.")
    best_model = model_rf # <--- THIS DEFINES THE NAME
else:
    print("Selecting Linear Regression as the best model.")
    best_model = model_lr # <--- OR THIS DEFINES THE NAME

# --- SAVING ---
print("Saving model...")
with open(MODEL_FILE, "wb") as f:
    pickle.dump(best_model, f) # Now 'best_model' exists!

print(f"Model saved as {MODEL_FILE}")