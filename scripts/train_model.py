import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from features.bert_surprisal import sentence_surprisal

df = pd.read_csv("data/features.csv")

df["surprisal"] = df["text"].apply(sentence_surprisal)


X = df.drop("label", axis=1)
y = df["label"]

# 🔹 Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save everything needed
joblib.dump({
    "model": model,
    "feature_names": feature_names,
    "label_encoder": label_encoder
}, "models/difficulty.pkl")

print("Model retrained with numeric labels.")
