import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

print("📂 Loading datasets...")
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

# Label the data
fake["label"] = "fake"
true["label"] = "real"

# Balance the dataset
min_len = min(len(fake), len(true))
fake = fake.sample(n=min_len, random_state=42)
true = true.sample(n=min_len, random_state=42)

# Combine both
df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ Datasets loaded successfully!")

# Combine title and text
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Split
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# TF-IDF
print("⚙️ Initializing TF-IDF vectorizer...")
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2))
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# Train
print("🧠 Training PassiveAggressiveClassifier...")
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(tfidf_train, y_train)

# Evaluate
print("🧮 Evaluating model...")
y_pred = model.predict(tfidf_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save
print("💾 Saving model files...")
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
print("🎉 Model and vectorizer saved successfully!")
