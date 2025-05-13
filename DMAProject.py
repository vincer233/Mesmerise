import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report # type: ignore

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Example multi-genre tag
df['genres'] = df['genre'].apply(lambda x: x.split(', '))

# Features & target
X = df[['danceability', 'energy', 'tempo', 'acousticness', 'valence']]
y = df['genres']

# Binarize genres
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
