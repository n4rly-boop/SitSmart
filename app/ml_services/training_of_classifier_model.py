import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

path = 'C:/Users/artem/Study/PMLaDL/SitSmart/app/data/features.csv'

data = pd.read_csv(path)
print(data.head())
data = data.dropna()
print(data.shape)
y = data["label"]
X = data.drop(["image_name", "label"], axis=1).values

data = pd.read_csv(path)
print(data.head())
print("\nNans:")
print(data.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=91)

preprocessor = StandardScaler()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['sag', 'saga'],
}

classifier_model = LogisticRegression(max_iter=3000, random_state=91)

grid_search = GridSearchCV(
    classifier_model,
    param_grid,
    cv=12,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1)

grid_search.fit(X_train_processed, y_train)

print("Best Params:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test_processed)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=["good posture", "bad_posture"]))

# Visualisation
ConfusionMatrixDisplay(cm, display_labels=["good posture", "bad_posture"]).plot(cmap='autumn')

model_path = 'C:/Users/artem/Study/PMLaDL/SitSmart/app/models/model.pkl'
scaler_path = 'C:/Users/artem/Study/PMLaDL/SitSmart/app/models/scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(best_classifier, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")