import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

path = r"..\app\data\dataset.csv"

data = pd.read_csv(path)
y = data["label"]
X = data.drop(["label"], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=91)

preprocessor = StandardScaler()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['sag', 'saga'],
}

classifier_model = LogisticRegression(max_iter=2000, random_state=91)

grid_search = GridSearchCV(
    classifier_model,
    param_grid,
    cv=8,
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

with open('../app/data/models/model.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)
with open('../app/data/models/scaler.pkl', 'wb') as f:
    scaler = pickle.load(f)

def get_prediction(input_data):
    scaled = scaler.transform(input_data)
    return best_classifier.predict(scaled)