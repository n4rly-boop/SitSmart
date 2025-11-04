import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    recall_score,
    precision_score,
    f1_score
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

path = 'C:/Users/artem/Study/PMLaDL/SitSmart/app/ml_training/data/features.csv'

data = pd.read_csv(path)
print(data.head())

print(data.isnull().sum())

data = data.dropna()

angle_cols = ['shoulder_line_angle_deg', 'head_tilt_deg']
data[angle_cols] = data[angle_cols].abs()
y = data["label"]
X = data.drop(["image_name", "label"], axis=1).values

print(f"\nThe size of the data after deleting the NaN: {data.shape}")

print("\nClass distribution:")
print(data['label'].value_counts())

y = data["label"]
X = data.drop(["image_name", "label"], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=91
)

preprocessor = StandardScaler()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['sag', 'saga'],
    'class_weight': [
        None,
        'balanced',
        {'good': 1, 'bad': 2},
        {'good': 1, 'bad': 3},
        {'good': 1, 'bad': 1.5}
    ]
}

classifier_model = LogisticRegression(max_iter=3000, random_state=91)

grid_search = GridSearchCV(
    classifier_model,
    param_grid,
    cv=12,
    scoring='recall_macro',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train_processed, y_train)

print(f"\nThe best parameters: {grid_search.best_params_}")
print(f"Best CV Recall Score: {grid_search.best_score_:.4f}")

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test_processed)
y_pred_proba = best_classifier.predict_proba(X_test_processed)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['good', 'bad'])
print(cm)

print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=["good_posture", "bad_posture"], output_dict=True)
print(classification_report(y_test, y_pred, target_names=["good_posture", "bad_posture"]))

recall_good = report['good_posture']['recall']
recall_bad = report['bad_posture']['recall']
precision_good = report['good_posture']['precision']
precision_bad = report['bad_posture']['precision']
f1_good = report['good_posture']['f1-score']
f1_bad = report['bad_posture']['f1-score']

print(f"\nDetailed metrics:")
print(f"Recall for good_posture: {recall_good:.4f}")
print(f"Recall for bad_posture: {recall_bad:.4f}")
print(f"Mean recall: {(recall_good + recall_bad) / 2:.4f}")

class_labels = best_classifier.classes_
bad_class_idx = np.where(class_labels == 'bad')[0][0]

fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
disp = ConfusionMatrixDisplay(cm, display_labels=["good_posture", "bad_posture"])
disp.plot(cmap='RdYlGn_r', ax=ax1, colorbar=False)
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 2. ROC Curve (для класса 'bad')
ax2 = plt.subplot(2, 3, 2)
fpr, tpr, thresholds_roc = roc_curve(
    (y_test == 'bad').astype(int),
    y_pred_proba[:, bad_class_idx]
)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax2)
ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax2.set_title(f'ROC Curve - bad_posture (AUC = {roc_auc:.3f})', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# 3. Precision-Recall Curve (для класса 'bad')
ax3 = plt.subplot(2, 3, 3)
precision, recall, thresholds_pr = precision_recall_curve(
    (y_test == 'bad').astype(int),
    y_pred_proba[:, bad_class_idx]
)
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_display.plot(ax=ax3)
ax3.set_title('Precision-Recall Curve - bad_posture', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Comparing metrics by class
ax4 = plt.subplot(2, 3, 4)
metrics_data = {
    'good_posture': [recall_good, precision_good, f1_good],
    'bad_posture': [recall_bad, precision_bad, f1_bad]
}
x = np.arange(3)
width = 0.35
labels = ['Recall', 'Precision', 'F1-Score']

bars1 = ax4.bar(x - width / 2, metrics_data['good_posture'], width,
                label='Good Posture', alpha=0.8, color='#2ecc71')
bars2 = ax4.bar(x + width / 2, metrics_data['bad_posture'], width,
                label='Bad Posture', alpha=0.8, color='#e74c3c')

ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Metrics by class', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.legend()
ax4.set_ylim([0, 1.1])
ax4.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax5 = plt.subplot(2, 3, 5)
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values('rank_test_score').head(10)

y_pos = np.arange(len(cv_results_sorted))
scores = cv_results_sorted['mean_test_score'].values

colors = plt.cm.RdYlGn(scores / scores.max())
bars = ax5.barh(y_pos, scores, alpha=0.8, color=colors)
ax5.set_yticks(y_pos)
ax5.set_yticklabels([f"Config {i + 1}" for i in range(len(cv_results_sorted))])
ax5.invert_yaxis()
ax5.set_xlabel('Mean CV Recall Score', fontsize=12)
ax5.set_title('Top 10 GridSearch', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

for i, (bar, score) in enumerate(zip(bars, scores)):
    ax5.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9)

ax6 = plt.subplot(2, 3, 6)
thresholds_plot = np.linspace(0, 1, 100)
precisions_list = []
recalls_list = []
f1_scores_list = []

for threshold in thresholds_plot:
    y_pred_threshold = (y_pred_proba[:, bad_class_idx] >= threshold)
    y_pred_threshold_labels = np.where(y_pred_threshold, 'bad', 'good')

    if len(np.unique(y_pred_threshold_labels)) > 1:
        prec = precision_score(y_test, y_pred_threshold_labels, pos_label='bad', zero_division=0)
        rec = recall_score(y_test, y_pred_threshold_labels, pos_label='bad', zero_division=0)
        f1 = f1_score(y_test, y_pred_threshold_labels, pos_label='bad', zero_division=0)
        precisions_list.append(prec)
        recalls_list.append(rec)
        f1_scores_list.append(f1)
    else:
        precisions_list.append(0)
        recalls_list.append(1 if y_pred_threshold[0] else 0)
        f1_scores_list.append(0)

ax6.plot(thresholds_plot, recalls_list, label='Recall', linewidth=2.5, color='#3498db')
ax6.plot(thresholds_plot, precisions_list, label='Precision', linewidth=2.5, color='#e67e22')
ax6.plot(thresholds_plot, f1_scores_list, label='F1-Score', linewidth=2,
         linestyle='--', color='#9b59b6', alpha=0.7)
ax6.axvline(x=0.5, color='red', linestyle='--', alpha=0.5,
            linewidth=2, label='Default Threshold (0.5)')

optimal_recall_idx = np.argmax(np.array(recalls_list) * (np.array(precisions_list) > 0.5))
optimal_threshold = thresholds_plot[optimal_recall_idx]
ax6.scatter(optimal_threshold, recalls_list[optimal_recall_idx],
            color='green', s=150, zorder=5, marker='*',
            label=f'Optimal (t={optimal_threshold:.2f})')

ax6.set_xlabel('Classification threshold', fontsize=12)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Metrics at different thresholds (bad_posture)', fontsize=14, fontweight='bold')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('C:/Users/artem/Study/PMLaDL/SitSmart/app/ml_training/training_visualization.png',
            dpi=300, bbox_inches='tight')
print("\nVisualization saved in training_visualization.png")
plt.show()

fig2, axes = plt.subplots(2, 2, figsize=(18, 12))

ax1 = axes[0, 0]
cv_results_full = pd.DataFrame(grid_search.cv_results_)

cv_results_valid = cv_results_full[~cv_results_full['mean_test_score'].isna()]

for solver in ['sag', 'saga']:
    solver_data = cv_results_valid[cv_results_valid['param_solver'] == solver]
    grouped = solver_data.groupby('param_C')['mean_test_score'].mean().reset_index()
    grouped = grouped.sort_values('param_C')
    ax1.plot(grouped['param_C'].astype(str), grouped['mean_test_score'],
             marker='o', label=f'Solver: {solver}', linewidth=2.5, markersize=8)

ax1.set_xlabel('C (Regularization parameter)', fontsize=12)
ax1.set_ylabel('Mean CV Recall Score', fontsize=12)
ax1.set_title('The effect of the C parameter on Recall', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2 = axes[0, 1]

def format_class_weight(cw):
    if cw is None:
        return 'None'
    elif cw == 'balanced':
        return 'Balanced'
    elif isinstance(cw, dict):
        return str(cw)
    else:
        return str(cw)


cv_results_valid['class_weight_str'] = cv_results_valid['param_class_weight'].apply(format_class_weight)

class_weight_effects = cv_results_valid.groupby('class_weight_str')['mean_test_score'].agg(
    ['mean', 'std']).reset_index()
class_weight_effects = class_weight_effects.sort_values('mean', ascending=False)

y_pos = np.arange(len(class_weight_effects))
bars = ax2.barh(y_pos, class_weight_effects['mean'].values,
                xerr=class_weight_effects['std'].values,
                alpha=0.8, capsize=5, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(y_pos))))

ax2.set_yticks(y_pos)
ax2.set_yticklabels(class_weight_effects['class_weight_str'].values)
ax2.invert_yaxis()
ax2.set_xlabel('Mean CV Recall Score', fontsize=12)
ax2.set_title('The effect of class_weight on Recall', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

for i, (bar, mean_val) in enumerate(zip(bars, class_weight_effects['mean'].values)):
    ax2.text(mean_val + 0.005, i, f'{mean_val:.4f}', va='center', fontsize=10, fontweight='bold')

ax3 = axes[1, 0]
all_scores = cv_results_valid['mean_test_score'].values
ax3.hist(all_scores, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
ax3.axvline(grid_search.best_score_, color='red', linestyle='--',
            linewidth=2.5, label=f'Best Score: {grid_search.best_score_:.4f}')
ax3.axvline(all_scores.mean(), color='green', linestyle='--',
            linewidth=2, label=f'Mean Score: {all_scores.mean():.4f}')
ax3.set_xlabel('CV Recall Score', fontsize=12)
ax3.set_ylabel('fontsize', fontsize=12)
ax3.set_title('Distribution Recall Scores', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

ax4 = axes[1, 1]
top_n = 15
top_configs = cv_results_valid.sort_values('rank_test_score').head(top_n)

x_pos = np.arange(top_n)
width = 0.35

bars1 = ax4.bar(x_pos - width / 2, top_configs['mean_train_score'].values,
                width, label='Train Score', alpha=0.8, color='#2ecc71')
bars2 = ax4.bar(x_pos + width / 2, top_configs['mean_test_score'].values,
                width, label='CV Score', alpha=0.8, color='#e74c3c')

ax4.set_xlabel('Configuration', fontsize=12)
ax4.set_ylabel('Recall Score', fontsize=12)
ax4.set_title('Train vs CV Recall for the best models', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{i + 1}' for i in range(top_n)])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('C:/Users/artem/Study/PMLaDL/SitSmart/app/ml_training/hyperparameter_analysis.png',
            dpi=300, bbox_inches='tight')
print("Hyperparameter analysis is saved in hyperparameter_analysis.png")
plt.show()

model_path = 'C:/Users/artem/Study/PMLaDL/SitSmart/app/models/model.pkl'
scaler_path = 'C:/Users/artem/Study/PMLaDL/SitSmart/app/models/scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(best_classifier, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"\nModel saved in {model_path}")
print(f"Scaler saved in {scaler_path}")

metrics_summary = {
    'best_params': str(grid_search.best_params_),
    'best_cv_recall': grid_search.best_score_,
    'test_recall_good_posture': recall_good,
    'test_recall_bad_posture': recall_bad,
    'test_precision_good_posture': precision_good,
    'test_precision_bad_posture': precision_bad,
    'test_f1_good_posture': f1_good,
    'test_f1_bad_posture': f1_bad,
    'roc_auc_bad_posture': roc_auc,
    'test_accuracy': report['accuracy']
}

metrics_df = pd.DataFrame([metrics_summary])
metrics_df.to_csv('C:/Users/artem/Study/PMLaDL/SitSmart/app/ml_training/metrics_summary.csv',
                  index=False)

print(f"\nFinal Recall for bad_posture: {recall_bad:.4f}")
print(f"Final Recall for good_posture: {recall_good:.4f}")
print(f"Mean Recall: {(recall_good + recall_bad) / 2:.4f}")