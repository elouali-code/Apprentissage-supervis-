import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
from xgboost import XGBClassifier

print("Chargement des données...")
try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nEntraînement du XGBoost ...")
start_time = time.time()

# Un modèle XGBoost standard, rapide et stable
xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"   # indispensable avec les dernières versions
)

xgb_clf.fit(X_train, y_train)

print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

print("\nÉvaluation test")
y_pred = xgb_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy XGBoost test: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred))

print("\nÉvaluation entraînement")
y_pred_train = xgb_clf.predict(X_train)

accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Accuracy XGBoost entraînement: {accuracy_train:.2%}")
print("-" * 30)
print(classification_report(y_train, y_pred_train))


# -------------------------------
# Importances des features (XGBoost)
# -------------------------------

importances = xgb_clf.feature_importances_
feature_names = X.columns

feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 des critères les plus importants (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_xgb.png')