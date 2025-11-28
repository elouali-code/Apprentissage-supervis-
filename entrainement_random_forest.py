import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("Chargement des données...")
try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nEntraînement du Random Forest ...")
start_time = time.time()

# n_estimators=100 : On plante 100 arbres 
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

print("\nÉvaluation...")
y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Forest : {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred))

# On regarde quelles colonnes le modèle a le plus utilisées
importances = rf_clf.feature_importances_
feature_names = X.columns

feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 des critères les plus importants (Random Forest)')
plt.xlabel('Importance (Score de Gini)')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
