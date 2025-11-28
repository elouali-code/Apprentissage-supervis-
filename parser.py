import csv
import random

PATH1 = "X_processed.csv"
PATH2 = "y_labels.csv"

def parse_dataset_features(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert values to float (ou int si tu préfères)
            parsed_row = {k: float(v) for k, v in row.items() if k != 'OCCP'}
            data.append(parsed_row)
    return data

def parse_dataset_labels(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert values to float (ou int si tu préfères)
            parsed_row = {k: 0 if v == "False" else 1 for k, v in row.items()}
            data.append(parsed_row)
    return data

def train_test_split_custom(X, y, test_ratio=0.2, seed=42):
    random.seed(seed)

    # Combiner X et y pour mélanger ensemble
    combined = list(zip(X, y))
    random.shuffle(combined)

    # Re-séparer X et y
    X_shuffled, y_shuffled = zip(*combined)

    # Split
    split_point = int(len(X) * (1 - test_ratio))
    X_train = X_shuffled[:split_point]
    X_test  = X_shuffled[split_point:]
    y_train = y_shuffled[:split_point]
    y_test  = y_shuffled[split_point:]

    return list(X_train), list(X_test), list(y_train), list(y_test)


# -------------------------------
# Exemple d'utilisation
# -------------------------------

X = parse_dataset_features(PATH1)
y = parse_dataset_labels(PATH2)

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_ratio=0.2)



print("Train size:", len(X_train))
print("Test size:", len(X_test))
