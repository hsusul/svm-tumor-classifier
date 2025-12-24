from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def main():


    # Load dataset
    data = load_breast_cancer()
    X = data.data            # (569, 30)
    y = data.target          # (569,) 0=malignant, 1=benign

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train SVM 
    model = SVC(kernel="linear")
    model.fit(X_train_scaled, y_train)

    # Predict and eval
    preds = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["malignant", "benign"]))


    # Train two SVMs and compare
    models = [
    ("linear", SVC(kernel="linear")),
    ("rbf",    SVC(kernel="rbf", gamma="scale")),
    ]

    for name, model in models:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        print("\n====", name.upper(), "====")
        print("Accuracy:", accuracy_score(y_test, preds))
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, target_names=["malignant", "benign"]))

    best_C = 1
    best_gamma = "scale"

    model = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["malignant","benign"]))

    print(f"\nBEST: C={best_C}, gamma={best_gamma}")

if __name__ == "__main__":
    main()
