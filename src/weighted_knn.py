import data
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier


def get_feature_importance(x_train, y_train):

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(x_train, y_train)
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    return indices
    
def validate_KNN(k_Vals, x_train, y_train, x_val, y_val):
    
    accuracies = []

    for k in k_Vals:
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(x_train, y_train)
        score = model.score(x_val, y_val)
        print(f"k={k}, accuracy={score * 100:.2f}%")
        accuracies.append(score)
    
    return kVals[accuracies.index(max(accuracies))]

        

def test_KNN(k, x_train, y_train, x_test, y_test):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(x_train, y_train)

    return model.score(x_test, y_test)

if __name__ == '__main__':
    # Importing data:
    print("RUNNING...")
    (x_train, x_val, x_test, y_train, y_val, y_test) = data.import_data()

    important_features = get_feature_importance(x_train, y_train)

    N = 100  # Change this to select number of features based on IG
    selected_features = important_features[:N]
    kVals = [1, 5, 10, 20, 30]
    # Apply the selection to training, val, and test data
    x_train_selected = x_train[:, selected_features]
    x_test_selected = x_test[:, selected_features]
    x_val_selected = x_val[:, selected_features]

    best_k = validate_KNN(kVals, x_train_selected, y_train, x_val_selected, y_val)
    score = test_KNN(best_k, x_train_selected, y_train, x_test_selected, y_test)
    print("-------------BEST K ----------------")
    print(f"best k={best_k}, test_accuact on that K = {score}")


