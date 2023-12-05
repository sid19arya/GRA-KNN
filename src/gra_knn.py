import data
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def get_feature_importance(x_train, y_train):

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(x_train, y_train)
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    return indices
    
def validate_GRA_KNN(k_Vals, x_train, y_train, x_val, y_val):
    
    accuracies = []

    for k in k_Vals:
        count = 0
        for i in range(len(x_val)):
            x = x_val[i]

            distance_matrix = np.argsort(abs(x - x_train), axis=0)
            k_neighbor_labels = np.array([])
            k_star = k
            for l in range(len(x)):
                if k_star <= 0: break

                distances_along_axis = np.array(distance_matrix[:,l][:k_star])

                k_neighbor_labels = np.concatenate([k_neighbor_labels,distances_along_axis])

                k_star -= 1
            
            labels = y_train[np.array(k_neighbor_labels).flatten().astype(int)]

            most_common = np.bincount(labels).argmax()
            if most_common == y_val[i]:
                count += 1
        score = count / len(x_val)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)
    
    return kVals[accuracies.index(max(accuracies))]

        

def test_KNN(k, x_train, y_train, x_test, y_test):
    count = 0
    for i in range(len(x_test)):
        x = x_test[i]

        distance_matrix = np.argsort(abs(x - x_train), axis=0)
        k_neighbor_labels = np.array([])
        k_star = k
        for l in range(len(x)):
            if k_star <= 0: break

            distances_along_axis = np.array(distance_matrix[:,l][:k_star])

            k_neighbor_labels = np.concatenate([k_neighbor_labels,distances_along_axis])

            k_star -= 1
        
        labels = y_train[np.array(k_neighbor_labels).flatten().astype(int)]

        most_common = np.bincount(labels).argmax()
        if most_common == y_test[i]:
            count += 1
    
    score = count / len(x_test)

    return score

if __name__ == '__main__':
    # Importing data:
    print("RUNNING...")
    (x_train, x_val, x_test, y_train, y_val, y_test) = data.import_data()

    
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.99)
    x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.99)
    x_val, _ ,y_val, _ =  train_test_split(x_val, y_val, test_size=0.99)

    important_features = get_feature_importance(x_train, y_train)

    N = 100  # Change this to select number of features based on IG
    selected_features = important_features[:N]
    kVals = [5,30, 50,100, 150, 200, 250, 300]
    # Apply the selection to training, val, and test data
    x_train_selected = x_train[:, selected_features]
    x_test_selected = x_test[:, selected_features]
    x_val_selected = x_val[:, selected_features]

    best_k = validate_GRA_KNN(kVals, x_train_selected, y_train, x_val_selected, y_val)
    score = test_KNN(best_k, x_train_selected, y_train, x_test_selected, y_test)
    print("-------------BEST K ----------------")
    print(f"best k={best_k}, test_accuact on that K = {score}")


