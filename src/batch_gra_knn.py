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
    
def validate_GRA_KNN(kVals, x_train, y_train, x_val, y_val, top_features_indices):
    
    accuracies = []

    for k in kVals:
        count = 0
        for i in range(len(x_val)): 
            x = x_val[i]
            k_neighbor_labels = np.array([])
            k_star = k
            for l in range(len(selected_features)):
                if k_star <= 0: break
                
                imp_feature_neighbours = top_features_indices[l] # THIS IS FOR COVARIANCE

                distances = np.linalg.norm(x[imp_feature_neighbours] - x_train[:,imp_feature_neighbours], axis=1)
                distances_along_axis = np.argsort(distances)[:k_star]
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
        k_neighbor_labels = np.array([])
        k_star = k
        for l in range(len(selected_features)):
            if k_star <= 0: break
            # distances = [np.linalg.norm(x - data_point) for data_point in x_train]
            imp_feature_neighbours = top_features_indices[l] # THIS IS FOR COVARIANCE
            # imp_feature_neighbours = np.array([l, l+1, l+27, l+28, l+29, l-1, l-27, l-28, l-29]) # this is for neighbour pixesl
            # mask = imp_feature_neighbours <= 783
            # imp_feature_neighbours = imp_feature_neighbours[mask]
            distances = np.linalg.norm(x[imp_feature_neighbours] - x_train[:,imp_feature_neighbours], axis=1)
            distances_along_axis = np.argsort(distances)[:k_star]
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

    
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.98)
    x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.98)
    x_val, _ ,y_val, _ =  train_test_split(x_val, y_val, test_size=0.98)

    important_features = get_feature_importance(x_train, y_train)

    N = 100  # Change this to select number of features based on IG
    selected_features = important_features[:N]
    kVals = [5, 10, 15, 20, 25, 30, 35, 40, 45,50]
    # Apply the selection to training, val, and test data
    x_train_selected = x_train[:, selected_features]
    x_test_selected = x_test[:, selected_features]
    x_val_selected = x_val[:, selected_features]

    covariance_matrix = np.cov(x_train_selected, rowvar=False)
    num_top_features = 20
    top_features_indices = np.argsort(-np.abs(covariance_matrix), axis=1)[:, 1:num_top_features + 1]

    best_k = validate_GRA_KNN(kVals, x_train_selected, y_train, x_val_selected, y_val, top_features_indices)
    score = test_KNN(best_k, x_train_selected, y_train, x_test_selected, y_test)
    print("-------------BEST K ----------------")
    print(f"best k={best_k}, test_accuact on that K = {score}")


