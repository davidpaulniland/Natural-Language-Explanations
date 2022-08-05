from imports import *

def get_accuracy(pred, y_test):
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print("Accuracy", accuracy_score(y_test, pred).round(2))

# get_accuracy(predictions, y_test)

def split(X_res, y_res, test_size ):
    X_train, X_test, y_train, y_test = train_test_split(X_res,y_res, shuffle = True, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test

def permutation_function(model, X_train, X_test, y_test):
    permu = permutation_importance(model, X_test, y_test, 
                        n_repeats = 3, random_state=0)
    permu_mean = permu.importances_mean
    permu_std = permu.importances_std
    permu_dataframe = {"Variable": X_train.columns.values,"Mean Importance": permu_mean, "Standard Deviation":permu_std}
    permu_dataframe = pd.DataFrame(permu_dataframe)
    permu_dataframe = permu_dataframe.sort_values(['Mean Importance'], ascending=[False])

    sorted_idx = permu.importances_mean.argsort()
    fig, ax = plt.subplots()

    ax.boxplot(
        permu.importances[sorted_idx].T, vert = False, labels = X_train.columns[sorted_idx]
    )
    ax.set_title("Permutation Feature Importances")
    fig.tight_layout()
    plt.show()
    print(permu_dataframe)
    permu_dataframe.to_csv("rain_PFI.csv")

    return permu_dataframe

# permutation_function(model, X_train, X_test, y_test)

def rf_function(X_train, y_train, X_test, y_test, estimators):
    RF_model = RandomForestClassifier(n_estimators=estimators, random_state=1).fit(X_train, y_train)
    RF_pred = RF_model.predict(X_test)
    get_accuracy(RF_pred, y_test)
    return RF_pred, RF_model


def gb_function(X_train, y_train, X_test, y_test, estimators, lr, max_features, max_depth):
    GB_model = GradientBoostingClassifier(n_estimators=estimators, learning_rate=lr, max_features=max_features, max_depth=max_depth, random_state=0)
    GB_model.fit(X_train, y_train)
    GB_pred = GB_model.predict(X_test)
    get_accuracy(GB_pred, y_test)
    return GB_pred, GB_model 


def adab_function(X_train, y_train, X_test, y_test, estimators, lr):
    ADAB_model = AdaBoostClassifier(n_estimators=estimators,  
                                    learning_rate=lr, 
                                    random_state=101)
    ADAB_model.fit(X_train, y_train)
    ADAB_pred = ADAB_model.predict(X_test)
    get_accuracy(ADAB_pred, y_test)
    return ADAB_pred, ADAB_model
