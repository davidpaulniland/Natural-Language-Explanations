from imports import *
def pd_params(number_of_fits, feature_i):
    #feature_i = 7
    #number_of_fits = 6
    features = [(feature_i,)]
    return feature_i, number_of_fits, features

def partial_dependence_function(model, X_train, feature_i, features):
    PartialDependenceDisplay.from_estimator(model, X_train, features)
    pdp, axes = partial_dependence(model, X_train, [feature_i])
    tup = partial_dependence(model, X_train, [feature_i])
    return tup
#tup = partial_dependence_function(ADAB_model)


def lowess_regression_pdp(X_train, tup, number_of_fits, feature_i, target_name):    
    print("TUP________________",tup)

    y_lr_pdp = np.array(tup[0][0])
    X_lr_pdp = np.array(tup[1][0])
    print("X_lr_PDP______________",X_lr_pdp)
    print("Y lr pdp ", y_lr_pdp) 
    r_squared = 0.0
    bandwidths = np.linspace(0.001, 0.999, 999)

    for num_iter in bandwidths:
        # Model fitting
        lowess_model = lowess.Lowess()
        lowess_model.fit(X_lr_pdp,y_lr_pdp, frac=num_iter,num_fits = number_of_fits, robust_iters=8)

        # Model prediction
        x_pred_pdp = X_lr_pdp
        y_pred_pdp = lowess_model.predict(x_pred_pdp)
        current_r_squared = r2_score(y_lr_pdp, y_pred_pdp)

        if current_r_squared > r_squared:
            r_squared = current_r_squared
            best_bandwidth = num_iter

    lowess_model = lowess.Lowess()
    lowess_model.fit(X_lr_pdp,y_lr_pdp, frac=best_bandwidth,num_fits = number_of_fits, robust_iters=8)

    x_pred_pdp = X_lr_pdp
    y_pred_pdp = lowess_model.predict(x_pred_pdp)
    #y_pred_pdp = lowess_model.predict(X_lr_pdp)
    print('final R squared', r2_score(y_lr_pdp, y_pred_pdp))

    # Plot lowess & PDP 
    #sns.lineplot(x_pred_pdp, y_pred_pdp, color="red", label="LOWESS")   # This is the fit of the LOWESS
    print(y_lr_pdp)

    sns.lineplot(X_lr_pdp, y_lr_pdp, color="blue", label =target_name)   
    plt.xlabel(X_train.columns.values[feature_i])
    plt.ylabel("Partial Dependence of "+target_name)
    plt.title("Partial Dependence Plot ")
    plt.show()

    # weighting locations of each local regression model 
    weighting_locs = lowess_model.weighting_locs
    print(f'There are {len(weighting_locs)} weighting locations')
    print("weighting Locations....", weighting_locs)
    design_matrix = lowess_model.design_matrix
    return weighting_locs, design_matrix, X_lr_pdp, y_lr_pdp, x_pred_pdp, y_pred_pdp

def get_slope_coefficients(X_lr, design_matrix):
    slope_coefficients = design_matrix[:,1]
    #sns.lineplot(np.linspace(min(X_lr),max(X_lr),design_matrix.shape[0]), design_matrix[:,1])
    #plt.ylabel("Predicted Cancer Probability")
    #plt.title("Slope coefficients")
    #plt.show()
    #print("Lowess model slope coefficients")
    #print(slope_coefficients)
    return slope_coefficients
#slope_coefficients_pdp = get_slope_coefficients(X_lr_pdp,design_matrix_pdp)

def get_intercept_coefficients(X_lr, design_matrix):
    intercept_coefficients = design_matrix[:,0]
    #sns.lineplot(np.linspace(min(X_lr),max(X_lr),design_matrix.shape[0]), design_matrix[:,0])
    #plt.ylabel("Predicted Cancer Probability")
    # the intercept (often labeled the constant) is the expected mean value of Y when all X=0. 
    #plt.title("Intercept Coefficients")
    #print("Intercept Coefficients")
    print(intercept_coefficients)

    return intercept_coefficients

def plot_pdp(x_pred, y_pred, X_train, feature_i, target_name):
    sns.lineplot(x_pred, y_pred, color="red", label="LOWESS")   
    plt.xlabel(X_train.columns.values[feature_i])
    plt.ylabel("Partial Dependence of " + str(target_name))
    plt.title("PDP")
    plt.show()




#feature_i = 2
def ale_function(model, X_train, feature_i, target_name, feature_name):
    rf_ale = ALE(model.predict, feature_names = X_train.columns.values, target_names=[target_name])
    
    rf_exp = rf_ale.explain(np.array(X_train))

    plot_ale(rf_exp, features=[feature_name], fig_kw={'figwidth':12, 'figheight':8})
    #print(rf_exp['ale_values'][feature_i].reshape(27,))
    
    X_lr_ale = np.array([rf_exp['feature_values'][:][feature_i]])
    y_lr_ale = np.array([rf_exp['ale_values'][feature_i]]) 
    
    print("ALE VALUES",np.array([rf_exp['ale_values'][feature_i]]))
    print("FEATURE VALUES", np.array([rf_exp['feature_values'][:][feature_i]]))

    
    print("SHAPE_________",y_lr_ale.shape[1]) # come back to this. This will help for automating the process when changing variable
    print("X SHAPE______",X_lr_ale.shape)
    print("Y SHAPE______",y_lr_ale.shape)

    
    q = 0
    for q in range(0,500):
        try:
            y_lr_ale = y_lr_ale[0].reshape(q,)
            X_lr_ale = X_lr_ale.reshape(q,) 
        except:
            pass
    print("X SHAPE______",X_lr_ale.shape)
    print("Y SHAPE______",y_lr_ale.shape)

    #y_lr_ale = y_lr_ale[0].reshape(31,)
    #X_lr_ale = X_lr_ale.reshape(31,) 
    #X_lr_ale = X_lr_ale[0].reshape(27,)
    #print("THIS IS THE ARRAY",np.array([rf_exp['ale_values']]))
    #print("THIS IS The 1....",np.array([rf_exp['ale_values'][1]]))
    #print("Feature Values________",rf_exp['feature_values'])


    print("X", X_lr_ale)
    print("y", y_lr_ale)
    plt.show()

    return X_lr_ale, y_lr_ale

def lowess_regression_ale(feature_i, X_lr_ale, y_lr_ale, X_train, target_name, number_of_fits):

    
    features = [(feature_i,)]
    r_squared = 0.0
    bandwidths = np.linspace(0.001, 0.999, 999)

    for num_iter in bandwidths:

        # Model fitting
        lowess_model = lowess.Lowess()
        lowess_model.fit(X_lr_ale,y_lr_ale, frac=num_iter,num_fits = number_of_fits, robust_iters=20)
        
        # Model prediction
        x_pred_ale = X_lr_ale
        y_pred_ale = lowess_model.predict(x_pred_ale)
        current_r_squared = r2_score(y_lr_ale, y_pred_ale)

        if current_r_squared > r_squared:
            r_squared = current_r_squared
            best_bandwidth = num_iter
        else: 
            pass

    lowess_model = lowess.Lowess()
    lowess_model.fit(X_lr_ale,y_lr_ale, frac=best_bandwidth,num_fits = number_of_fits, robust_iters=8)

    x_pred_ale = X_lr_ale
    #y_pred_ale = lowess_model.predict(x_pred_ale) # original i think.. 
    y_pred_ale = lowess_model.predict(X_lr_ale)
    print("final R squared", r2_score(y_lr_ale, y_pred_ale))
    

    # Plot lowess & ALE
    #sns.lineplot(x_pred_ale, y_pred_ale, color="red", label="LOWESS")   # This is the fit of the LOWESS
    sns.lineplot(X_lr_ale, y_lr_ale, color="blue", label =target_name)   
    plt.xlabel(X_train.columns.values[feature_i])
    plt.ylabel("Accumulated Local Effects of " +target_name)
    plt.title("Accumulated Local Effects")
    plt.show()

    # weighting locations of each local regression model 
    weighting_locs = lowess_model.weighting_locs
    print(f'There are {len(weighting_locs)} weighting locations')

    
    design_matrix = lowess_model.design_matrix
    
    return weighting_locs, design_matrix, X_lr_ale, y_lr_ale, x_pred_ale, y_pred_ale
