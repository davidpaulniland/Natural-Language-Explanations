from imports import * 
import functions

def DataCleaning():
    nba_data = pd.read_csv("nba_logreg.csv")
    print(nba_data.head)

    print(nba_data.info())

    y = pd.DataFrame()
    y['Career > 5 Years'] = nba_data['TARGET_5Yrs']
    X = nba_data.drop('TARGET_5Yrs', axis = 1)
    X = X.drop('3P%', axis = 1)
    X = X.drop('Name', axis = 1)
    print(X.columns)
    

    X.rename(columns = {'GP': 'Games played rookie season'}, inplace = True)
    X.rename(columns = {'MIN': 'Avg minutes played per game'}, inplace = True)
    X.rename(columns = {'PTS': 'Avg points per game'}, inplace = True)
    X.rename(columns = {'FGM': 'Avg field goals made per game'}, inplace = True)
    X.rename(columns = {'FGA': 'Avg field goals attempted per game'}, inplace = True)
    X.rename(columns = {'FG%': 'Avg field goal percent'}, inplace = True)
    X.rename(columns = {'3P Made': 'Avg three-pointers scored'}, inplace = True)
    X.rename(columns = {'3PA': 'Avg three-pointers attempted'}, inplace = True)
    X.rename(columns = {'FTM': 'Avg free throws made per game'}, inplace = True)
    X.rename(columns = {'FTA': 'Avg free throws attempted per game'}, inplace = True)
    X.rename(columns = {'FT%': 'Free throw percent'}, inplace = True)
    X.rename(columns = {'OREB': 'Avg offensive rebounds per game'}, inplace = True)
    X.rename(columns = {'DREB': 'Avg defensive rebound per game'}, inplace = True)
    X.rename(columns = {'REB': 'Avg total rebounds per game'}, inplace = True)
    X.rename(columns = {'AST': 'Avg assists per game'}, inplace = True)
    X.rename(columns = {'STL': 'Avg steals per game'}, inplace = True)
    X.rename(columns = {'BLK': 'Avg blocks per game'}, inplace = True)
    X.rename(columns = {'TOV': 'Avg turnovers per game'}, inplace = True)

    
    print(X.info())
    oversample = RandomOverSampler(sampling_strategy=1)
    X_res, y_res = oversample.fit_resample(X, y)

    y_res.value_counts()
    #X_res = StandardScaler().fit_transform(X_res)

    #y_res = np.array(y_res)

    #print("__________", np.any(np.isnan(X_res)))
    #print("__________", np.any(np.isnan(y_res)))
    #print("______", np.all(np.isfinite(X_res)))
    #print("______", np.all(np.isfinite(y_res)))



    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, shuffle = True, test_size = 0.1, random_state = 1)


    #GB_pred, GB_model = functions.gb_function(X_train, y_train, X_test, y_test, 300, 0.001, 10, 10) # 0.8
    #GB_pred, GB_model = functions.gb_function(X_train, y_train, X_test, y_test, estimators = 200, lr=0.5, max_features= 18, max_depth=15)
    #RF_pred, RF_model = functions.rf_function(X_train, y_train, X_test, y_test, estimators=700)
    return X_train, X_test, y_train, y_test

