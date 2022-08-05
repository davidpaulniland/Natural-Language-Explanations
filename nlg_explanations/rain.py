from imports import *

def DataExplorationRain():
    weather_data = pd.read_csv("weatherAUS.csv")
    print(weather_data.shape)
    print(weather_data.info)
    sns.countplot(x=weather_data["RainTomorrow"])
    corrmatrix = weather_data.corr()
    cmap = sns.diverging_palette(260, -10, s=50, l=75, n=6, as_cmap=True)
    plt.subplots(figsize=(18,18))
    sns.heatmap(corrmatrix, cmap = cmap, annot=True, square=True)
    missing_vals = pd.DataFrame(weather_data.isnull().sum()).sort_values(by=0, ascending = False)
    print("Percentage of missing values...",missing_vals/len(weather_data)*100)
    return weather_data



def DataCleaningRain(weather_data):
# numerical cols 
    weather_data['MinTemp'] = weather_data['MinTemp'].fillna(weather_data['MinTemp'].mean())
    weather_data['MaxTemp'] = weather_data['MaxTemp'].fillna(weather_data['MaxTemp'].mean())
    weather_data['Rainfall'] = weather_data['Rainfall'].fillna(weather_data['Rainfall'].mean())
    weather_data['Evaporation'] = weather_data['Evaporation'].fillna(weather_data['Evaporation'].mean())
    weather_data['Sunshine'] = weather_data['Sunshine'].fillna(weather_data['Sunshine'].mean())
    weather_data['WindGustSpeed'] = weather_data['WindGustSpeed'].fillna(weather_data['WindGustSpeed'].mean())
    weather_data['WindSpeed9am'] = weather_data['WindSpeed9am'].fillna(weather_data['WindSpeed9am'].mean())
    weather_data['WindSpeed3pm'] = weather_data['WindSpeed3pm'].fillna(weather_data['WindSpeed3pm'].mean())
    weather_data['Humidity9am'] = weather_data['Humidity9am'].fillna(weather_data['Humidity9am'].mean())
    weather_data['Humidity3pm'] = weather_data['Humidity3pm'].fillna(weather_data['Humidity3pm'].mean())
    weather_data['Pressure9am'] = weather_data['Pressure9am'].fillna(weather_data['Pressure9am'].mean())
    weather_data['Pressure3pm'] = weather_data['Pressure3pm'].fillna(weather_data['Pressure3pm'].mean())
    weather_data['Cloud9am'] = weather_data['Cloud9am'].fillna(weather_data['Cloud9am'].mean())
    weather_data['Cloud3pm'] = weather_data['Cloud3pm'].fillna(weather_data['Cloud3pm'].mean())
    weather_data['Temp9am'] = weather_data['Temp9am'].fillna(weather_data['Temp9am'].mean())
    weather_data['Temp3pm'] = weather_data['Temp3pm'].fillna(weather_data['Temp3pm'].mean())

    # categorical cols
    weather_data['Date'] = weather_data['Date'].fillna(weather_data['Date'].mode()[0])
    weather_data['Location'] = weather_data['Location'].fillna(weather_data['Location'].mode()[0])
    weather_data['WindGustDir'] = weather_data['WindGustDir'].fillna(weather_data['WindGustDir'].mode()[0])
    weather_data['WindDir9am'] = weather_data['WindDir9am'].fillna(weather_data['WindDir9am'].mode()[0])
    weather_data['WindDir3pm'] = weather_data['WindDir3pm'].fillna(weather_data['WindDir3pm'].mode()[0])
    weather_data['RainToday'] = weather_data['RainToday'].fillna(weather_data['RainToday'].mode()[0])
    weather_data['RainTomorrow'] = weather_data['RainTomorrow'].fillna(weather_data['RainTomorrow'].mode()[0])
    print("COLS", weather_data.columns)

    y = weather_data[['RainTomorrow']]
    X = weather_data.drop('RainTomorrow', axis=1)
    X = X.drop('Date', axis = 1)

    oversample = RandomOverSampler(sampling_strategy = 1)
    X_res, y_res = oversample.fit_resample(X, y)
    print(y_res['RainTomorrow'].value_counts())
    encoder = LabelEncoder()
    
    X_res['Location'] = encoder.fit_transform(X_res['Location'])
    X_res['WindGustDir'] = encoder.fit_transform(X_res['WindGustDir'])
    X_res['WindDir9am'] = encoder.fit_transform(X_res['WindDir9am'])
    X_res['WindDir3pm'] = encoder.fit_transform(X_res['WindDir3pm'])
    X_res['RainToday'] = encoder.fit_transform(X_res['RainToday'])
    y_res = y_res.replace({'Yes': 1, 'No': 0})
    print("COLS", X_res.columns)
    
    print("X_res Head_______________",X_res.head(100))
    print("Y_res Value Counts____________",y_res.value_counts())
    return(X_res, y_res)



# X_train, X_test, y_train, y_test = train_test_split(X_res,y_res, shuffle = True, test_size=0.2, random_state=1)
# print(y_test.value_counts())

# RF_model_rain = RandomForestClassifier(n_estimators=600, random_state=1).fit(X_train, y_train)
# RF_pred_rain = RF_model_rain.predict(X_test)


# get_accuracy(RF_pred_rain)
