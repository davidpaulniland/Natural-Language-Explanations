from imports import *
import functions
def DataCleaningCancer():
    cancer_data = pd.read_csv("risk_factors_cervical_cancer.csv")
    cancer_data = cancer_data.replace('?', np.nan)
    cancer_data = cancer_data.drop('Citology', 1)
    cancer_data = cancer_data.drop('Hinselmann', 1)
    cancer_data = cancer_data.drop('Schiller', 1)
    cancer_data.rename(columns={'Biopsy': 'Cervical Cancer'}, inplace=True)


    feature_names = cancer_data.columns.values.tolist()

    #pd.set_option('display.max_columns', 40)
    cancer_data = cancer_data.apply(pd.to_numeric)
    cancer_data = cancer_data.fillna(cancer_data.mean().to_dict())
    print(cancer_data.describe(include='all'))
    

    X = cancer_data.drop('Cervical Cancer', axis=1)
    y = cancer_data[['Cervical Cancer']]
    print("Columns...",X.columns)

    oversample = RandomOverSampler(sampling_strategy = 1)
    X_res, y_res = oversample.fit_resample(X, y)
    
    return X_res, y_res








