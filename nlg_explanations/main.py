from imports import *
import functions
import cervical_cancer
import local_regression
import language_generators_permu
import language_generators_pdp_ale
import rain
import nba

def main():

    """ Cervical Cancer """
    def CervicalCancer():
        X_res, y_res = cervical_cancer.DataCleaningCancer()
        X_train, X_test, y_train, y_test = functions.split(X_res, y_res, 0.2)
        RF_pred, RF_model = functions.rf_function(X_train, y_train, X_test, y_test, 600)
        GB_pred, GB_model = functions.gb_function(X_train, y_train, X_test, y_test, 100, 0.0001, 5, 20)
        ADAB_pred, ADAB_model = functions.adab_function(X_train, y_train, X_test, y_test, 800, lr=0.1)

        """ Permutation Importance """
        #permu_dataframe_gb = functions.permutation_function(GB_model, X_train, X_test, y_test)
        #language_generators_permu.NLG_permu(permu_dataframe_gb)
        #permu_dataframe_rf = functions.permutation_function(RF_model, X_train, X_test, y_test)
        #language_generators_permu.NLG_permu(permu_dataframe_rf)
        #functions.NLG_permu(permu_dataframe_adab)
        #permu_dataframe_adab = functions.permutation_function(ADAB_model, X_train, X_test, y_test)

        """ Partial Dependence """
        # SELECT NUMBER OF FITS AND VARIABLE BELOW
        #feature_i, number_of_fits, features = local_regression.pd_params(18, 27)
        #tup = local_regression.partial_dependence_function(GB_model, X_train, feature_i, features)
        #weighting_locs, design_matrix, X_lr_pdp, y_lr_pdp, x_pred_pdp, y_pred_pdp = local_regression.lowess_regression_pdp(X_train, tup, number_of_fits, feature_i, target_name = 'Cancer')
        #slope_coefficients_pdp = local_regression.get_slope_coefficients(X_lr_pdp,design_matrix)
        #intercept_coefficients_pdp = local_regression.get_intercept_coefficients(X_lr_pdp, design_matrix)
        #language_generators_pdp_ale.NLG(X_train, y_train, intercept_coefficients_pdp, slope_coefficients_pdp, x_pred_pdp, y_pred_pdp, weighting_locs, feature_i, pdp = True, ale = False, sentence_threshold=5, rounding_parameter=3)
        #local_regression.plot_pdp(x_pred_pdp, y_pred_pdp, X_train, feature_i, target_name='Cancer')
                
        """ ALE """
        #feature_i, number_of_fits, features = local_regression.pd_params(18, 1)
        #X_lr_ale, y_lr_ale = local_regression.ale_function(GB_model, X_train, feature_i, target_name="Cancer", feature_name = "Number of sexual partners")
        #weighting_locs_ale, design_matrix_ale, X_lr_ale, y_lr_ale, x_pred_ale, y_pred_ale = local_regression.lowess_regression_ale(feature_i, X_lr_ale, y_lr_ale, X_train, target_name = "Cancer", number_of_fits =18)
        #intercept_coefficients_ale = local_regression.get_intercept_coefficients(X_lr_ale, design_matrix_ale)
        #slope_coefficients_ale = local_regression.get_slope_coefficients(X_lr_ale,design_matrix_ale)
        #language_generators_pdp_ale.NLG(X_train, y_train, intercept_coefficients_ale, slope_coefficients_ale, x_pred_ale, y_pred_ale, weighting_locs_ale, feature_i, pdp = False, ale = True, sentence_threshold =5, rounding_parameter=4)
    
    
    #functions.plot_explanation(x_pred_ale, y_pred_ale, X_train, feature_i)
    def WeatherData():
        """ Weather Data """
        weather_data = rain.DataExplorationRain()
        X_res, y_res = rain.DataCleaningRain(weather_data)
        X_train, X_test, y_train, y_test = functions.split(X_res, y_res, 0.2)
        #GN_pred, GB_model = functions.gb_function(X_train, y_train, X_test, y_test, 1, 1, 5, 20) 
        # THIS BELOW IS THE SLOW MORE ACCURATE ONE!!!
        RF_pred, RF_model = functions.rf_function(X_train, y_train, X_test, y_test, 600)
        GB_pred, GB_model = functions.gb_function(X_train, y_train, X_test, y_test, 50, 0.0001, 5, 20)
        ADAB_pred, ADAB_model = functions.adab_function(X_train, y_train, X_test, y_test, 800, lr=0.1)

        """ Permutation Importance """
        #permu_dataframe_gb = functions.permutation_function(GB_model, X_train, X_test, y_test)
        #language_generators_permu.NLG_permu(permu_dataframe_gb)
        
        """ Partial Dependence """
        #feature_i, number_of_fits, features = local_regression.pd_params(20, 12)
        #tup = local_regression.partial_dependence_function(GB_model, X_train, feature_i, features)
        #weighting_locs, design_matrix, X_lr_pdp, y_lr_pdp, x_pred_pdp, y_pred_pdp = local_regression.lowess_regression_pdp(X_train, tup, number_of_fits, feature_i, target_name = 'RainTomorrow')
        #slope_coefficients_pdp = local_regression.get_slope_coefficients(X_lr_pdp,design_matrix)
        #intercept_coefficients_pdp = local_regression.get_intercept_coefficients(X_lr_pdp, design_matrix)
        #language_generators_pdp_ale.NLG(X_train, y_train, intercept_coefficients_pdp, slope_coefficients_pdp, x_pred_pdp, y_pred_pdp, weighting_locs, feature_i, pdp= True, ale = False, sentence_threshold=5, rounding_parameter=5)
        #local_regression.plot_pdp(x_pred_pdp, y_pred_pdp, X_train, feature_i, target_name= 'RainTomorrow')

        """ ALE """
        #eature_i, number_of_fits, features = local_regression.pd_params(25, 18)
        #feature_i, number_of_fits, features = local_regression.pd_params(25, 5)
        #X_lr_ale, y_lr_ale = local_regression.ale_function(GB_model, X_train, feature_i, target_name="RainTomorrow", feature_name="Sunshine")
        #weighting_locs_ale, design_matrix_ale, X_lr_ale, y_lr_ale, x_pred_ale, y_pred_ale = local_regression.lowess_regression_ale(feature_i, X_lr_ale, y_lr_ale, X_train, target_name = "Rain Tomorrow", number_of_fits=18)
        #intercept_coefficients_ale = local_regression.get_intercept_coefficients(X_lr_ale, design_matrix_ale)
        #slope_coefficients_ale = local_regression.get_slope_coefficients(X_lr_ale,design_matrix_ale)
        #language_generators_pdp_ale.NLG(X_train, y_train, intercept_coefficients_ale, slope_coefficients_ale, x_pred_ale, y_pred_ale, weighting_locs_ale, feature_i, pdp = False, ale = True, sentence_threshold=5, rounding_parameter = 7)


    def NBA():
        X_train, X_test, y_train, y_test = nba.DataCleaning()
        
        
        RF_pred, RF_model = functions.rf_function(X_train, y_train, X_test, y_test, 800)
        GN_pred, GB_model = functions.gb_function(X_train, y_train, X_test, y_test, 300, 0.001, 10, 10)
        ADAB_pred, ADAB_model = functions.adab_function(X_train, y_train, X_test, y_test, 800, lr=0.1)

        """ Permutation Importance """
        #permu_dataframe_gb = functions.permutation_function(GB_model, X_train, X_test, y_test)
        #language_generators_permu.NLG_permu(permu_dataframe_gb)

        """ Partial Dependence """
        #feature_i, number_of_fits, features = local_regression.pd_params(25, 3)
        #tup = local_regression.partial_dependence_function(GB_model, X_train, feature_i, features)
        #weighting_locs, design_matrix, X_lr_pdp, y_lr_pdp, x_pred_pdp, y_pred_pdp = local_regression.lowess_regression_pdp(X_train, tup, number_of_fits, feature_i, target_name = 'Career > 5 Years')
        #slope_coefficients_pdp = local_regression.get_slope_coefficients(X_lr_pdp,design_matrix)
        #intercept_coefficients_pdp = local_regression.get_intercept_coefficients(X_lr_pdp, design_matrix)
        #language_generators_pdp_ale.NLG(X_train, y_train, intercept_coefficients_pdp, slope_coefficients_pdp, x_pred_pdp, y_pred_pdp, weighting_locs, feature_i, pdp= True, ale = False, sentence_threshold=5, rounding_parameter = 3)
        #local_regression.plot_pdp(x_pred_pdp, y_pred_pdp, X_train, feature_i, target_name= 'Career > 5 Years')

        """ ALE """
        #feature_i, number_of_fits, features = local_regression.pd_params(25, 0)
        #X_lr_ale, y_lr_ale = local_regression.ale_function(GB_model, X_train, feature_i, target_name="Career > 5 Years", feature_name="Games played rookie season")
        #weighting_locs_ale, design_matrix_ale, X_lr_ale, y_lr_ale, x_pred_ale, y_pred_ale = local_regression.lowess_regression_ale(feature_i, X_lr_ale, y_lr_ale, X_train, target_name = "Career > 5 years", number_of_fits = 18)
        #intercept_coefficients_ale = local_regression.get_intercept_coefficients(X_lr_ale, design_matrix_ale)
        #slope_coefficients_ale = local_regression.get_slope_coefficients(X_lr_ale,design_matrix_ale)
        #language_generators_pdp_ale.NLG(X_train, y_train, intercept_coefficients_ale, slope_coefficients_ale, x_pred_ale, y_pred_ale, weighting_locs_ale, feature_i, pdp = False, ale = True, sentence_threshold=5, rounding_parameter =6)

    #CervicalCancer()
    #WeatherData()
    NBA()
if __name__ == '__main__':
    main()
else:
    pass