# from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE
from imports import *




df = pd.DataFrame([0.001, 0.002, 0.001, 0.001, 9.0002])
print(df)
avg_num_decimals = [sum((df[col].astype(str).str.split('.', expand=True)[1]).apply(lambda x: len(str(x))))/len((df[col].astype(str).str.split('.', expand=True)[1]).apply(lambda x: len(str(x)))) for col in df.columns]
print("____", type(avg_num_decimals))

int_list = [int(pew) for pew in avg_num_decimals]
int_list = int_list[0]
#print("____", type(int_list[0]))
#rounded = np.round(df, int_list[0])

rounded_df = np.round(df, int_list)
print(rounded_df)




"""permu_dataframe = pd.read_csv("rain_PFI.csv")

print(permu_dataframe.head(10))
important_variables = permu_dataframe[permu_dataframe.iloc[:,2]>0]
not_important_variables = permu_dataframe[permu_dataframe.iloc[:,2]==0]
neg_important_variables = permu_dataframe[permu_dataframe.iloc[:,2]<0]

print("\n\n\n")

#def bandify(data, pos):
    #if pos == True:
    #    upp = 1/3
    #    med = 2/3 
    #else:
    #    upp = -1/3
    #    med = -2/3
    #sorted_importance = data['Mean Importance'].sort_values(ascending=False)
    #cum_sum = sorted_importance.cumsum()
    #norm_cumsum = cum_sum/cum_sum.max()
    #data['Normalized'] = norm_cumsum
    #upper = data[data['Normalized']<upp]
    #medium = data[(data['Normalized']>=upp) & (data['Normalized']<med)]
    #lower = data[data['Normalized']>=med]
    #return upper, medium, lower
#upper_band, medium_band, lower_band = bandify(important_variables, True)
#neg_upper_band, neg_medium_band, neg_lower_band = bandify(neg_important_variables, False)

def bandify(data):
    sorted_importance = data['Mean Importance'].sort_values(ascending=False)
    cum_sum = sorted_importance.cumsum()
    norm_cumsum = cum_sum/cum_sum.max()
    data['Normalized'] = norm_cumsum

    
    data['Index'] = data.index
    print("THIS",data.head(20))
    X = np.array([data['Normalized'], data['Index']])
    print(X.shape)
    plt.scatter(data['Normalized'], data['Index'])
    plt.show()
    #scaler = StandardScaler()
    #scaled_X = scaler.fit_transform(X)
    print("SCALE", X[0])
    print(X[0].shape)
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(X)
    print(kmeans.labels_)




bandify(important_variables)
#important_variables = bandify(important_variables)

important_variables = important_variables['Variable']
not_important_variables = not_important_variables['Variable']
neg_important_variables = neg_important_variables['Variable']



#  the permutation feature importance takes into account 
#  both the main feature effect and the interaction effects on model performance.


    

upper_string = ""
medium_string = ""
lower_string = ""
grouped_string = ""
upper_band = upper_band['Variable']
medium_band = medium_band['Variable']
lower_band = lower_band['Variable']



# Important Variables
grouped_string += "Permutation feature importance measures the increase in the model's prediction error after the feature's values are permuted. "
# REWRITE THESE BELOW! 
grouped_string += "A feature is considered “important” if shuffling its values increases the model error because, in this case, the model relied on the feature for the prediction. "
grouped_string += "A feature is considered “unimportant” if shuffling its values leaves the model error unchanged because, in this case, the model ignored the feature for the prediction. \n\n"

if len(permu_dataframe) == len(important_variables):
    grouped_string += "Permutation feature importance has revealed that all "+str(len(important_variables))+" variables are important. "
elif len(permu_dataframe) == 1:
    grouped_string += "Permutation feature importance has revealed that there is "+str(len(important_variables))+" variable that is important. "
else:
    grouped_string += "Permutation feature importance has revealed that there are "+ str(len(important_variables))+" important variables. "



if len(important_variables) != 0:
    grouped_string += "Removing these variables individually has led to a decrease in the model's accuracy. "
    #grouped_string += "The variables that are important are "+str(important_string) + ". "
else:
    pass

# Upper Important Variables
if len(upper_band) > 1:  
    for p in upper_band:
        if p == upper_band.iloc[-2]:
            upper_string = upper_string + " \"" + str(upper_band.iloc[-2]) + "\" and "
        else:
            upper_string += "\""+ p + "\", "
    grouped_string += "The most important variables are " + str(upper_string[0:-2]) + ". "
elif len(upper_band) == 1:
    # THIS IS THE ISSUE BELOW 
    grouped_string += "The most important variable is \"" + str(upper_band.iloc[0]) + "\". "
else:
    # No variables are very important
    pass

# Medium Important Variables
if len(medium_band) > 1:
    for q in medium_band:
        if q == medium_band.iloc[-2]:
            medium_string = medium_string +" \""+str(medium_band.iloc[-2]) + "\" and " 
        else:
            medium_string += "\""+ q + "\" ,"
    grouped_string += "The variables that play some role in the prediction are "+ str(medium_string[0:-2]) +". "
elif len(medium_band) == 1:
    grouped_string += "The variable that plays some role in the prediction is "+ str(medium_band.iloc[0]) +". "
else:
    # No variables are medium important 
    pass
    

# Lower Important Variables
if len(lower_band) > 1: 
    for t in lower_band:
        if t == lower_band.iloc[-2]:
            lower_string = lower_string + " \"" + str(lower_band.iloc[-2]) + "\" and "
        else:
            lower_string += "\"" + t + "\", "
    grouped_string += "The variables "+ str(lower_string[0:-2]) +" have some effect on the prediction, but not a lot. "
elif len(lower_band) == 1:
    grouped_string += "The variable "+ str(lower_band.iloc[0]) +" has some effect on the prediction, but not a lot. "
else:   
    # No variables are marginally important 
    pass
grouped_string += "\n\n"


# Unimportant and Negative 
if len(not_important_variables) == 0 and len(neg_important_variables) == 0:
    grouped_string += "There are no variables that do not effect on the model's accuracy, and there are no variables that have a negative effect on the outcome. "
else:
    # Unimportant
    if len(not_important_variables) > 1: 
        grouped_string += "There are "+ str(len(not_important_variables)) + " variables that don't effect the model's accuracy. "
        grouped_string += "Removing them individually did not change overall accuracy, either positively or negatively. "
        grouped_string += "The variables are "
        for er in not_important_variables:
            if er == not_important_variables.iloc[-2]:
                grouped_string = grouped_string + " \""+ not_important_variables.iloc[-2] + "\" and "
                
            else:
    
                grouped_string += "\""+ er +"\", " 
        grouped_string = grouped_string[0:-2] + "." 
    elif len(not_important_variables) == 1: 
        grouped_string += "There is " + str(len(not_important_variables)) + " variable that doesn't effect the model's accuracy. "
        grouped_string += "Removing it might not change overall accuracy. "
    else:
        pass
    grouped_string += "\n\n"

    # Negative importance
    if len(neg_important_variables) > 1:
        grouped_string += str(len(neg_important_variables)) + " variables impact the model's accuracy negatively. "
        grouped_string += "Removing them individually has led to an increase in overall accuracy. "
    elif len(neg_important_variables) == 1:
        grouped_string += "There is " + str(len(neg_important_variables)) + " variable that impacts the model's accuracy negatively. "
        grouped_string += "Removing it has led to an increase in overall accuracy. "
    else:
        pass
    

    neg_upper_band = pd.DataFrame(neg_upper_band['Variable']) 
    neg_medium_band = pd.DataFrame(neg_medium_band['Variable']) 
    neg_lower_band = pd.DataFrame(neg_lower_band['Variable']) 
    neg_upper_string = ""
    neg_medium_string = ""
    neg_lower_string = ""

    if len(neg_upper_band) > 1: 
        for tey in neg_upper_band['Variable']:
            if tey == neg_upper_band['Variable'].iloc[-2]:
                neg_upper_string = neg_upper_string + " \"" + str(neg_upper_band['Variable'].iloc[-2]) + "\" and "
            else:
                neg_upper_string += "\"" + tey + "\", "
        grouped_string += "The variables "+ str(neg_upper_string[0:-2]) +" have a strong negative effect on the accuracy. After they were individually removed, the accuracy went up. "
    elif len(neg_upper_band) == 1:
        grouped_string += "The variable "+ str(neg_upper_band['Variable'].iloc[0]) +" has a strong negative effect on the accuracy. After it was removed, the accuracy went up. "
        
    else:   
        # No variables are marginally important 
        pass

    if len(neg_medium_band) > 1: 
        for tese in neg_medium_band['Variable']:
            if tese == neg_medium_band['Variable'].iloc[-2]:
                neg_medium_string = neg_medium_string + " \"" + str(neg_medium_band['Variable'].iloc[-2]) + "\" and "
            else:
                neg_upper_string += "\"" + tese + "\", "
        grouped_string += "The variables "+ str(neg_medium_string[0:-2]) +" have a negative effect on the accuracy, but not a lot. After they were individually removed, the accuracy went up. "
    elif len(neg_medium_band) == 1:
        grouped_string += "The variable "+ str(neg_medium_band['Variable'].iloc[0]) +" has a negative effect on the accuracy, but not a lot. After it was removed, the accuracy went up. "
    else:   
        # No variables are marginally important 
        pass

    if len(neg_lower_band) > 1: 
        for te in neg_lower_band['Variable']:
            if te == neg_lower_band['Variable'].iloc[-2]:
                neg_lower_string = neg_lower_string + " \"" + str(neg_lower_band['Variable'].iloc[-2]) + "\" and "
            else:
                neg_lower_string += "\"" + te + "\", "
        grouped_string += "The variables "+ str(neg_lower_string[0:-2]) +" have a negative effect on the accuracy, but not a lot. After they were individually removed, the accuracy went up slightly. "
    elif len(neg_lower_band) == 1:
        grouped_string += "The variable "+ str(neg_lower_band['Variable'].iloc[0]) +" has a negative effect on the accuracy, but not a lot. After it was removed, the accuracy went up slightly. "
    else:   
        # No variables are marginally important 
        pass
    print("\n\n\n")

print(grouped_string)"""


from imports import *
def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts) 
    
def NLG(X_train, y_train, intercept_coefficients, slope_coefficients,x_pred, y_pred, weighting_locs, feature_i, pdp, ale, sentence_threshold, rounding_parameter):
    interpret_sentence = "\n\n"
    def rounders(df, rounding_parameter):
        #df = pd.DataFrame([0.001, 0.002, 0.001, 0.001, 9.0002])
        #print(df)
        df = pd.DataFrame(df)
        avg_num_decimals = [sum((df[col].astype(str).str.split('.', expand=True)[1]).apply(lambda x: len(str(x))))/len((df[col].astype(str).str.split('.', expand=True)[1]).apply(lambda x: len(str(x)))) for col in df.columns]
        rounded = avg_num_decimals[0]
        print("ROUNDED", round(rounded))
        rounded = rounded/rounding_parameter
        print(rounded)
        rounded = round(rounded)
        print(rounded)
        #int_list = [int(pew) for pew in avg_num_decimals]
        #print("ROUNDED", np.round(df, int_list[0]))
        rounded = np.round_(np.array(df), decimals = rounded)
        #rounded = round(df/rounding_parameter)
        print("THESE.....................", rounded)
        return rounded
    rounded = rounders(y_pred, rounding_parameter=rounding_parameter)
    y_pred = rounded
    #x_pred = rounders(x_pred, rounding_parameter = rounding_parameter)
    print("X_pred", x_pred)
    if pdp == True:
        interpret_sentence += "Partial dependence measures the marginal effect a feature has on the predicted outcome. "
        interpret_sentence += "It helps users understand the relationship between the target and a feature of interest in the context of the overall data. "
        interpret_sentence += "Partial dependence does not use just one instance of a feature; it averages across many instances. "
        interpret_sentence += "This averaging step can lead to unlikely data points in the feature's values that are not in the dataset. "
        #interpret_sentence += "Partial dependence assume variables are independent and not correlated. "
        interpret_sentence += "\n\n"+"Overall, the average marginal effect of the model predicting \""+y_train.columns.values[0]+"\" based on \"" +X_train.columns.values[feature_i]+"\" is between " + str(np.min(y_pred))+ " and " + str(np.max(y_pred)) +". \n"    

    elif ale == True:
        interpret_sentence += "Accumulated local effects is a model-agnostic explainability technique that evaluates the relationship between feature variables and target variables. "
        #interpret_sentence += "They partially isolate the effects of other features, which makes it robust against correlations. "
        interpret_sentence += "Accumulated local effects handle feature correlations by averaging and accumulating the difference in predictions across the conditional distribution, isolating the effects of the feature variable of interest. "
        interpret_sentence += "This average means that there is no creating of unlikely data instances. "
        #interpret_sentence += "Accumulated Local Effects uses the conditional distribution of the feature of interest to generate augmented data. "
        #interpret_sentence += "This helps to reduce the effects of correlated features. "
        #interpret_sentence += "Unlike Partial Dependence, Accumulated Local Effects does not create unlikely data instances. "
        interpret_sentence += "\n\n"+"Overall, the average marginal effect of the model predicting \""+y_train.columns.values[0]+"\" based on \"" +X_train.columns.values[feature_i]+"\" is between " + str(np.min(y_pred))+ " and " + str(np.max(y_pred)) +". \n"
    else:
        pass

    
    num_reg = 0
    index_slope = 0
    intercepts = np.array(intercept_coefficients)
    slopes = np.array(slope_coefficients)
    rounded_slopes = [np.round(z, decimals = 4) for z in slopes]
    
    print("rounded_slopes", rounded_slopes)

    print("BEFORE_________", weighting_locs)
    weighting_locs = np.round_(weighting_locs)
    print("_________", weighting_locs)
    print(x_pred.shape)
    print(y_pred.shape)
    print(weighting_locs.shape)
    
    constant = False
    increases = False
    decreases = False
    index_int = 0

    
    if str(min(y_pred)) == str(max(y_pred)):
        interpret_sentence += "\nThere is very little change in the model's prediction based on changes in the feature \""+str(X_train.columns.values[feature_i])+"\"."
    else:
        interpret_sentence += "\nThe feature \"" + str(X_train.columns.values[feature_i]) + "\" directly affects the model's outcome. "

    temp_sentence = ""
    counter = 0 
    print(y_pred)
    print(x_pred)
    
    counter = 0
    increases_count = 0
    decreases_count = 0 
    constant_count = 0
    temp_sentence_count = 0
    to = False


    for previous, item, nxt in previous_and_next(rounded_slopes):

        
        # First round  

        if (previous == None) & (pdp == True):
            temp_sentence += "The partial dependence of the feature \"" +str(X_train.columns.values[feature_i])+"\" on the model predicting \""+y_train.columns.values[0]+ "\" "
            temp_sentence += "starts at "+str(y_pred[0][0])+". "
            temp_sentence += "Then, the effect "
            index_int += 1
        elif (previous == None) & (ale == True):
            temp_sentence += "The accumulated local effects of the feature \"" +str(X_train.columns.values[feature_i])+"\" on the model predicting \"" + y_train.columns.values[0] + "\" "
            temp_sentence += "starts at "+str(y_pred[0][0])+". "
            temp_sentence += "Then, the effect "
            index_int += 1
        else:
            pass
        
        
        ##### CONSTANT #####
        if item == previous:
            if constant == True: 
                pass                
                #index_int += 1
            else:
                if constant_count == 0:
                    temp_sentence += "stays constant at "+ str(y_pred[index_int][0]) + " "
                    increases = False
                    decreases = False
                    constant = True 
                    constant_count += 1
                    temp_sentence += "when the feature's value is around "+ str(int(x_pred[index_int]))+ ", "
                    temp_sentence_count += 1
                    #index_int += 1

                elif constant_count == 1:
                    temp_sentence += "stays constant again at "+ str(y_pred[index_int][0]) + " "
                    increases = False
                    decreases = False
                    constant = True 
                    constant_count += 1
                    temp_sentence += "when the feature's value is around "+ str(int(x_pred[index_int]))+ ", "
                    temp_sentence_count += 1
                    #index_int += 1
                else:
                    # this worked but not now?... temp_sentence += "stays constant another time at "+ str(y_pred[index_int]) + " "
                    temp_sentence += "stays constant another time at "+  str(y_pred[index_int][0])+ " "
                    increases = False
                    decreases = False
                    constant = True 
                    # resets the counter "again" to avoid repitition 
                    constant_count = 1
                    temp_sentence += "when the feature's value is around "+ str(int(x_pred[index_int]))+ ", "
                    temp_sentence_count += 1
                    #index_int += 1
                    

            
        ###### INCREASE #####
        elif (item > 0):
            if increases == True:
                #index_int += 1
                pass
            else:
                if increases_count == 0:
                    temp_sentence += "increases to "+ str(y_pred[index_int][0]) +" "
                    increases = True
                    decreases = False 
                    constant = False 
                    increases_count += 1
                    temp_sentence += "when the feature's value is around "+ str(int(x_pred[index_int]))+", "
                    temp_sentence_count += 1 
                    #index_int += 1
                    
                elif increases_count == 1:
                    temp_sentence += "increases again to "+ str(y_pred[index_int][0]) +" "
                    increases = True
                    decreases = False 
                    constant = False 
                    increases_count += 1
                    temp_sentence += "when the feature's value is around "+ str(int(x_pred[index_int]))+", "         
                    temp_sentence_count += 1         
                    #index_int += 1

                else:
                    temp_sentence += "increases another time to "+ str(y_pred[index_int][0]) +" "
                    increases = True
                    decreases = False 
                    constant = False 
                    # resets the counter to "again" to avoid repitition 
                    increases_count = 1
                    temp_sentence += "when the feature's value is around "+ str(int(x_pred[index_int]))+", "
                    temp_sentence_count += 1
                    #index_int += 1

            #index_int += 1 
                    
                
                
        ##### DECREASE #####
        elif (item < 0):
            if decreases == True:
                pass
                #index_int += 1
            else:
                if decreases_count == 0:
                    temp_sentence += "decreases to " + str(y_pred[index_int][0]) + " "
                    decreases = True
                    increases = False 
                    constant = False
                    decreases_count += 1
                    temp_sentence += "when the feature's value is around " + str(int(x_pred[index_int]))+", "
                    temp_sentence_count += 1
                    #index_int += 1

                elif decreases_count > 1:
                    temp_sentence += "decreases again to " + str(y_pred[index_int][0]) + " "
                    decreases = True
                    increases = False 
                    constant = False
                    decreases_count += 1
                    temp_sentence += "when the feature's value is around " + str(int(x_pred[index_int]))+", "
                    temp_sentence_count += 1
                    #index_int += 1
                else:
                    temp_sentence += "decreases another time to " + str(y_pred[index_int][0]) + " "
                    decreases = True
                    increases = False 
                    constant = False
                    # resets the counter to "again" to avoid repitition 
                    decreases_count += 0
                    temp_sentence += "when the feature's value is around " + str(int(x_pred[index_int]))+", "
                    temp_sentence_count += 1
                    #index_int += 1
        else:
            pass
            """            if to == False: 
                temp_sentence += " TO " + str(y_pred[index_int][0]) + " "
                to = True
            else: 
                to = True"""
                


    
        
            #index_int += 1
        index_int += 1
    
        counter += 1 
    print("TEMP SENTENCE COUNT...........", temp_sentence_count)
    if temp_sentence_count > sentence_threshold:
        variable_sentence = ""
        if pdp == True:
            probability_or_effect = "an effect "
            variable_sentence += "The partial dependence of the feature \"" +str(X_train.columns.values[feature_i])+"\" on the model predicting \""+y_train.columns.values[0]+ "\" "
            variable_sentence += "starts at "+str(y_pred[0][0])+". "
            #variable_sentence += "Then, the probability \n\n"
        elif ale == True:
            probability_or_effect = "an accumulated local effect "
            variable_sentence += "The accumulated local effects of the feature \"" +str(X_train.columns.values[feature_i])+"\" on the model predicting \"" + y_train.columns.values[0] + "\" "
            variable_sentence += "starts at "+str(y_pred[0][0])+". "
            #variable_sentence += "Then the effect \n\n"
        else:
            pass
        

        maximum = np.max(y_pred)
        minimum = np.min(y_pred)
        index_min = np.where(y_pred == minimum)
        index_max = np.where(y_pred == maximum)
        index_min = int(index_min[0][0])
        index_max = int(index_max[0][0])
  
        variable_sentence += "There is a lot of variability in the output based on the feature variable. "
        variable_sentence += "The model is most likely to predict \""+y_train.columns.values[0]+ "\" based on the feature \"" +str(X_train.columns.values[feature_i]) +"\" when the feature is at "
        variable_sentence += str(round(x_pred[index_max])) + " with " + probability_or_effect + "of "
        variable_sentence += str(y_pred[index_max][0]) + ". "

        variable_sentence += "The model is least likely to predict \"" +y_train.columns.values[0] + "\" when the feature has a value of "
        variable_sentence += str(round(x_pred[index_min])) + " with " + probability_or_effect + "of "
        variable_sentence += str(y_pred[index_min][0]) + ". "
        
        


        interpret_sentence += variable_sentence
    else:
        interpret_sentence += temp_sentence[0:-2] + "."
    
    print("_____",interpret_sentence, "\n\n\n")
