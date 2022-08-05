from imports import *
def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts) 
    
def NLG(X_train, y_train, intercept_coefficients, slope_coefficients,x_pred, y_pred, weighting_locs, feature_i, pdp, ale, sentence_threshold, rounding_parameter):
    interpret_sentence = "\n\n"
    y_pred = np.array(y_pred)
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
    y_pred = rounders(y_pred, rounding_parameter=rounding_parameter)
    #y_pred = rounders(y_pred, rounding_parameter=rounding_parameter)
    #x_pred = rounders(x_pred, rounding_parameter=rounding_parameter)
    #x_pred = rounders(weighting_locs, rounding_parameter=rounding_parameter)
    #x_pred = weighting_locs
    #x_pred = rounders(x_pred, rounding_parameter = rounding_parameter)


    
    num_reg = 0
    index_slope = 0
    intercepts = np.array(intercept_coefficients)
    slopes = np.array(slope_coefficients)
    rounded_slopes = [np.round(z, decimals = 8) for z in slopes]
    
    
    print("rounded_y", y_pred)


    #weighting_locs = np.round_(weighting_locs)
    #print("_________", weighting_locs)
    print("x_pred", x_pred)
    print("x_pred", x_pred)
    print("x_pred", x_pred.shape)
    print("y_pred", y_pred.shape)
    x_pred.reshape(1, len(x_pred))
    y_pred.reshape(1, len(y_pred))
    print("x_pred reshaped", list(x_pred))
    print("y_pred reshaped", list(y_pred))
    #print("weighting_locs", weighting_locs.shape)

    #x_pred = rounders(x_pred, rounding_parameter=rounding_parameter)
    
    

    index_int = 0
    if pdp == True:
        interpret_sentence += "Partial dependence measures the marginal effect a feature has on the predicted outcome. "
        interpret_sentence += "It helps users understand the relationship between the target and a feature of interest in the context of the overall data. "
        interpret_sentence += "Partial dependence does not use just one instance of a feature; it averages across many instances. "
        interpret_sentence += "This averaging step can lead to unlikely data points in the feature's values that are not in the dataset. "
        #interpret_sentence += "Partial dependence assume variables are independent and not correlated. "
        interpret_sentence += "\n\n"+"Overall, the average marginal effect of the model predicting \""+y_train.columns.values[0]+"\" based on \"" +X_train.columns.values[feature_i]+"\" is between " + str(min(y_pred))+ " and " + str(np.max(y_pred)) +". \n"    

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

    
    if str(min(y_pred)) == str(max(y_pred)):
        interpret_sentence += "\nThere is very little change in the model's prediction based on changes in the feature \""+str(X_train.columns.values[feature_i])+"\"."
    else:
        interpret_sentence += "\nThe feature \"" + str(X_train.columns.values[feature_i]) + "\" directly affects the model's outcome. "

    temp_sentence = ""

 
    #print("y_pred", y_pred)
    #print("y_pred", y_pred.shape)
    #print("y_pred. reshape", y_pred.reshape(1, len(y_pred)))
    #print("x_pred", x_pred)
    #print("x_pred", x_pred.shape)
    #print("x_pred. reshape", x_pred.reshape(1, len(x_pred)))
    #print("rounded slopes", rounded_slopes)
    #print("rounded slopes", len(rounded_slopes))
    #print("weighting_locs ", weighting_locs)
    #print("weighting_locs ", len(weighting_locs))
    #print("intercepts ", intercept_coefficients)
    #print("y_pred", y_pred)
    #print("intercepts ", len(intercept_coefficients))

    
    
    #x_pred = x_pred.reshape(1, len(x_pred))
    #y_pred = y_pred.reshape(1, len(y_pred))


    x_pred = list(x_pred)
    y_pred = list(y_pred)
    

 
    increases_count = 0
    decreases_count = 0 
    constant_count = 0
    temp_sentence_count = 0

    constant_count = 0 
    increases_count = 0
    decreases_count = 0 


    print("List x_pred ", x_pred)
    print("List y_pred ", y_pred)
    

    #for previous, item, nxt in previous_and_next(rounded_slopes):
    for previous, item, nxt in previous_and_next(y_pred):
        

        
        # First round  

        if (previous == None) & (pdp == True):
            temp_sentence += "The partial dependence of the feature \"" +str(X_train.columns.values[feature_i])+"\" on the model predicting \""+y_train.columns.values[0]+ "\" "
            
        elif (previous == None) & (ale == True):
            temp_sentence += "The accumulated local effects of the feature \"" +str(X_train.columns.values[feature_i])+"\" on the model predicting \"" + y_train.columns.values[0] + "\" "

            
            
        else:
            pass
        if previous == None:
            
            start_y = str(y_pred[0][0])
            start_x = str(int(x_pred[0]))
            #x_pred = np.append(x_pred, x_pred[-1])
            #y_pred = np.append(y_pred, y_pred[-1])
            #x_pred = list(x_pred)
            #y_pred = list(y_pred)
            print("x_pred", x_pred)
            print("y_pred", y_pred)
            temp_sentence += "starts with an effect of " + start_y + " when the feature's value is at " + start_x +". "
            temp_sentence += "Then, the effect "


            index_int += 1
      
            

        else:
            
        
        
        ##### CONSTANT #####
        #if item == 0:
            if nxt != None:

                if item == previous:

                    
                    if constant_count == 0:
                        constant_from = str(int(x_pred[index_int]))
                        
                        constant_count = 1
                    else:

                        pass


                    if item == nxt:
                        pass

                    else:
                        
                        constant_y = str(y_pred[index_int][0])
                        constant_x = str(int(x_pred[index_int]))
                        if nxt == None:
                            temp_sentence = temp_sentence[0:-2]
                            temp_sentence += " and "
                        else:
                            pass
                        

                        temp_sentence += "stays constant"
                        
                        if start_x == constant_x:
                            temp_sentence += ", "
                            constant_count = 0
                        else:
                            temp_sentence += " at "
                        if constant_from == constant_x:
                            temp_sentence += constant_y +" when the feature's value is at "+ constant_x + ", "
                            temp_sentence_count += 1
                            constant_count = 0
                        else:
                            temp_sentence += constant_y +" when the feature's value is from "+constant_from +" to " + constant_x + ", "
                            temp_sentence_count += 1
                            constant_count = 0
                        #constant_count = 0

                            

                    
                ###### INCREASE #####
                elif (item > previous):    

                    if increases_count == 0:
                        
                        increases_count = 1

                    else:
                        pass

                    
                    if (nxt > item):
                        pass

                    else:
                        
                        inc_y = str(y_pred[index_int][0])
                        inc_x = str(int(x_pred[index_int]))
                        if nxt == None:
                            temp_sentence = temp_sentence[0:-2]
                            temp_sentence += " and "
                        else:
                            pass

                        temp_sentence += "increases "
                        if start_x == inc_x:
                            if start_y == inc_y:
                                pass
                            else:    
                                temp_sentence += "to an effect of "+ inc_y
                            temp_sentence += ", "
                        else:
                            temp_sentence += "to "
                            inc_y = str(y_pred[index_int][0])
                            inc_x = str(int(x_pred[index_int]))
                            temp_sentence +=  inc_y + " when the feature's value is at " + inc_x + ", "
                            temp_sentence_count += 1
                        increases_count = 0
                        
                            
                        
                        
                ##### DECREASE #####
                #elif (item < 0):
                elif (item < previous):

                    if decreases_count == 0:
                        dec_y = str(y_pred[index_int][0])
                        
                        decreases_count = 1
                    else:
                        pass
                        

                    if (nxt < 0):
                        pass

                    else:
                        
                        if nxt == None:
                            temp_sentence = temp_sentence[0:-2]
                            temp_sentence += " and "
                        else:
                            pass
                        dec_y = str(y_pred[index_int][0])
                        dec_x = str(int(x_pred[index_int]))


                        temp_sentence += "decreases "
                                    
                        if start_x == dec_x:
                            if start_y == dec_y:
                                pass
                            else:    
                                temp_sentence += "to an effect of "+ dec_y
                            temp_sentence += ", "
                            
                        else:
                            temp_sentence += "to "
                            dec_y = str(y_pred[index_int][0])
                            dec_x = str(int(x_pred[index_int]))
                            temp_sentence += dec_y +" when the feature's value is at " + dec_x + ", "
                            temp_sentence_count += 1
                        decreases_count = 0
                        
                
                    
            
                


            else:
                pass
            
            index_int += 1
        





        
    
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
        

        #maximum_x = np.max(x_pred)
        #minimum_x = np.min(x_pred)
        maximum = np.max(y_pred)
        minimum = np.min(y_pred)

        index_min = np.where(y_pred == minimum)
        index_max = np.where(y_pred == maximum)
        index_min = int(index_min[0][0])
        index_max = int(index_max[0][0])
        print("max x", max(x_pred))
        print("min x", min(x_pred))

        print("max y ", y_pred[index_max])
        print("min y ", y_pred[index_min])

        maximum_x = x_pred[index_max]
        minimum_x = x_pred[index_min]

        maximum_y = y_pred[index_max]
        minimum_y = y_pred[index_min]



        #print(index_min)
        #print(index_max)
        #index_min = int(index_min)
        #index_max = int(index_max)
  
        variable_sentence += "There is a lot of variability in the output based on the feature variable. "

        """ MAXIMUM """
        variable_sentence += "The model is most likely to predict \""+y_train.columns.values[0]+ "\" based on the feature \"" +str(X_train.columns.values[feature_i]) +"\" when the feature is at "
        variable_sentence += str(int(maximum_x)) + " with " + probability_or_effect + "of "
        variable_sentence += str(maximum_y[0]) +". "

        """ MINIMUM """
        variable_sentence += "The model is least likely to predict \"" +y_train.columns.values[0] + "\" when the feature has a value of "
        variable_sentence += str(int(minimum_x)) + " with " + probability_or_effect + "of "
        variable_sentence += str(minimum_y[0])+". "
        
        


        interpret_sentence += variable_sentence
    else:
        interpret_sentence += temp_sentence[0:-2] + "."
    
    print("_____",interpret_sentence, "\n\n\n")
