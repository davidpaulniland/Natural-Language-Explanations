from unittest.util import three_way_cmp
from imports import *

def NLG_permu(permu_dataframe):
    important_variables = permu_dataframe[permu_dataframe.iloc[:,1]>0]
    not_important_variables = permu_dataframe[permu_dataframe.iloc[:,1]==0]
    neg_important_variables = permu_dataframe[permu_dataframe.iloc[:,1]<0]

    print("\n\n\n")
    print("IMPORTANT VARIABLE SHAPE", important_variables.shape)

    def cluster(data):
        sorted_importance = data['Mean Importance'].sort_values(ascending=False)
        cum_sum = sorted_importance.cumsum()
        norm_cumsum = cum_sum/cum_sum.max()
        data['Normalized'] = norm_cumsum
        print(data['Normalized'])
        X = data['Normalized']
        print("NORMALIZED SHAPE",data['Normalized'].shape)
        print("X SHAPE", X.shape)
        X = X.values.reshape(-1,1)
        print("X reshaped",X)
        db_scan = DBSCAN(eps=0.9, min_samples = 1)
        
        db_scan.fit(X) 
        
        kmeans = KMeans(n_clusters = 5)
        kmeans.fit(X)
        print("DB Scan Labels____",db_scan.labels_)
        print("K means Labels_______________",kmeans.labels_)
        print("K means Cluster Centers_____", kmeans.cluster_centers_)
        
        return db_scan.labels_, kmeans.labels_
    db_clustered, kmeans_clustered = cluster(important_variables)

    important_variables['kmeans'] = kmeans_clustered


    list_of_uniques = important_variables['kmeans'].unique()
    print("List of uniques", list_of_uniques)
    print("List of uniques", list_of_uniques[0])
    print("List of uniques", list_of_uniques[1])
    print("List of uniques", list_of_uniques[2])
    print("List of uniques", list_of_uniques[3])
    print("List of uniques", list_of_uniques[4])
    
    print("HERE", important_variables.head(25))

    def define_bands(number, variables):
        band = variables[variables['kmeans']==list_of_uniques[number]]
        return band 
    if len(list_of_uniques) >= 0:
        zero  = define_bands(0, variables = important_variables)
    else:
        pass
    if len(list_of_uniques) >= 1:
        one = define_bands(1,variables = important_variables)
    else:
        pass
    if len(list_of_uniques) >= 2:
        two = define_bands(2,variables = important_variables)
    else:
        pass
    if len(list_of_uniques) >= 3:
        three = define_bands(3,variables = important_variables)
    else:
        pass
    if len(list_of_uniques) == 5:
        
        four = define_bands(4,variables = important_variables)
    else:
        pass

    print("HELLO!")
    print(zero, one, two, three, four)





    important_variables = important_variables['Variable']
    not_important_variables = not_important_variables['Variable']
    neg_important_variables = neg_important_variables['Variable']


    
    #  the permutation feature importance takes into account 
    #  both the main feature effect and the interaction effects on model performance.


        

    highest_string = ""
    high_string = ""
    medium_string = ""
    lower_string = ""
    lowest_string = ""
    grouped_string = ""
    """    upper_band = upper_band['Variable']
    medium_band = medium_band['Variable']
    lower_band = lower_band['Variable']"""
    highest_band = zero['Variable']
    high_band = one['Variable']
    medium_band = two['Variable']
    lower_band = three['Variable']
    lowest_band = four['Variable']

    print("HERE!!!!!!!!!!")
    print(highest_band)
    print(high_band)
    print(medium_band)
    print(lower_band)
    print(lowest_band)



    # Important Variables
    grouped_string += "Permutation feature importance measures the model's prediction error increase after a feature's values are permuted. "
    # REWRITE THESE BELOW! 
    grouped_string += "A feature is considered important if shuffling its values increases the model error because, in this case, the model relied on the feature for the prediction. "
    grouped_string += "A feature is considered less important if shuffling its values leaves the model error unchanged because, in this case, the model ignored the feature for the prediction. \n\n"

    

    if len(permu_dataframe) == len(important_variables):
        grouped_string += "Permutation feature importance has revealed that all "+str(len(important_variables))+" variables are important. "
    elif len(permu_dataframe) == 1:
        grouped_string += "Permutation feature importance has revealed that there is "+str(len(important_variables))+" variable that is important. "
    else:
        grouped_string += "Permutation feature importance has revealed that there are "+ str(len(important_variables))+" important variables. "



    if len(important_variables) != 0:
        grouped_string += "Individually permuting these " +str(len(important_variables))+ " variables has led to a decrease in the model's accuracy. "
        #grouped_string += "The variables that are important are "+str(important_string) + ". "
    else:
        pass

    # Highest Band 
    if len(highest_band) > 1:  
        for p in highest_band:
            if p == highest_band.iloc[-2]:
                highest_string = highest_string + "\"" + str(highest_band.iloc[-2]) + "\" and "
            else:
                highest_string += "\""+ p + "\", "
        grouped_string += "The most important variables are " + str(highest_string[0:-2]) + ". "
    elif len(highest_band) == 1:
        # THIS IS THE ISSUE BELOW 
        grouped_string += "The most important variable is \"" + str(highest_band.iloc[0]) + "\". "
    else:
        # No variables are very important
        pass

    # High Important Variables
    if len(high_band) > 1:
        for q in high_band:
            if q == high_band.iloc[-2]:
                high_string = high_string +"\""+str(high_band.iloc[-2]) + "\" and " 
            else:
                high_string += "\""+ q + "\" ,"
        grouped_string += "The variables "+ str(high_string[0:-2]) +" are also very important. "
    elif len(high_band) == 1:
        grouped_string += "The variable \""+ str(high_band.iloc[0]) +"\" is also very important. "
    else:
        # No variables are high important 
        pass
        

    # Medium Important 
    if len(medium_band) > 1: 
        for tea in medium_band:
            if tea == medium_band.iloc[-2]:
                medium_string = medium_string + "\"" + str(medium_band.iloc[-2]) + "\" and "
            else:
                medium_string += "\"" + tea + "\", "
        grouped_string += "The variables "+ str(medium_string[0:-2]) +" positively affect the prediction, but not a lot. "
    elif len(medium_band) == 1:
        grouped_string += "The variable \""+ str(medium_band.iloc[0]) +"\" positively affects the prediction, but not a lot. "
    else:   
        # No variables are marginally important 
        pass
    

        # Low Important 
    if len(lower_band) > 1: 
        for tie in lower_band:
            if tie == lower_band.iloc[-2]:
                lower_string = lower_string + "\"" + str(lower_band.iloc[-2]) + "\", "
            else:
                lower_string += "\"" + tie + "\", "
        grouped_string += "Other variables shown to have a positive effect on the model's accuracy were "+ str(lower_string[0:-2]) +", "
    elif (len(lower_band) == 1) & (len(lowest_band == 0)):
        grouped_string += "The variable \""+ str(lower_band.iloc[0]) +"\" was shown to have a positive effect on the model's accuracy, but only a very small amount. "
    elif(len(lower_band) == 1) & (len(lowest_band == 1)):
        grouped_string += "The variables "+ str(lower_band.iloc[0]) + " and "+ str(lowest_band.iloc[0]) + " were shown to have a positive effect on the model's accuracy, but only a very small amount. "

    else:   
        # No variables are marginally important 
        pass

        # Lowest Important 
    if len(lowest_band) > 1: 
        for teu in lowest_band:
            if teu == lowest_band.iloc[-2]:
                lowest_string = lowest_string + "\"" + str(lowest_band.iloc[-2]) + "\" and "
            else:
                lowest_string += "\"" + teu + "\", "
        grouped_string += str(lowest_string[0:-2]) +". These variables affect the prediction, but only a tiny amount. "
    elif (len(lowest_band) == 1) & (len(lower_band) == 0):
        grouped_string +="The variable \"" + str(lowest_band.iloc[0]) +"\" was shown to have some positive effect on the model's accuracy, but only a tiny amount. "
    else:   
        # No variables are marginally important 
        pass
    """    if len(lower_band) > 1: 
        for tie in lower_band:
            if tie == lower_band.iloc[-2]:
                lower_string = lower_string + " \"" + str(lower_band.iloc[-2]) + "\" and "
            else:
                lower_string += "\"" + tie + "\", "
        grouped_string += "The variables "+ str(lower_string[0:-2]) +" have only a small effect on the prediction. "
    elif len(lower_band) == 1:
        grouped_string += "The variable "+ str(lower_band.iloc[0]) +" has only a small on the prediction, but not a lot. "
    else:   
        # No variables are marginally important 
        pass
    grouped_string += "\n\n"

        # Lowest Important 
    if len(lowest_band) > 1: 
        for teu in lowest_band:
            if teu == lowest_band.iloc[-2]:
                lowest_string = lowest_string + " \"" + str(lowest_band.iloc[-2]) + "\" and "
            else:
                lowest_string += "\"" + teu + "\", "
        grouped_string += "The variables "+ str(lowest_string[0:-2]) +" have some effect on the prediction, but only a very small amount. "
    elif len(lowest_band) == 1:
        grouped_string += "The variable "+ str(lowest_band.iloc[0]) +" has some effect on the prediction, but only a very small amount. "
    else:   
        # No variables are marginally important 
        pass"""
    grouped_string += "\n"


    # Unimportant and Negative 
    if len(not_important_variables) == 0 and len(neg_important_variables) == 0:
        #grouped_string += "There are no variables that do not effect on the model's accuracy, and there are no variables that have a negative effect on the outcome. "
        pass
        
    else:
        
        # Unimportant
        if len(not_important_variables) > 1: 
            grouped_string +=  str(len(not_important_variables)) + " variables don't affect the model's accuracy. "
            grouped_string += "Permuting them individually did not change overall accuracy, either positively or negatively. "
            grouped_string += "The variables are "
            for er in not_important_variables:
                if er == not_important_variables.iloc[-2]:
                    grouped_string = grouped_string + " \""+ not_important_variables.iloc[-2] + "\" and "
                    
                else:
        
                    grouped_string += "\""+ er +"\", " 
            grouped_string = grouped_string[0:-2] + "." 
        elif len(not_important_variables) == 1: 
            grouped_string +=  str(len(not_important_variables)) + " variable doesn't affect the model's accuracy. "
            grouped_string += "Removing it might not change overall accuracy. "
        else:
            pass
        grouped_string += "\n"

        # Negative importance
        if len(neg_important_variables) > 1:
            grouped_string += str(len(neg_important_variables)) + " variables impact the model's accuracy negatively. "
            grouped_string += "Removing one of them could lead to the model's accuracy increasing. The variables are "
            for asd in neg_important_variables:
                if asd == neg_important_variables.iloc[-2]:
                    grouped_string = grouped_string + " \"" + neg_important_variables.iloc[-2] + "\" and "
                else:
                    grouped_string += "\""+ asd + "\", "
            grouped_string = grouped_string[0:-2] +"."

        elif len(neg_important_variables) == 1:
            grouped_string += str(len(neg_important_variables)) + " variable impacts the model's accuracy negatively. "  
            grouped_string += "\"" + str(neg_important_variables.iloc[0]) + "\" is the variable. Removing it might improve overall accuracy. "
        else:
            pass
        




        """if len(neg_important_variables) > 1:
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
            pass"""
        print("\n\n\n")
    print(grouped_string)
    #sns.lineplot(x_pred, y_pred, color="red", label="LOWESS")   
    #plt.xlabel(X_train.columns.values[feature_i])
    #plt.ylabel("Predicted Cancer Probability")
    #plt.show()

