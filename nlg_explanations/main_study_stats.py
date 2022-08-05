from scipy.stats import f_oneway
import seaborn as sns 
import pylab
from collections import Counter
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from scipy.stats import bartlett
import numpy as np
import scipy.stats
from scipy import stats
import pandas as pd
import statistics
from pingouin import ancova
from tables import Cols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
def main_study(array, tot):
    print((sum(array))/tot)
excel = pd.read_excel('Survey_condensed_July 14, binary_answers.xlsx')
print(excel.head())
print(excel.tail())


print(excel[excel['Mode']=="Visual"])

print("THIS",excel[excel["SC0"]==1])

print(excel[excel["Mode"]=="Multimodal"])


excel["SC0"] = excel["SC0"]/9*100






multimodal_data = excel[excel["Mode"]=="Multimodal"]
print("Shape multimodal data", multimodal_data.shape)
visual_data = excel[excel["Mode"]=="Visual"]
text_data = excel[excel["Mode"]=="Text"]
print("Mutlimodal_data", multimodal_data)


multimodal_data = multimodal_data[multimodal_data["SC0"]>56]

print("MULTIMODAL DATA_ _ _ _ _ _ _ _ _ ", multimodal_data)
print(multimodal_data.shape)
print(visual_data.shape)
print(text_data.shape)

frames = [multimodal_data, visual_data, text_data]
excel = pd.concat(frames)
print("THIS IS EXCEL _ _  _ _ _ _ excel", excel)

print("Amount of people not completed the multimodal condition", len(multimodal_data[multimodal_data["Progress"]<100]))
print("Amount of people not completed the visual condition", len(visual_data[visual_data["Progress"]<100]))
print("Amount of people not completed the text condition", len(text_data[text_data["Progress"]<100]))


print(excel.shape)
excel = excel[excel["Progress"]==100]
print(excel.shape)
ml_experience = np.round(excel["Q48"])
print(ml_experience)

#ml_experience = np.array([0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,4,4,4, 5, 8])
def years_experience(ml_experience):
    print("Machine Learning in Years Mean", np.mean(ml_experience))
    print("Machine Learning in Years Std", np.std(ml_experience))
    print("Machine Learning in Years Median", np.median(ml_experience))
    counter = Counter()
    years = np.unique(ml_experience)
    print(years)
    print(ml_experience)
    ml_experience = list(ml_experience)
    for i in ml_experience:
        counter[i] += 1
    counter
    print("This is counter", counter)
    x = list(counter.keys())
    y = list(counter.values())
    ypos = np.arange(len(y))
    print(ypos)
    plt.bar(  x, y)
    plt.title("Machine learning experience in years")
    plt.ylabel("Number of participants")
    plt.xlabel("Years of experience")

    plt.xticks([0,1,2,3,4,5,6,7,8])
    plt.show()
years_experience(ml_experience)




def XAI_level():
    #print(excel['XAI Level ']==)
    print(len(excel['XAI Level ']))
    beginner = (len(excel[excel['XAI Level ']== "Beginner"]))/len(excel) *100
    print(beginner)
    intermediate = (len(excel[excel['XAI Level ']== "Intermediate"]))/len(excel) *100
    print(intermediate)
    proficient = (len(excel[excel['XAI Level ']== "Proficient"]))/len(excel) *100
    print(proficient)

    xai_experience_main = [beginner,intermediate,proficient]
    plt.bar(["Beginner", "Intermediate", "Proficcient"],xai_experience_main)
    plt.title("XAI experience level ")
    plt.ylabel("Percent of total participants")
    plt.xlabel("Stated experience level of XAI")
    plt.show()
XAI_level()

def scores():
    print("_________Mean score multimodal__________")
    multimodal_scores = multimodal_data["SC0"]
    total_multimodal = len(multimodal_scores)
    main_study(multimodal_scores, total_multimodal)

    print("_________Mean score visual_________")
    visual_scores = visual_data["SC0"]
    total_visual = len(visual_scores)
    main_study(visual_scores, total_visual)

    
    text_scores = text_data["SC0"]
    total_text = len(text_scores)
    print("___________Mean score text__________")
    main_study(text_scores, total_text)
    return multimodal_scores, text_scores, visual_scores
multimodal_scores, text_scores, visual_scores = scores()

print(multimodal_scores, visual_scores, text_scores)





def duration():
    duration_dataframe =  np.array([multimodal_data["Duration (in seconds)"], visual_data["Duration (in seconds)"], text_data["Duration (in seconds)"]])
    #duration_dataframe.columns = ["Multimodal", "Visual", "Text"]
    print("Duration in seconds multimodal data", np.mean(multimodal_data['Duration (in seconds)']))
    print("Duration in seconds visual data", np.mean(visual_data['Duration (in seconds)']))
    print("Duration in seconds text data", np.mean(text_data['Duration (in seconds)']))
    return duration_dataframe
duration_dataframe = duration()
print(duration_dataframe)



print("\n\n")

def create_boxplot(data, ylabel, title):

    ax = plt.boxplot(x=data)
    #help(plt.yticks())
    #plt.yticks(np.minimum(data), np.maximum(data))
    plt.xticks(np.arange(1, 4, 1), labels=["Multimodal condition", "Visual condition", "Text condition"])
    plt.title(title)
    plt.ylabel(ylabel)
    
    #plt.Axes.set_yscale('log')

    plt.show()
data_for_boxplot = np.array([multimodal_scores, visual_scores, text_scores])
create_boxplot(data_for_boxplot, ylabel = "Percentage of correct answers", title = "Correct answers for each condition")

data_for_boxplot_logscale = np.array([np.log(multimodal_scores), np.log(visual_scores), np.log(text_scores)])
create_boxplot(data_for_boxplot_logscale, ylabel = "Percentage of correct answers (log scale)", title = "Correct answers for each condition")

log_duration_dataframe = np.array([np.log(duration_dataframe[0]), np.log(duration_dataframe[1]), np.log(duration_dataframe[2])])
create_boxplot(log_duration_dataframe, ylabel= "Duration in seconds (log scale)", title = "Duration of each condition")


def normality_test():
    #shapiro_test_data = np.random.normal(loc=20, scale = 5, size =150)
    
    def is_gaussian(shapiro_test_data):
        stat, p = shapiro(shapiro_test_data)
        print("\n")
        print('stat=%.3f, p=%.3f\n' % (stat, p))
        if p > 0.05:
            print("Likely gaussian")
        else:
            print("Likely not gaussian")
        print("_ _ _ _ _ _ _ _ _ ")
  
    print("Multimodal")
    is_gaussian(multimodal_scores)
    print("Visual")
    is_gaussian(visual_scores)    
    print("Text")
    is_gaussian(text_scores)
    print("Duration Multimodal")
    is_gaussian(duration_dataframe[0])
    print("Duration Visual")
    is_gaussian(duration_dataframe[1])
    print("Duration Text")
    is_gaussian(duration_dataframe[2])

normality_test()
print("\n\n")


def homogeneity():
    stat, p = bartlett(multimodal_scores, text_scores, visual_scores)
    print("homogeneity...", stat, p)
homogeneity()
def one_way_anova(a, b, c):
    print("\n\n")
    print(" _ _ _ _ _ _ _ ")
    print("\n")
    print("One Way Anova") 
    print(f_oneway(a,b,c))
    print("\n\n")
    print(" _ _ _ _ _ _ _ ")
one_way_anova(multimodal_scores, visual_scores, text_scores)
print("One way anova duration")
one_way_anova(duration_dataframe[0], duration_dataframe[1], duration_dataframe[2])

"""  Note: Q48 is ML Experience  """


print(ancova(data=excel, dv='SC0', covar='Q48', between='Mode'))
#print(ancova(data=excel, dv='Duration (in seconds)', covar='Q48', between='Mode'))
"""def density_plot():
    sns.kdeplot(data[0], label = "Multimodal")
    #sns.kdeplot(visual_scores, shade = True)
    #sns.kdeplot(text_scores, shade =True)
    plt.title("Score Data")
    plt.show()
density_plot()"""

tukey = pairwise_tukeyhsd(endog=excel['SC0'],
                          groups=excel['Mode'],
                          alpha=0.05)
print("TUKEY___\n\n")                        
print(tukey)





multimodal_data = multimodal_data.fillna(value = 0)
visual_data = visual_data.fillna(value = 0)
text_data = text_data.fillna(value = 0)
print("LEN", len(multimodal_data['Q. PFI NBA (m)']))
print("LEN", len(multimodal_data["PFI Cancer (m)"]))
print(multimodal_data['Q. PFI NBA (m)'])

def pfi():
    """_ _ _ _ _ _ Multimodal Data _ _ _ _ _ """
    pfi_nba_m = multimodal_data["Q. PFI NBA (m)"].sum()/len(multimodal_data["Q. PFI NBA (m)"])*100
    pfi_cancer_m = multimodal_data['PFI Cancer (m)'].sum()/len(multimodal_data["PFI Cancer (m)"])*100
    pfi_weather_m = multimodal_data['Q. PFI Weather (m)'].sum()/len(multimodal_data["Q. PFI Weather (m)"])*100

    """_ _ _ _ _ _ Visual Data _ _ _ _ _ """
    pfi_nba_v = visual_data["PFI NBA (v)"].sum()/len(visual_data["PFI NBA (v)"])*100
    pfi_cancer_v = visual_data['PFI Cancer (v)'].sum()/len(visual_data["PFI Cancer (v)"])*100
    pfi_weather_v = visual_data['PFI Weather (v)'].sum()/len(visual_data["PFI Weather (v)"])*100
    
    """_ _ _ _ _ _ Text Data _ _ _ _ _ """
    pfi_nba_t = text_data["PFI NBA (t)"].sum()/len(text_data["PFI NBA (t)"])*100
    pfi_cancer_t = text_data["PFI Cancer (t)"].sum()/len(text_data["PFI Cancer (t)"])*100
    pfi_weather_t = text_data['PFI Weather (t)'].sum()/len(text_data["PFI Weather (t)"])*100

    print((pfi_nba_m + pfi_cancer_m + pfi_weather_m)/3)
    print((pfi_nba_v + pfi_cancer_v + pfi_weather_v)/3)
    print((pfi_nba_t + pfi_cancer_t + pfi_weather_t)/3)
    return pfi_nba_m, pfi_cancer_m, pfi_weather_m, pfi_nba_v, pfi_cancer_v, pfi_weather_v, pfi_nba_t, pfi_cancer_t, pfi_weather_t
pfi_nba_m, pfi_cancer_m, pfi_weather_m, pfi_nba_v, pfi_cancer_v, pfi_weather_v, pfi_nba_t, pfi_cancer_t, pfi_weather_t = pfi()

def pdp():
    """_ _ _ _ _ _ Multimodal Data _ _ _ _ _ """
    pdp_nba_m = multimodal_data["Q. PDP NBA (m)"].sum()/len(multimodal_data["Q. PDP NBA (m)"])*100
    pdp_cancer_m = multimodal_data['PDP Cancer (m)'].sum()/len(multimodal_data["PDP Cancer (m)"])*100
    pdp_weather_m = multimodal_data['Q. PDP Weather (m)'].sum()/len(multimodal_data["Q. PDP Weather (m)"])*100

    """_ _ _ _ _ _ Visual Data _ _ _ _ _ """
    pdp_nba_v = visual_data["PDP NBA (v)"].sum()/len(visual_data["PDP NBA (v)"])*100
    pdp_cancer_v = visual_data['PDP Cancer (v)'].sum()/len(visual_data["PDP Cancer (v)"])*100
    pdp_weather_v = visual_data['PDP Weather (v)'].sum()/len(visual_data["PDP Weather (v)"])*100
    
    """_ _ _ _ _ _ Text Data _ _ _ _ _ """
    pdp_nba_t = text_data["PDP NBA (t)"].sum()/len(text_data["PDP NBA (t)"])*100
    pdp_cancer_t = text_data["PDP Cancer (t)"].sum()/len(text_data["PDP Cancer (t)"])*100
    pdp_weather_t = text_data['PDP Weather (t)'].sum()/len(text_data["PDP Weather (t)"])*100

    return pdp_cancer_m, pdp_nba_m, pdp_weather_m, pdp_cancer_v, pdp_nba_v, pdp_weather_v, pdp_cancer_t, pdp_nba_t, pdp_weather_t
pdp_cancer_m, pdp_nba_m, pdp_weather_m, pdp_cancer_v, pdp_nba_v, pdp_weather_v, pdp_cancer_t, pdp_nba_t, pdp_weather_t = pdp()

def ale():
    """_ _ _ _ _ _ Multimodal Data _ _ _ _ _ """
    ale_nba_m = multimodal_data["Q. ALE NBA (m)"].sum()/len(multimodal_data["Q. ALE NBA (m)"])*100
    ale_cancer_m = multimodal_data['ALE Cancer (m)'].sum()/len(multimodal_data["ALE Cancer (m)"])*100
    ale_weather_m = multimodal_data['Q. ALE Weather (m)'].sum()/len(multimodal_data["Q. ALE Weather (m)"])*100

    """_ _ _ _ _ _ Visual Data _ _ _ _ _ """
    ale_nba_v = visual_data["ALE NBA (v)"].sum()/len(visual_data["ALE NBA (v)"])*100
    ale_cancer_v = visual_data['ALE Cancer (v)'].sum()/len(visual_data["ALE Cancer (v)"])*100
    ale_weather_v = visual_data['ALE Weather (v)'].sum()/len(visual_data["ALE Weather (v)"])*100
    
    """_ _ _ _ _ _ Text Data _ _ _ _ _ """
    ale_nba_t = text_data["ALE NBA (t)"].sum()/len(text_data["ALE NBA (t)"])*100
    ale_cancer_t = text_data["ALE Cancer (t)"].sum()/len(text_data["ALE Cancer (t)"])*100
    ale_weather_t = text_data['ALE Weather (t)'].sum()/len(text_data["ALE Weather (t)"])*100

    return ale_cancer_m, ale_nba_m, ale_weather_m, ale_cancer_v, ale_nba_v, ale_weather_v, ale_cancer_t, ale_nba_t, ale_weather_t
ale_cancer_m, ale_nba_m, ale_weather_m, ale_cancer_v, ale_nba_v, ale_weather_v, ale_cancer_t, ale_nba_t, ale_weather_t = ale()




def avg_pfi():
    print("PFI m", round((pfi_cancer_m + pfi_nba_m + pfi_weather_m) / 3))
    print("PFI v", round((pfi_cancer_v + pfi_nba_v + pfi_weather_v) / 3))
    print("PFI t", round((pfi_cancer_t + pfi_nba_t + pfi_weather_t) / 3))
    print("PFI TOTAL", round((pfi_cancer_m + pfi_nba_m + pfi_weather_m + pfi_cancer_v + pfi_nba_v + pfi_weather_v + pfi_cancer_t + pfi_nba_t + pfi_weather_t) / 9))
avg_pfi()

def avg_pdp():
    print("PDP m", round((pdp_cancer_m + pdp_nba_m + pdp_weather_m) / 3))
    print("PDP v", round((pdp_cancer_v + pdp_nba_v + pdp_weather_v) / 3))
    print("PDP t", round((pdp_cancer_t + pdp_nba_t + pdp_weather_t) / 3))
    print("PDP TOTAL", round((pdp_cancer_m + pdp_nba_m + pdp_weather_m + pdp_cancer_v + pdp_nba_v + pdp_weather_v + pdp_cancer_t + pdp_nba_t + pdp_weather_t) / 9))

avg_pdp()

def avg_ale():
    print("ALE m", round((ale_cancer_m + ale_nba_m + ale_weather_m) / 3))
    print("ALE v", round((ale_cancer_v + ale_nba_v + ale_weather_v) / 3))
    print("ALE t", round((ale_cancer_t + ale_nba_t + ale_weather_t) / 3))
    print("ALE TOTAL", round((ale_cancer_m + ale_nba_m + ale_weather_m + ale_cancer_v + ale_nba_v + ale_weather_v + ale_cancer_t + ale_nba_t + ale_weather_t) / 9))

avg_ale()

def avg_cancer():
    print("")
avg_cancer()
