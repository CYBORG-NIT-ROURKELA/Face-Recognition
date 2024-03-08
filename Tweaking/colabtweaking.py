'''This file contains few functions that are used to tweak the model and get the performance metrics of the model'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from deepface.commons import distance as dst



def setup():
    df = pd.read_csv("distance_deepface.csv")
    df = df.drop("Unnamed: 0", axis = 1)

    return df

def initial_readings(df):
    tp_mean = round(df[df.decision == "Yes"].mean().values[0], 4)
    tp_std = round(df[df.decision == "Yes"].std().values[0], 4) 
    fp_mean = round(df[df.decision == "No"].mean().values[0], 4)
    fp_std = round(df[df.decision == "No"].std().values[0], 4)

    Yes_min = df[df['decision']=="Yes"]['distance'].min()
    Yes_max = df[df['decision']=="Yes"]['distance'].max()
    No_min = df[df['decision']=="No"]['distance'].min()
    No_max = df[df['decision']=="No"]['distance'].max()

    return tp_mean, tp_std, fp_mean, fp_std, Yes_min, Yes_max, No_min, No_max

def threshold_func(tp_mean, tp_std):
    return round(tp_mean + 1 * tp_std, 4)

def predictions_func(threshold,df):
    df['predictions'] = df['distance'].apply(lambda x: "Yes" if x <= threshold else "No")

    return df

def final_readings(df):
    tp,_ = df.query("decision == 'Yes' and predictions == 'Yes' ").shape
    fp,_ = df.query("decision == 'No' and predictions == 'Yes'").shape
    tn,_ = df.query("decision == 'No' and predictions == 'No'").shape
    fn,_ = df.query("decision == 'Yes' and predictions == 'No'").shape
    len,_= df.shape

    return tp, fp, tn, fn, len

def score(tp,fp,tn,fn,len,df,threshold):

  df['predictions'] = df['distance'].apply(lambda x: "Yes" if x <= threshold else "No")

  tp,_ = df.query("decision == 'Yes' and predictions == 'Yes' ").shape
  fp,_ = df.query("decision == 'No' and predictions == 'Yes'").shape
  tn,_ = df.query("decision == 'No' and predictions == 'No'").shape
  fn,_ = df.query("decision == 'Yes' and predictions == 'No'").shape

  pos_accuracy = round((tp + tn)/len,4)
  pos_precision = round((tp)/(tp + fp),4)
  pos_recall    = round((tp)/(tp + fn),4)
  f1_score = round((pos_precision*pos_recall)/(pos_precision+pos_recall),4)

  return pos_accuracy, pos_precision,pos_recall,f1_score

def scoreboard(tp,fp,tn,fn,len,df,threshold,start = 0.4,end = 5,skip = 0.5):

  scoreboard_value = {}
  for i,thresh in enumerate(np.arange(start,end,skip)):
        threshold = round(thresh,4)
        
        scoreboard_value[f'-----Iteration: {i+1} ----- threshold: {threshold}-----'] = list(score(tp,fp,tn,fn,len,df,threshold))

  return scoreboard_value

