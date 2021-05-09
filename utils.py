#utility functions
# packages for data preparation
import pandas as pd
import numpy as np
import os
import datetime as dt
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
from sklearn import preprocessing
import matplotlib.colors as colors
# packages for model pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
# packages for model performance evaluation
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score # Area under PR curve
from sklearn.metrics import matthews_corrcoef # Matthews Correlation Coefficient <- for ranking only
from sklearn.metrics import cohen_kappa_score # Cohen's kappa
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve


def dataset_brief(df, name):
    print('Dataset {2} provided of size:\t{0:1d} rows and {1:1d} columns\n'.\
      format(df.shape[0], df.shape[1], name))
    print('List of columns in the dataset:')
    print(df.columns.tolist())
    x = (df.isnull().sum()/df.shape[0]).sort_values(ascending=False)
    print('over 50% values missing in following columns:')
    print(x[x>0.5])
    id_columns = [x for x in df.columns if 'id' in x]
    for column in id_columns:
        print('Column {0} has {1:1d} unique values'.\
              format(column, 
                     df[column].nunique(dropna=False)))
        if df[column].isnull().sum() >0: 
            print('has missing id')
        else: 
            continue

def get_dow_name(df, col_name):
    return pd.to_datetime(df.loc[:,(col_name)]).dt.day_name()


def my_odds_and_risks(df):
    n = df.shape[0]
    v = range(n)
    for rows in [[a,b] for a in v for b in v if a != b]:
        print(list(df.index[rows]))
        b, a, d, c = np.ravel(df.iloc[rows,:].to_numpy())
        print('OR/RR : {0:.2f} / {1:.2f}'.\
              format((a/c)/(b/d), df.iloc[rows,-1].values[0]/df.iloc[rows,-1].values[1]))
        
def hist(data, bins, title, density=True, range = None):
    fig = plt.figure(figsize=(13, 6))
    ax = plt.axes()
    plt.ylabel("Proportion (Histogram)")
    values, base, _ = plt.hist(data, bins = bins, density=density, 
                               alpha = 0.5, color = "green", range = range, 
                               label = "Histogram")
    ax_bis = ax.twinx()
    values = np.append(values,0)
    ax_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1],
                color='darkorange', marker='o', linestyle='-', markersize = 1, 
                label = "Cumulative Histogram" )
    plt.xlabel("days")
    plt.ylabel("Proportion (Cumulative Histogram")
    plt.title(title)
    ax_bis.legend();
    ax.legend();
    plt.show()


def pt_icu_etl(raw_icu):
    # drop duplicates, exclude rows without icu admission time
    raw_pt_icu = \
        raw_icu.query('intime.notnull() and outtime.notnull() and icustay_id!=229922 and (ttd_days >=0 or ttd_days.isnull())')
    raw_pt_icu.loc[:, ('age_bins')] =\
        pd.cut(raw_pt_icu.age_years, \
               [-math.inf,44,54,64,74,math.inf], \
               labels =['under44','45-54', '55-64','65-74', 'over75'])
    # get day of the week of ICU intime
    raw_pt_icu.loc[:,('intime_weekday')] = get_dow_name(raw_pt_icu, 'intime')
    WKN = {'Saturday', 'Sunday'}
    # recognise weekend shifts
    raw_pt_icu.loc[:,('icu_adm_weekend')] = \
        [True if x in WKN else False for x in raw_pt_icu.intime_weekday]
    # convert to datetime type but extract only dates
    datetime_cols = {'dob', 'admittime', 'dischtime', 'intime', 'outtime', 
                     'hosp_deathtime', 'dod',}
    raw_pt_icu.loc[:, datetime_cols] = \
        raw_pt_icu.loc[:, datetime_cols].apply(lambda x: pd.to_datetime(x).dt.date, axis=0)
    raw_pt_icu.loc[:, 'ttd_bins'] = \
        pd.cut(raw_pt_icu.ttd_days.values, bins=[-math.inf,7,14,30, math.inf])
    #death in 7 days (ttd < 7) to create mortality label
    LABEL = 'standard_mortality_label'
    raw_pt_icu.loc[:, (LABEL)] = raw_pt_icu['ttd_days'] < 7    
    return raw_pt_icu


def admission_etl_function(raw_ad):
    adms_dt_cols = {'admittime', 'dischtime', 'deathtime',}
    raw_ad.loc[:, adms_dt_cols] = \
    raw_ad.loc[:, adms_dt_cols].apply(lambda x: pd.to_datetime(x).dt.date, axis=0)
    raw_admission = raw_ad.sort_values(by=['subject_id', 'admittime']).set_index(['subject_id'])
    raw_admission.loc[:, ('prev_dischtime')] =\
        raw_admission.groupby(level=0).dischtime.shift(1)
    raw_admission.loc[:, ('tt_next_adm_days')] = \
        raw_admission.admittime - raw_admission.prev_dischtime
    raw_admission.loc[:, ('re_adm_in30d')] = \
        raw_admission.tt_next_adm_days.dt.days <= 30
    raw_admission.loc[:, ('len_of_adm')] = \
        (raw_admission.dischtime - raw_admission.admittime).dt.days
    raw_admission.loc[:, ('english_speaker')] = \
        raw_admission.groupby(level=0)['language'].apply(lambda x: 'ENGL' in list(x))
    raw_admission.loc[:, ('admission_type')] = \
        raw_admission.loc[:, ('admission_type')].apply(lambda x: x.replace(' ', '').lower())
    raw_admission.loc[:, ('insurance')] = \
        raw_admission.loc[:, ('insurance')].apply(lambda x: x.replace(' ', '').lower())
    return raw_admission


def hourly_vitals_etl(raw_vitals):
    raw_vitals_condition = \
        ((raw_vitals.spo2<0)|(raw_vitals.spo2>100))| \
        (\
         (raw_vitals.temperature<0)|(raw_vitals.temperature>108)|\
         ((raw_vitals.temperature>45)&(raw_vitals.temperature<96))\
        )| \
        ((raw_vitals.resprate<0)|(raw_vitals.resprate>196))| \
        ((raw_vitals.heartrate<0)|(raw_vitals.heartrate>480)) | \
        ((raw_vitals.sysbp<0)|(raw_vitals.sysbp>300))| \
        ((raw_vitals.diasbp<0)|(raw_vitals.diasbp>250)) | \
        ((raw_vitals.glucose<0)|(raw_vitals.glucose>1500))| \
        ((raw_vitals.meanarterialpressure<0)|(raw_vitals.meanarterialpressure>400))
    vital_df = raw_vitals[~raw_vitals_condition]
    #ETL - Vital
    vital_df.loc[vital_df.temperature > 43, ('temperature')] = \
    vital_df.query('temperature > 43').temperature.apply(lambda x: (x-32)*5/9)
    vital_df.loc[:, ('sys_bp_category')] =\
    pd.cut(vital_df.sysbp, [-math.inf, 120, 129, 139, 180, math.inf], \
           labels=('normal','elevated','HBP-stg1', 'HBP-stg2', 'HBP-Crisis'))
    vital_df.loc[:, ('dias_bp_category')] =\
    pd.cut(vital_df.diasbp, [-math.inf, 80, 89, 120, math.inf], \
           labels=('normal','HBP-stg1', 'HBP-stg2', 'HBP-Crisis'))
    vital_df.loc[:, ('bp_elevated')] =\
        (vital_df.sys_bp_category=='elevated')&(vital_df.dias_bp_category=='normal')
    vital_df.loc[:, ('bp_hbp_s1')] =\
        (vital_df.sys_bp_category=='HBP-stg1')|(vital_df.dias_bp_category=='HBP-stg1')
    vital_df.loc[:, ('bp_hbp_s2')] =\
        (vital_df.sys_bp_category=='HBP-stg2')|(vital_df.dias_bp_category=='HBP-stg2')
    vital_df.loc[:, ('bp_hyptsn_crisis')] =\
        (vital_df.sys_bp_category=='HBP-Crisis')|(vital_df.dias_bp_category=='HBP-Crisis')
    vital_df = \
    vital_df.assign(
        abnorm_spo2 = lambda x: x.spo2 < 95, # spo2 below 95 -> high risk of hypoxemia
        fever = lambda x: x.temperature > 38, # over 38C =fever   
        tachycardia = lambda x: x.heartrate > 100, #tachycardia
        bradycardia = lambda x: x.heartrate < 60,# bradycardia
        diabetes = lambda x: x.glucose >199, #assume rancdom plasma glucose test
        abnorm_map = lambda x: (x.meanarterialpressure > 100)|(x.meanarterialpressure < 60),
        #for doctors to check blood flow, resistance and pressure to supply bloody to major organs
    )
    #ICU first 24hr vitals data
    first_24_vital = vital_df.query('hr>0 and hr<=24')
    first_24_vital_agg = \
    first_24_vital.loc[:, ('icustay_id', 'bp_elevated', 'bp_hbp_s1', 'bp_hbp_s2', 
                           'bp_hyptsn_crisis','abnorm_spo2', 'fever', 'tachycardia',
                           'bradycardia', 'diabetes', 'abnorm_map')].\
                    set_index('icustay_id').groupby(level=0).apply(sum).\
                    apply(lambda x: x > 0, axis=1).astype(int)
    return first_24_vital_agg


def hourly_gcs_etl(raw_gcs): 
    #ETL - GCS
    raw_gcs.loc[:, ('gcs_category')] = \
        pd.cut(raw_gcs.gcs, [0,8,12,math.inf], 
               labels=['severe','moderate','mild']) # based on common GCS classifications
    raw_gcs.loc[:, ('eye_no_resp')] = raw_gcs.gcseyes == 1.0 # eye no response
    raw_gcs.loc[:, ('motor_no_resp')] = raw_gcs.gcsmotor == 1.0 # motor no response
    raw_gcs.loc[:, ('verbal_no_resp')] = raw_gcs.gcsverbal == 1.0 # verbal no response
    raw_gcs.loc[:, ('gcs_severe')] = raw_gcs.gcs_category == 'severe'
    raw_gcs.loc[:, ('gcs_moderate')] = raw_gcs.gcs_category == 'moderate'
    first_24_gcs = raw_gcs.query('hr>0 and hr<=24') #conscuousness
    first_24_gcs_agg = \
    first_24_gcs.loc[:, ('icustay_id', 'endotrachflag', 'eye_no_resp', 'motor_no_resp',
                         'verbal_no_resp', 'gcs_severe', 'gcs_moderate')].\
                    set_index('icustay_id').groupby(level=0).apply(sum).\
                    apply(lambda x: x > 0, axis=1).astype(int)
    return first_24_gcs_agg


def hourly_labs_etl(raw_labs):    
    #ETL - Labs
    # 'glucose', #diabetes
    # 'bilirubin','alaninetransaminase', 'aspartatetransaminase', #Hepatocytedamage #acute hepatitis
    # 'chloride', 'sodium', #electrolites
    # 'creatinine', 'albumin','bloodureanitrogen', #kidney functionality
    # 'hemoglobin', 'hematocrit', #anaemia
    # 'whitebloodcell',  'platelets' , #leukemia
    raw_labs = \
    raw_labs.assign(
        abnorm_bicarbonate = lambda x: (x.bicarbonate<23)|(x.bicarbonate>29), 
        abnorm_albumin = lambda x: (x.albumin<3.5)|(x.albumin>5), 
        abnorm_troponin = lambda x: (x.troponin>0.4),
        abnorm_bloodureanitrogen = lambda x: (x.bloodureanitrogen<7)|(x.bloodureanitrogen>20), 
        abnorm_partialpressureo2 = lambda x: (x.partialpressureo2<75)|(x.partialpressureo2>100),
        abnorm_bilirubin = lambda x: (x.bilirubin<0.1)|(x.bilirubin>1.0),
        abnorm_alt = lambda x: (x.alaninetransaminase<7)|(x.alaninetransaminase>56),
        abnorm_ast = lambda x: (x.aspartatetransaminase<5)|(x.aspartatetransaminase>40),
        abnorm_hemoglobin = lambda x: (x.hemoglobin<116)|(x.hemoglobin>166),
        abnorm_hematocrit = lambda x: (x.hematocrit<35.5)|(x.hematocrit>48.6),
        abnorm_wbc = lambda x: (x.whitebloodcell<3.4)|(x.whitebloodcell>9.6),
        abnorm_platelets = lambda x: (x.platelets<135)|(x.platelets>371),
        abnorm_sodium = lambda x: (x.sodium<135)|(x.sodium>145),
        abnorm_chloride = lambda x: (x.chloride<95)|(x.chloride>110),
        abnorm_creatinine = lambda x: (x.creatinine<0.6)|(x.creatinine>1.3),
        abnorm_glucose = lambda x: (x.glucose>199),
        abnorm_neutrophil = lambda x: (x.neutrophil<45)|(x.neutrophil>75), 
        abnorm_creactiveprotein = lambda x: (x.creactiveprotein>10),
        abnorm_lactate = lambda x: (x.lactate>1.0),
        abnorm_inr = lambda x: (x.intnormalisedratio<2)|(x.intnormalisedratio>3),
    )
    #blood tests, negative hours=pre-ICU
    first_24_labs = raw_labs.query('hr>0 and hr<=24')
    first_24_labs_agg = \
    first_24_labs.loc[:,\
                       ('icustay_id', 'abnorm_albumin', 'abnorm_bilirubin', 'abnorm_alt',
                        'abnorm_ast', 'abnorm_hemoglobin', 'abnorm_hematocrit', 'abnorm_wbc',
                        'abnorm_platelets', 'abnorm_sodium', 'abnorm_chloride',
                        'abnorm_bicarbonate', 'abnorm_troponin', 'abnorm_bloodureanitrogen',
                        'abnorm_partialpressureo2', 'abnorm_creatinine', 'abnorm_glucose',
                        'abnorm_neutrophil', 'abnorm_creactiveprotein', 'abnorm_lactate',
                        'abnorm_inr')].\
                    set_index('icustay_id').groupby(level=0).apply(sum).\
                    apply(lambda x: x > 0, axis=1).astype(int)
    return first_24_labs_agg


def baseline_model_performance(X, y, dictionary_of_algos, X_train, y_train, X_test, y_test, labels):
    """
    passing list of algorithms to run and compare baseline performance
    also need to pass in training and test feature sets and target sets
    and list of labels for confusion matrix visualisation
    """
    for key, classifier in dictionary_of_algos.items():
        start = time.time()
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:,1]
        end = time.time()
        used = end - start
        # Confusion Matrix
        cm = confusion_matrix(y_pred, y_test)
        # ROC True Positive Rate, False Positive Rate
        fpr, tpr, thresholds0 = roc_curve(y_test, y_pred_proba)
        # Precision and Recall
        precision, recall, thresholds1 = precision_recall_curve(y_test, y_pred_proba)
        # F1 Score
        f1 = f1_score(y_test, y_pred)
        # Accuracy
        accuracy = classifier.score(X_test, y_test)
        # ROC_AUC, PR_AUC
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        print('\n')
        print('--------'+key+' Baseline Model---------')
        print('\n')
        # Print Baseline Results
        print("Job took: {}s".format(used))
        print("Baseline Model parameters:\n {}".format(classifier.get_params))
        print("Best cross-validation score (Accuracy): {:.4f}".format(classifier.score(X_test, y_test)))
        print('Accuracy: {0:.3f} \nF1-score: {1:.3f} \nPR_AUC: {2:.3f} \nROC_AUC: {3:.3f} \n'.
              format(accuracy, f1, pr_auc, roc_auc))
        
        # get importance
        if key=='Logistic Regression':
            importance = classifier.coef_[0]
        else:
            importance = classifier.feature_importances_
        # summarize feature importance
        cols = X.columns.values
        df_importance = pd.DataFrame({'feature':cols, 'importance':importance})
        important_10 = df_importance.nlargest(15, 'importance').sort_values(by='importance', ascending=False)
        print(important_10)        
        print(classification_report(y_test, y_pred))
        print('\n')
        
        BIG_SIZE=22
        plt.style.use('seaborn')
        fig, ax = plt.subplots(2,2,figsize=(35, 23))
        #ax = ax.flatten()
        sns.heatmap(cm, annot=True, fmt='.0f', ax=ax[0,0],
                    cmap="Dark2", norm=colors.PowerNorm(gamma=0.5),
                    annot_kws={"size": 18})
        ax[0,0].set_xlabel('Predicted', fontsize=BIG_SIZE);
        ax[0,0].set_ylabel('Actual', fontsize=BIG_SIZE); 
        ax[0,0].set_title('Confusion Matrix', fontsize=BIG_SIZE); 
        ax[0,0].xaxis.set_ticklabels(labels); 
        ax[0,0].yaxis.set_ticklabels(labels);
        ax[0,0].tick_params(labelsize=BIG_SIZE)
        
        ax[0,1].barh("feature", "importance",data=important_10)
        ax[0,1].set_xlabel('Features', fontsize=BIG_SIZE);
        ax[0,1].set_ylabel('Importance', fontsize=BIG_SIZE);
        ax[0,1].set_title('Feature Importance', fontsize=BIG_SIZE); 
        ax[0,1].tick_params(labelsize=BIG_SIZE)
        
        ax[1,0].plot(fpr, tpr, marker='.', label=key)
        ax[1,0].plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        ax[1,0].set_xlabel('False Positive Rate (FPR)', fontsize=BIG_SIZE);
        ax[1,0].set_ylabel('True Positive Rate (TPR)', fontsize=BIG_SIZE);
        ax[1,0].set_title('ROC Curve', fontsize=BIG_SIZE)
        ax[1,0].tick_params(labelsize=BIG_SIZE)
        
        ax[1,1].plot(recall, precision, marker='.', label=key)
        no_skill = len(y[y==True]) / len(y)
        ax[1,1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        ax[1,1].set_xlabel('Recall', fontsize=BIG_SIZE);
        ax[1,1].set_ylabel('Precision', fontsize=BIG_SIZE);
        ax[1,1].set_title('PR Curve', fontsize=BIG_SIZE)
        ax[1,1].tick_params(labelsize=BIG_SIZE)
        plt.show()


def tuning_models_GSCV(X, y, X_train, y_train, X_test, y_test, labels, nested_dict_of_models_params, scaler, scoring, refit_measure='F1'):
    """
    Function takes training and test set, label names,
    a dictionary of models and sets of parameters, a chosen scaler, list of metrics for scoring,
    and a refit measure (defaulted to f1-score)
    Output: 
    lists of prunned models, pruning time, best cross-validation scores, average precisions,
    Matthew's Correlation Coefficients, Cohen's Kappa Coefficients, accuracy, 
    true/false positive counts, true/flase negative counts, 
    true positve rate and false positive rate for plotting ROC Curve,
    precision and recall for plotting PR curve.
    """
    # empty list to store output for each model
    models = []
    time_used = []
    best_cv_scores = []
    roc_test_set = []
    pr_test_set = []
    f1_test_set = []
    ap_test_set = []
    mc_test_set = []
    ck_test_set = []
    acc_test_set = []
    fpr_test_set = []
    tpr_test_set = []
    precision_test_set = []
    recall_test_set = []
    tn_count = []
    fp_count = []
    fn_count = []
    tp_count = []
    tuned_models = []
    
    for model, details in nested_dict_of_models_params.items():
        models.append(model)
        clf = details['clf']
        params = details['params']        
        # build pipeline
        pipeline = Pipeline([('Transformer', scaler), ('Estimator', clf)])
        # time pruning 
        start = time.time()
        
        # Repeated Stratified K-Fold Cross-Validation set:
        cv = StratifiedShuffleSplit(n_splits=7, test_size=0.3, random_state=0)
        grid = GridSearchCV(pipeline, params,
                            cv=cv, n_jobs=-1, scoring=scoring, refit='F1')       
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test) #model prediction
        y_pred_proba = grid.predict_proba(X_test)[:,1] #prediction probability
        end = time.time()
        used = end - start #time the computation
        tuned_models.append(grid)
        time_used.append(used)
        best_cv_scores.append(grid.best_score_)

        # For Visualisation - Confusion Matrix
        cm = confusion_matrix(y_pred, y_test)
        tn, fp, fn, tp = cm.ravel()
        tn_count.append(tn)
        fp_count.append(fp)
        fn_count.append(fn)
        tp_count.append(tp)
        # For Visualisation - ROC, False Positive Rate, True Positive Rate
        fpr, tpr, thresholds0 = roc_curve(y_test, y_pred_proba)
        fpr_test_set.append(fpr)
        tpr_test_set.append(tpr)
        # For Visualisation - PR, Precision, Recall
        precision, recall, thresholds1 = precision_recall_curve(y_test, y_pred_proba)
        precision_test_set.append(precision)
        recall_test_set.append(recall)
        # For Model evaluation
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred_proba)
        mc = matthews_corrcoef(y_test, y_pred)
        ck = cohen_kappa_score(y_test, y_pred)
        f1_test_set.append(f1)
        ap_test_set.append(ap)
        mc_test_set.append(mc)
        ck_test_set.append(ck)

        # Area under curve: ROC_AUC, PR_AUC
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        accuracy = (tn + tp)/(tn + fp + tp + fn)
        acc_test_set.append(accuracy)
        roc_test_set.append(roc_auc)
        pr_test_set.append(pr_auc)
        # get importance
        if model=='Logistic Regression':
            importance = \
            grid.best_estimator_.named_steps["Estimator"].coef_[0]
        else:
            importance = \
            grid.best_estimator_.named_steps["Estimator"].feature_importances_
        # summarize feature importance
        cols = X.columns.values
        df_importance = pd.DataFrame({'feature':cols, 'importance':importance})
        important_15 = df_importance.nlargest(15, 'importance').\
                            sort_values(by='importance', ascending=False)
        
        print('\n')
        print('--------', model,'Tuned Model---------')
        print('\n')
        # Print Grid Search CV Results
        print("Job took: {}s".format(used))
        print("Set of parameters from Exhaustive Grid Search CV:\n {}".format(grid.get_params))
        print("Best parameters from Grid Search CV:\n {}".format(grid.best_params_))
        print("Best cross-validation score: {:.4f}".format(grid.best_score_))
        print(cm)
        print("Accuracy: {:.4f}".format((tp+tn)/(tp+tn+fp+fn)))
        print("Recall (FN): {:.4f}".format((tp)/(tp+fn)))
        print("Precision (FP): {:.4f}".format((tp)/(tp+fp)))
        print('F1-score: {0:.3f} \nPR_AUC: {1:.3f} \nROC_AUC: {2:.3f} \n'.
              format(f1, pr_auc, roc_auc))
        print(classification_report(y_test, y_pred))
        print('---------------------------------------\n')
        print(important_15)

        BIG_SIZE=22
        plt.style.use('seaborn')
        fig, ax = plt.subplots(2,1,figsize=(35, 23))
        ax = ax.flatten()
        # Visualise confusion matrix
        sns.heatmap(cm, annot=True, fmt='.0f', ax=ax[0],
                    cmap="Dark2", norm=colors.PowerNorm(gamma=0.5),
                    annot_kws={"size": 18})
        ax[0].set_xlabel('Predicted', fontsize=BIG_SIZE);
        ax[0].set_ylabel('Actual', fontsize=BIG_SIZE); 
        ax[0].set_title('Confusion Matrix', fontsize=BIG_SIZE); 
        ax[0].xaxis.set_ticklabels(labels); 
        ax[0].yaxis.set_ticklabels(labels);
        ax[0].tick_params(labelsize=BIG_SIZE)
        
        # Visualise top 15 most important features
        ax[1].barh("feature", "importance",data=important_15)
        ax[1].set_xlabel('Features', fontsize=BIG_SIZE);
        ax[1].set_ylabel('Importance', fontsize=BIG_SIZE);
        ax[1].set_title('Feature Importance', fontsize=BIG_SIZE); 
        ax[1].tick_params(labelsize=BIG_SIZE)
        plt.show()
    return tuned_models, models, time_used, best_cv_scores, \
            roc_test_set, pr_test_set, f1_test_set, ap_test_set, \
            mc_test_set, ck_test_set, acc_test_set, tn_count, fp_count,\
            fn_count, tp_count, fpr_test_set, tpr_test_set, precision_test_set, recall_test_set


def tuning_models_RSCV(X_train, y_train, X_test, y_test, labels, nested_dict_of_models_params, scaler, scoring, refit_measure='F1'):
    """"""
    models = []
    time_used = []
    best_cv_scores = []
    roc_test_set = []
    pr_test_set = []
    f1_test_set = []
    ap_test_set = []
    mc_test_set = []
    ck_test_set = []
    acc_test_set = []
    tn_count = []
    fp_count = []
    fn_count = []
    tp_count = []
    tuned_models = []
    
    for model, details in nested_dict_of_models_params.items():
        models.append(model)
        clf = details['clf']
        params = details['params']        
        
        pipeline = Pipeline([('Transformer', scaler), ('Estimator', clf)])

        start = time.time()
        # Repeated Stratified K-Fold Cross-Validation set:
        cv = StratifiedShuffleSplit(n_splits=15, test_size=0.25, random_state=0)
        grid = RandomizedSearchCV(pipeline, params, cv=cv, n_jobs=-1, scoring=scoring, refit=refit_measure)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test) #model prediction
        y_pred_proba = grid.predict_proba(X_test)[:,1] #prediction probability
        end = time.time()
        used = end - start #time the computation
        tuned_models.append(grid)
        time_used.append(used)
        best_cv_scores.append(grid.best_score_)

        # For Visualisation - Confusion Matrix
        cm = confusion_matrix(y_pred, y_test)
        tn, fp, fn, tp = cm.ravel()
        tn_count.append(tn)
        fp_count.append(fp)
        fn_count.append(fn)
        tp_count.append(tp)

        # For Visualisation - ROC, False Positive Rate, True Positive Rate
        fpr, tpr, thresholds0 = roc_curve(y_test, y_pred_proba)

        # For Visualisation - Precision, Recall
        precision, recall, thresholds1 = precision_recall_curve(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred_proba)
        mc = matthews_corrcoef(y_test, y_pred)
        ck = cohen_kappa_score(y_test, y_pred)
        f1_test_set.append(f1)
        ap_test_set.append(ap)
        mc_test_set.append(mc)
        ck_test_set.append(ck)

        # ROC_AUC, PR_AUC
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        accuracy = (tn + tp)/(tn + fp + tp + fn)
        acc_test_set.append(accuracy)
        roc_test_set.append(roc_auc)
        pr_test_set.append(pr_auc)
        
        print('\n')
        print('--------', model,'Tuned Model---------')
        print('\n')
        # Print Grid Search CV Results
        print("Job took: {}s".format(used))
        print("Set of parameters from Randomised Search CV:\n {}".format(grid.get_params))
        print("Best parameters from Grid Search CV:\n {}".format(grid.best_params_))
        print("Best cross-validation score: {:.4f}".format(grid.best_score_))
        print(cm)
        print("Accuracy: {:.4f}".format((tp+tn)/(tp+tn+fp+fn)))
        print("Recall (FN): {:.4f}".format((tp)/(tp+fn)))
        print("Precision (FP): {:.4f}".format((tp)/(tp+fp)))
        print('F1-score: {0:.3f} \nPR_AUC: {1:.3f} \nROC_AUC: {2:.3f} \n'.
              format(f1, pr_auc, roc_auc))
        print(classification_report(y_test, y_pred))
        print('---------------------------------------\n')
        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        #sns.heatmap(cm, annot=True, fmt='.0f', ax=ax,
        #            cmap="Dark2", norm=colors.PowerNorm(gamma=0.5)
        #            , annot_kws={"size": 14})
        #ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
        #ax.set_title(model+'Confusion Matrix', fontsize=14); 
        #ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
        #plt.show()
    return tuned_models, models, time_used, best_cv_scores, \
            roc_test_set, pr_test_set, f1_test_set, ap_test_set, \
            mc_test_set, ck_test_set, acc_test_set, tn_count, fp_count, fn_count, tp_count


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def highlight_min(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.min()
    return ['background-color: orange' if v else '' for v in is_min]
=======
                    set_index('icustay_id').groupby(level=0).apply(sum)
    return first_24_labs_agg


def plot_feature_importance(model):
    plt.rcParams["figure.figsize"] = (6,4)
    indices = np.argsort(model.feature_importances_)
    #indices = indices[-17:]

    column_names = [X.columns[i] for i in indices]
    n_features = X.iloc[:, indices].shape[1]
    plt.figure()
    plt.title("Final Model Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Feature Importance")
    plt.barh(range(n_features), model.feature_importances_[indices])
    plt.yticks(range(n_features), column_names)
    plt.show()
>>>>>>> 66ac77d4c34783e99ffbb90ac418d26cb8c7da1d
