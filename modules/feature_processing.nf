process process_features {
  input:
    path train_processed
    path test_processed
  output:
    path 'y2_en_val.csv', emit: y2_en_val
    path 'y1_en_val.csv', emit: y1_en_val
    path 'y2_enco.csv', emit: y2_enco
    path 'y1_enco.csv', emit: y1_enco
    path 'result_kbest_20d.csv', emit: kbest_20d
    path 'result_kbest_test20d.csv', emit: kbest_test_20d
    path 'result_kbest_vald.csv', emit: kbest_vald
    path 'result_kbest_20.csv', emit: kbest_20
    path 'result_kbest_test20.csv', emit: kbest_test_20
    path 'result_kbest_val.csv', emit: kbest_val
    path 'label_encoder2.pkl', emit: label_encoder2
    path 'label_encoder1.pkl', emit: label_encoder1

  script:
    """
    python3 - <<EOF
    import joblib
    import pandas as pd
    import numpy as np
    from imblearn.over_sampling import BorderlineSMOTE
    from sklearn.feature_selection import SelectKBest,chi2
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler
    
    train=pd.read_csv('${train_processed}')
    test=pd.read_csv('${test_processed}')

    print("Train set shape: ", train.shape)
    print("Test set shape: ", test.shape)

    X=train.iloc[:,:-2]
    y1=train.iloc[:,-2]
    y2=train.iloc[:,-1]
    X_test=test

    for i in X_test.columns:
        if X_test[i].dtype!=X[i].dtype:
            X_test[i]=X_test[i].astype(X[i].dtype.name)

    X_test=X_test.replace('-99',np.nan)
    X['radiation_exposure']=X['radiation_exposure'].replace('-','others')
    X['substance_abuse']=X['substance_abuse'].replace('-','others')
    X_test['radiation_exposure']=X_test['radiation_exposure'].replace('-','others')
    X_test['substance_abuse']=X_test['substance_abuse'].replace('-','others')
    X_test['WBC_count']=X_test['WBC_count'].mask(X_test['WBC_count']<0,np.nan)
    X_test['num_previous_abortion']=X_test['num_previous_abortion'].mask(X_test['num_previous_abortion']<0,np.nan)

    X_train1,X_val1,y_train1,y_val1= train_test_split(X,y1,stratify=y1,test_size=0.20)
    X_train2,X_val2,y_train2,y_val2= train_test_split(X,y2,stratify=y2,test_size=0.20)
    
    imp_mode=LabelEncoder()
    imp_mode_num=SimpleImputer(strategy='most_frequent')
    imp_median=SimpleImputer(strategy='median')
    pd.options.mode.chained_assignment = None

    for i in X.columns:
        if (X[i].dtype.name!='object')&(X[i].nunique()<=3):
            imp_mode_num.fit(np.array(X_train1[i]).reshape(-1,1))
            X_train1[i]=imp_mode_num.transform(np.array(X_train1[i]).reshape(-1,1))
            X_val1[i]=imp_mode_num.transform(np.array(X_val1[i]).reshape(-1,1))
            X_test[i]=imp_mode_num.transform(np.array(X_test[i]).reshape(-1,1))
        elif (X[i].dtype.name!='object')&(X[i].nunique()>3):
            imp_median.fit(np.array(X_train1[i]).reshape(-1,1))
            X_train1[i]=imp_median.transform(np.array(X_train1[i]).reshape(-1,1))
            X_val1[i]=imp_median.transform(np.array(X_val1[i]).reshape(-1,1))
            X_test[i]=imp_median.transform(np.array(X_test[i]).reshape(-1,1))
        else:
            imp_mode.fit(X_train1[i])
            X_train1[i]=imp_mode.transform(X_train1[i])
            X_val1[i]=imp_mode.transform(X_val1[i])
            X_test[i]=imp_mode.transform(X_test[i])

    ord_enc=OrdinalEncoder()
    ohe_enc=OneHotEncoder()
    min_max=MinMaxScaler()

    X_train1.reset_index(inplace=True)
    X_val1.reset_index(inplace=True)

    for i in X.columns:
        if (X[i].dtype.name=='object'):
            if i in X and X[i].nunique()<=2:
                ord_enc.fit(np.array(X_train1[i]).reshape(-1,1))
                X_train1.loc[:,i]=ord_enc.transform(np.array(X_train1[i]).reshape(-1,1))
                X_val1.loc[:,i]=ord_enc.transform(np.array(X_val1[i]).reshape(-1,1))
                X_test.loc[:,i]=ord_enc.transform(np.array(X_test[i]).reshape(-1,1))
            else:
                ohe_enc.fit(np.array(X_train1[i]).reshape(-1,1))
                X_encode_tr1=pd.DataFrame(ohe_enc.transform(np.array(X_train1[i]).reshape(-1,1)).toarray(),columns=ohe_enc.get_feature_names_out([i]))
                X_encode_va1=pd.DataFrame(ohe_enc.transform(np.array(X_val1[i]).reshape(-1,1)).toarray(),columns=ohe_enc.get_feature_names_out([i]))
                X_encode1=pd.DataFrame(ohe_enc.transform(np.array(X_test[i]).reshape(-1,1)).toarray(),columns=ohe_enc.get_feature_names_out([i]))
                X_train1=pd.concat([X_train1,X_encode_tr1],axis=1)
                X_val1=pd.concat([X_val1,X_encode_va1],axis=1)
                X_test=pd.concat([X_test,X_encode1],axis=1)
                X_train1.drop(columns=[i],inplace=True)
                X_val1.drop(columns=[i],inplace=True)
                X_test.drop(columns=[i],inplace=True)
    
    X_train1.drop(columns='index',inplace=True)
    X_val1.drop(columns='index',inplace=True)

    X2=min_max.fit_transform(X_train1)
    X2=pd.DataFrame(X2,columns=X_train1.columns)
    X2_val=min_max.transform(X_val1)
    X2_val=pd.DataFrame(X2_val,columns=X_val1.columns)

    X2_test=min_max.transform(X_test)
    X2_test=pd.DataFrame(X2_test,columns=X_test.columns)

    lab_enc1=LabelEncoder()
    y1_en=lab_enc1.fit_transform(y_train1)
    y1_en_val=lab_enc1.transform(y_val1)
    lab_enc2=LabelEncoder()
    y2_en=lab_enc2.fit_transform(y_train2)
    y2_en_val=lab_enc2.transform(y_val2)

    sm = BorderlineSMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X2, pd.DataFrame(y1_en))
    y_sm.value_counts(normalize=True) * 100

    y1_enco=np.array(y_sm).ravel()
    sel1=SelectKBest(chi2, k=25).fit(X_sm,y1_enco)
    cols=sel1.get_support(indices=True)
    result_kbest_20=X_sm.iloc[:,cols]

    result_kbest_val=X2_val.iloc[:,cols]
    result_kbest_test20=X2_test.iloc[:,cols]

    X_smd, y_smd = sm.fit_resample(X2, pd.DataFrame(y2_en))
    y_smd.value_counts(normalize=True) * 100
    y2_enco=np.array(y_smd).ravel()
    sel2=SelectKBest(chi2, k=25).fit(X_smd,y2_enco)
    cols=sel2.get_support(indices=True)
    result_kbest_20d=X_smd.iloc[:,cols]
    result_kbest_vald=X2_val.iloc[:,cols]
    result_kbest_test20d=X2_test.iloc[:,cols]

    y2_en_val_df = pd.DataFrame(y2_en_val)
    y2_en_val_df.to_csv('y2_en_val.csv', index=False)
    y1_en_val_df = pd.DataFrame(y1_en_val)
    y1_en_val_df.to_csv('y1_en_val.csv', index=False)
    y1_enco_df = pd.DataFrame(y1_enco)
    y1_enco_df.to_csv('y1_enco.csv', index=False)
    y2_enco_df = pd.DataFrame(y2_enco)
    y2_enco_df.to_csv('y2_enco.csv', index=False)
    result_kbest_20d_df = pd.DataFrame(result_kbest_20d)
    result_kbest_20d_df.to_csv('result_kbest_20d.csv', index=False)
    result_kbest_vald_df = pd.DataFrame(result_kbest_vald)
    result_kbest_vald_df.to_csv('result_kbest_vald.csv', index=False)
    result_kbest_test20d_df = pd.DataFrame(result_kbest_test20d)
    result_kbest_test20d_df.to_csv('result_kbest_test20d.csv', index=False)
    result_kbest_20_df = pd.DataFrame(result_kbest_20)
    result_kbest_20_df.to_csv('result_kbest_20.csv', index=False)
    result_kbest_val_df = pd.DataFrame(result_kbest_val)
    result_kbest_val_df.to_csv('result_kbest_val.csv', index=False)
    result_kbest_test20_df = pd.DataFrame(result_kbest_test20)
    result_kbest_test20_df.to_csv('result_kbest_test20.csv', index=False)
    joblib.dump(lab_enc1, 'label_encoder1.pkl')
    joblib.dump(lab_enc2, 'label_encoder2.pkl')

    EOF
    """
}