process random_forest {
  input:
    path kbest_20
    path y1_enco
    path kbest_val
    path y1_en_val
  output:
    path 'rf.pkl', emit: rf_model
    
  script:
    """
    python3 - <<EOF
    import joblib
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    result_kbest_20=pd.read_csv('${kbest_20}')
    y1_enco=pd.read_csv('${y1_enco}')
    result_kbest_val=pd.read_csv('${kbest_val}')
    y1_en_val=pd.read_csv('${y1_en_val}')

    rfc = RandomForestClassifier(n_estimators=1800,max_depth=20,max_features='sqrt',bootstrap=False, min_samples_leaf=2, min_samples_split=10,random_state=42)
    rfc.fit(result_kbest_20,y1_enco)
    cal_clf = CalibratedClassifierCV(rfc, method="sigmoid")
    cal_clf.fit(result_kbest_20,y1_enco)
    predict_y =cal_clf.predict(result_kbest_20)
    print ('The train f1_macro is:',f1_score(y1_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_val)
    print('The cross validation f1_macro is:',f1_score(y1_en_val, predict_y,average='macro'))
    joblib.dump(cal_clf, 'rf.pkl')
    EOF
    """
}

process random_forest_subclass {
  input:
    path kbest_20d
    path y2_enco
    path kbest_vald
    path y2_en_val
  output:
    path 'rf_d.pkl', emit: rf_model_subclass
    
  script:
    """
    python3 - <<EOF
    import joblib
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    result_kbest_20d=pd.read_csv('${kbest_20d}')
    y2_enco=pd.read_csv('${y2_enco}')
    result_kbest_vald=pd.read_csv('${kbest_vald}')
    y2_en_val=pd.read_csv('${y2_en_val}')

    rfc1= RandomForestClassifier(n_estimators=500,max_depth=30,min_samples_leaf=2,min_samples_split=5,bootstrap=False,random_state=42)
    rfc1.fit(result_kbest_20d,y2_enco)
    cal_clf = CalibratedClassifierCV(rfc1, method="sigmoid")
    cal_clf.fit(result_kbest_20d,y2_enco)
    predict_y =cal_clf.predict(result_kbest_20d)
    print ('The train f1_macro is:',f1_score(y2_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_vald)
    print('The cross validation f1_macro is:',f1_score(y2_en_val, predict_y,average='macro'))
    joblib.dump(cal_clf, 'rf_d.pkl')
    EOF
    """
}