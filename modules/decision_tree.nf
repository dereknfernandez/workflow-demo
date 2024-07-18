process decision_tree {
  input:
    path kbest_20
    path y1_enco
    path kbest_val
    path y1_en_val
  output:
    path 'dt.pkl', emit: dt_model
  
  publishDir path: "${launchDir}/outputs/models", mode: 'copy'

  script:
    """
    python3 - <<EOF
    import joblib
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.tree import DecisionTreeClassifier

    result_kbest_20=pd.read_csv('${kbest_20}')
    y1_enco=pd.read_csv('${y1_enco}')
    result_kbest_val=pd.read_csv('${kbest_val}')
    y1_en_val=pd.read_csv('${y1_en_val}')

    DT = DecisionTreeClassifier(max_depth=20,min_samples_leaf=50,random_state=42)
    DT.fit(result_kbest_20,y1_enco)
    cal_clf = CalibratedClassifierCV(DT, method="sigmoid")
    cal_clf.fit(result_kbest_20,y1_enco)
    predict_y =cal_clf .predict(result_kbest_20)
    print ('The train f1_macro is:',f1_score(y1_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_val)
    print('The cross validation f1_macro is:',f1_score(y1_en_val, predict_y,average='macro'))

    joblib.dump(cal_clf, 'dt.pkl')
    EOF
    """
}

process decision_tree_subclass {
  input:
    path kbest_20d
    path y2_enco
    path kbest_vald
    path y2_en_val
  output:
    path 'dt_d.pkl', emit: dt_model_subclass
  
  publishDir path: "${launchDir}/outputs/models", mode: 'copy'
    
  script:
    """
    python3 - <<EOF
    import joblib
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.tree import DecisionTreeClassifier

    result_kbest_20d=pd.read_csv('${kbest_20d}')
    y2_enco=pd.read_csv('${y2_enco}')
    result_kbest_vald=pd.read_csv('${kbest_vald}')
    y2_en_val=pd.read_csv('${y2_en_val}')

    DT = DecisionTreeClassifier(max_depth=20,min_samples_leaf=10,random_state=42)
    DT.fit(result_kbest_20d,y2_enco)
    cal_clf = CalibratedClassifierCV(DT, method="sigmoid")
    cal_clf.fit(result_kbest_20d,y2_enco)
    predict_y =cal_clf .predict(result_kbest_20d)
    print ('The train f1_macro is:',f1_score(y2_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_vald)
    print('The cross validation f1_macro is:',f1_score(y2_en_val, predict_y,average='macro'))
    
    joblib.dump(cal_clf, 'dt_d.pkl')
    EOF
    """
}