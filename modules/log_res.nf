process logistic_regression {
  input:
    path kbest_20
    path y1_enco
    path kbest_val
    path y1_en_val
  output:
    path 'logres.pkl', emit: logres_model
    
  script:
    """
    python3 - <<EOF
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    result_kbest_20=pd.read_csv('${kbest_20}')
    y1_enco=pd.read_csv('${y1_enco}')
    result_kbest_val=pd.read_csv('${kbest_val}')
    y1_en_val=pd.read_csv('${y1_en_val}')

    C1= [10 ** x for x in range(-5, 4)]
    cv_f1_macro=[]
    for i in C1:
        logisticR=LogisticRegression(penalty='l2',C=i,class_weight='balanced')
        logisticR.fit(result_kbest_20,y1_enco)
        cal_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
        cal_clf.fit(result_kbest_20,y1_enco)
        predict_y=cal_clf.predict(result_kbest_val)
        cv_f1_macro.append(f1_score(y1_en_val, predict_y,average='macro'))
    for i in range(len(cv_f1_macro)):
        print ('f1_macro for k = ',C1[i],'is',cv_f1_macro[i])
    best_C1 = np.argmax(cv_f1_macro)
    logisticR=LogisticRegression(penalty='l2',C=C1[best_C1],class_weight='balanced')
    logisticR.fit(result_kbest_20,y1_enco)
    cal_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
    cal_clf.fit(result_kbest_20,y1_enco)

    predict_y =cal_clf .predict(result_kbest_20)
    print ('For values of best C = ',C1[best_C1], "The train f1_macro is:",f1_score(y1_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_val)
    print('For values of best C = ',C1[best_C1], "The cross validation f1_macro is:",f1_score(y1_en_val, predict_y,average='macro'))
    joblib.dump(cal_clf, 'logres.pkl')
    EOF
    """
}

process logistic_regression_subclass {
  input:
    path kbest_20d
    path y2_enco
    path kbest_vald
    path y2_en_val
  output:
    path 'logres_d.pkl', emit: logres_model_subclass

  script:
    """
    python3 - <<EOF
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    result_kbest_20d=pd.read_csv('${kbest_20d}')
    y2_enco=pd.read_csv('${y2_enco}')
    result_kbest_vald=pd.read_csv('${kbest_vald}')
    y2_en_val=pd.read_csv('${y2_en_val}')

    C1= [10 ** x for x in range(-5, 4)]
    cv_f1_macro=[]
    for i in C1:
        logisticR=LogisticRegression(penalty='l2',C=i,class_weight='balanced',max_iter=1000)
        logisticR.fit(result_kbest_20d,y2_enco)
        cal_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
        cal_clf.fit(result_kbest_20d,y2_enco)
        predict_y=cal_clf.predict(result_kbest_vald)
        cv_f1_macro.append(f1_score(y2_en_val, predict_y,average='macro'))
    for i in range(len(cv_f1_macro)):
        print ('f1_macro for C = ',C1[i],'is',cv_f1_macro[i])
    best_C1 = np.argmax(cv_f1_macro)
    logisticR=LogisticRegression(penalty='l2',C=C1[best_C1],class_weight='balanced',max_iter=1000)
    logisticR.fit(result_kbest_20d,y2_enco)
    cal_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
    cal_clf.fit(result_kbest_20d,y2_enco)

    predict_y =cal_clf .predict(result_kbest_20d)
    print ('For values of best C = ',C1[best_C1], "The train f1_macro is:",f1_score(y2_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_vald)
    print('For values of best C = ',C1[best_C1], "The cross validation f1_macro is:",f1_score(y2_en_val, predict_y,average='macro'))

    joblib.dump(cal_clf, 'logres_d.pkl')
    EOF
    """
}