process knn {
  input:
    path kbest_20
    path y1_enco
    path kbest_val
    path y1_en_val
  output:
    path 'knn.pkl', emit: knn_model
  
  publishDir path: "${launchDir}/outputs/models", mode: 'copy'
    
  script:
    """
    python3 - <<EOF
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier

    result_kbest_20=pd.read_csv('${kbest_20}')
    y1_enco=pd.read_csv('${y1_enco}')
    result_kbest_val=pd.read_csv('${kbest_val}')
    y1_en_val=pd.read_csv('${y1_en_val}')

    nn=[x for x in range(1, 15, 2)]
    cv_f1_macro=[]
    for i in nn:
        knn=KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
        knn.fit(result_kbest_20,y1_enco)
        cal_clf = CalibratedClassifierCV(knn, method="sigmoid")
        cal_clf.fit(result_kbest_20,y1_enco)
        predict_y=cal_clf.predict(result_kbest_val)
        cv_f1_macro.append(f1_score(y1_en_val, predict_y,average='macro'))
    for i in range(len(cv_f1_macro)):
        print ('f1_macro for k = ',nn[i],'is',cv_f1_macro[i])
    best_nn = np.argmax(cv_f1_macro)
    knn=KNeighborsClassifier(n_neighbors=nn[best_nn])
    knn.fit(result_kbest_20,y1_enco)
    cal_clf = CalibratedClassifierCV(knn, method="sigmoid")
    cal_clf.fit(result_kbest_20,y1_enco)

    predict_y =cal_clf .predict(result_kbest_20)
    print ('For values of best nn = ', nn[best_nn], "The train f1_macro is:",f1_score(y1_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_val)
    print('For values of best nn = ', nn[best_nn], "The cross validation f1_macro is:",f1_score(y1_en_val, predict_y,average='macro'))

    joblib.dump(cal_clf, 'knn.pkl')
    EOF
    """
}

process knn_subclass {
  input:
    path kbest_20d
    path y2_enco
    path kbest_vald
    path y2_en_val
  output:
    path 'knn_d.pkl', emit: knn_model_subclass

  publishDir path: "${launchDir}/outputs/models", mode: 'copy'
  
  script:
    """
    python3 - <<EOF
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier

    result_kbest_20d=pd.read_csv('${kbest_20d}')
    y2_enco=pd.read_csv('${y2_enco}')
    result_kbest_vald=pd.read_csv('${kbest_vald}')
    y2_en_val=pd.read_csv('${y2_en_val}')

    nn=[x for x in range(1, 15, 2)]
    cv_f1_macro=[]
    for i in nn:
        knn=KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
        knn.fit(result_kbest_20d,y2_enco)
        cal_clf = CalibratedClassifierCV(knn, method="sigmoid")
        cal_clf.fit(result_kbest_20d,y2_enco)
        predict_y=cal_clf.predict(result_kbest_vald)
        cv_f1_macro.append(f1_score(y2_en_val, predict_y,average='macro'))
    for i in range(len(cv_f1_macro)):
        print ('f1_macro for k = ',nn[i],'is',cv_f1_macro[i])
    best_nn = np.argmax(cv_f1_macro)
    knn=KNeighborsClassifier(n_neighbors=nn[best_nn])
    knn.fit(result_kbest_20d,y2_enco)
    cal_clf = CalibratedClassifierCV(knn, method="sigmoid")
    cal_clf.fit(result_kbest_20d,y2_enco)

    predict_y =cal_clf .predict(result_kbest_20d)
    print ('For values of best nn = ', nn[best_nn], "The train f1_macro is:",f1_score(y2_enco, predict_y,average='macro'))
    predict_y = cal_clf.predict(result_kbest_vald)
    print('For values of best nn = ', nn[best_nn], "The cross validation f1_macro is:",f1_score(y2_en_val, predict_y,average='macro'))

    joblib.dump(cal_clf, 'knn_d.pkl')
    EOF
    """
}