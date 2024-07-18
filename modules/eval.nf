process evaluate_models {
  input:
    path test_dataset
    path knn_model
    path knn_model_subclass
    path logres_model
    path logres_model_subclass
    path dt_model
    path dt_model_subclass
    path label_encoder1
    path label_encoder2
    path kbest_test_20
    path kbest_test_20d
  
  output:
    path 'eval_knn.csv'
    path 'eval_dt.csv'
    path 'eval_logres.csv'
  
  publishDir path: "${launchDir}/outputs", mode: 'copy'
    
  script:
    """
    python3 - <<EOF
    import joblib
    import pandas as pd
    
    test=pd.read_csv('${test_dataset}')

    knn=joblib.load('${knn_model}')
    knn_d=joblib.load('${knn_model_subclass}')
    logres=joblib.load('${logres_model}')
    logres_d=joblib.load('${logres_model_subclass}')
    dt=joblib.load('${dt_model}')
    dt_d=joblib.load('${dt_model_subclass}')
    lab_enc1=joblib.load('${label_encoder1}')
    lab_enc2=joblib.load('${label_encoder2}')

    result_kbest_test20=pd.read_csv('${kbest_test_20}')
    result_kbest_test20d=pd.read_csv('${kbest_test_20d}')

    data_test=test[test.isnull().all(1)!=True]
    ids=data_test['Patient Id']

    knn_pred=knn.predict(result_kbest_test20)
    knn_subclass_pred=knn_d.predict(result_kbest_test20d)
    gen_disorder_knn=lab_enc1.inverse_transform(knn_pred)
    disorder_subclass_knn=lab_enc2.inverse_transform(knn_subclass_pred)
    output=pd.DataFrame({'Patient Id': ids,'Genetic_Disorder':gen_disorder_knn,'Disorder_Subclass':disorder_subclass_knn})
    output.to_csv('eval_knn.csv',index=False)

    logres_pred=logres.predict(result_kbest_test20)
    logres_subclass_pred=logres_d.predict(result_kbest_test20d)
    gen_disorder_logres=lab_enc1.inverse_transform(logres_pred)
    disorder_subclass_logres=lab_enc2.inverse_transform(logres_subclass_pred)
    output=pd.DataFrame({'Patient Id': ids,'Genetic_Disorder':gen_disorder_logres,'Disorder_Subclass':disorder_subclass_logres})
    output.to_csv('eval_logres.csv',index=False)

    dt_pred=dt.predict(result_kbest_test20)
    dt_subclass_pred=dt_d.predict(result_kbest_test20d)
    gen_disorder_dt=lab_enc1.inverse_transform(dt_pred)
    disorder_subclass_dt=lab_enc2.inverse_transform(dt_subclass_pred)
    output=pd.DataFrame({'Patient Id': ids,'Genetic_Disorder':gen_disorder_dt,'Disorder_Subclass':disorder_subclass_dt})
    output.to_csv('eval_dt.csv',index=False)
    EOF
    """
}