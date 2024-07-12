process prepare_data {
  input:
    path _datadir
  output:
    path "train_processed.csv", emit: train_set
    path "test_processed.csv", emit: test_set

  script:
	"""
    python3 - <<EOF
    import os
    import pandas as pd
    import numpy as np

    train=pd.read_csv('${_datadir}/train_genetic_disorders.csv')
    test=pd.read_csv('${_datadir}/test_genetic_disorders.csv')
    
    data_train=train.copy()
    data_test=test.copy()
    
    data_train[data_train.isnull().all(1)].shape
    print("Train set null rows: ",data_train[data_train.isnull().all(1)].shape[0])
    
    data_test[data_test.isnull().all(1)].shape
    print("Test set null rows: ",data_test[data_test.isnull().all(1)].shape[0])

    train_nonulls=data_train[data_train.isnull().all(1)!=True]
    test_nonulls=data_test[data_test.isnull().all(1)!=True]
    train_nonulls.columns = train_nonulls.columns.str.replace("'s", "")
    test_nonulls.columns = test_nonulls.columns.str.replace("'s", "")

    train_nonulls.drop(columns=["Patient Id", "Patient First Name", "Family Name", "Father name", "Institute Name", "Location of Institute", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Parental consent"], inplace=True)
    test_nonulls.drop(columns=["Patient Id", "Patient First Name", "Family Name", "Father name", "Institute Name", "Location of Institute", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Parental consent"], inplace=True)

    train_nonulls.rename(columns={"Genes in mother side":'defective_mother',
    'Inherited from father':'defective_father',
    'Maternal gene':'maternal_gene','Paternal gene':'paternal_gene',
    'Respiratory Rate (breaths/min)':'respiratory_rate','Heart Rate (rates/min':'heart_rate',
    'Parental consent':'parental_consent','Follow-up':'follow_up','Birth asphyxia':'birth_asphyxia',
    'Autopsy shows birth defect (if applicable)':'birth_defect_autopsy','Place of birth':'birth_place',
    'Folic acid details (peri-conceptional)':'folic_acid_periconceptional',
    'H/O serious maternal illness':'maternal_illness','H/O radiation exposure (x-ray)':'radiation_exposure',
    'H/O substance abuse':'substance_abuse','Assisted conception IVF/ART':'assisted_conception',
    'History of anomalies in previous pregnancies':'previous_pregnancy_anomalies',
    'Birth defects':'birth_defects','Blood test result':'blood_test_result','Genetic Disorder':'genetic_disorder',
    'Disorder Subclass':'disorder_subclass','Patient Age':'patient_age','Blood cell count (mcL)':'blood_cell_count',
    "Mother age":'mother_age',"Father age":'father_age','No. of previous abortion':'num_previous_abortion',
    'White Blood cell count (thousand per microliter)':'WBC_count'}, inplace=True)

    test_nonulls.rename(columns={"Genes in mother side":'defective_mother',
    'Inherited from father':'defective_father',
    'Maternal gene':'maternal_gene','Paternal gene':'paternal_gene',
    'Respiratory Rate (breaths/min)':'respiratory_rate','Heart Rate (rates/min':'heart_rate',
    'Parental consent':'parental_consent','Follow-up':'follow_up','Birth asphyxia':'birth_asphyxia',
    'Autopsy shows birth defect (if applicable)':'birth_defect_autopsy','Place of birth':'birth_place',
    'Folic acid details (peri-conceptional)':'folic_acid_periconceptional',
    'H/O serious maternal illness':'maternal_illness','H/O radiation exposure (x-ray)':'radiation_exposure',
    'H/O substance abuse':'substance_abuse','Assisted conception IVF/ART':'assisted_conception',
    'History of anomalies in previous pregnancies':'previous_pregnancy_anomalies',
    'Birth defects':'birth_defects','Blood test result':'blood_test_result','Genetic Disorder':'genetic_disorder',
    'Disorder Subclass':'disorder_subclass','Patient Age':'patient_age','Blood cell count (mcL)':'blood_cell_count',
    "Mother age":'mother_age',"Father age":'father_age','No. of previous abortion':'num_previous_abortion',
    'White Blood cell count (thousand per microliter)':'WBC_count'}, inplace=True)
    
    train_nonulls.iloc[:,-2].isnull().sum(),train_nonulls.iloc[:,-1].isnull().sum()
    train_nonulls=train_nonulls[(train_nonulls['genetic_disorder'].isnull()!=True)&(train_nonulls['disorder_subclass'].isnull()!=True)]

    train_nonulls.to_csv('train_processed.csv', index=False)
    test_nonulls.to_csv('test_processed.csv', index=False)
    EOF
    """
}