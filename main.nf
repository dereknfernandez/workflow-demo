nextflow.enable.dsl = 2

include { prepare_data } from './modules/prepare_data.nf'
include { process_features } from './modules/feature_processing.nf'
include { knn; knn_subclass } from './modules/knn.nf'
include { random_forest; random_forest_subclass } from './modules/random_forest.nf'
include { decision_tree; decision_tree_subclass } from './modules/decision_tree.nf'
include { logistic_regression; logistic_regression_subclass } from './modules/log_res.nf'
include { evaluate_models } from './modules/eval.nf'

workflow {
    _datadir = Channel.fromPath("${projectDir}/data/")

    prepare_data(_datadir)
    process_features(prepare_data.out.train_set, prepare_data.out.test_set)

    knn(process_features.out.kbest_20, process_features.out.y1_enco, process_features.out.kbest_val, process_features.out.y1_en_val)
    knn_subclass(process_features.out.kbest_20d, process_features.out.y2_enco, process_features.out.kbest_vald, process_features.out.y2_en_val)
    decision_tree(process_features.out.kbest_20, process_features.out.y1_enco, process_features.out.kbest_val, process_features.out.y1_en_val)
    decision_tree_subclass(process_features.out.kbest_20d, process_features.out.y2_enco, process_features.out.kbest_vald, process_features.out.y2_en_val)
    logistic_regression(process_features.out.kbest_20, process_features.out.y1_enco, process_features.out.kbest_val, process_features.out.y1_en_val)
    logistic_regression_subclass(process_features.out.kbest_20d, process_features.out.y2_enco, process_features.out.kbest_vald, process_features.out.y2_en_val)
    random_forest(process_features.out.kbest_20, process_features.out.y1_enco, process_features.out.kbest_val, process_features.out.y1_en_val)
    random_forest_subclass(process_features.out.kbest_20d, process_features.out.y2_enco, process_features.out.kbest_vald, process_features.out.y2_en_val)

    evaluate_models(
        _datadir, 
        knn.out.knn_model, 
        knn_subclass.out.knn_model_subclass, 
        random_forest.out.rf_model, 
        random_forest_subclass.out.rf_model_subclass, 
        logistic_regression.out.logres_model, 
        logistic_regression_subclass.out.logres_model_subclass, 
        decision_tree.out.dt_model, 
        decision_tree_subclass.out.dt_model_subclass,
        process_features.out.label_encoder1,
        process_features.out.label_encoder2,
        process_features.out.kbest_test_20,
        process_features.out.kbest_test_20d
    )
}
