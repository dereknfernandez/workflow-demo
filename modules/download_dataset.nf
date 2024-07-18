process download_data {
    output:
    path 'train_genetic_disorders.csv', emit: train_dataset
    path 'test_genetic_disorders.csv', emit: test_dataset

    script:
    """
    curl -o test_genetic_disorders.csv https://raw.githubusercontent.com/dereknfernandez/workflow-demo/main/data/test_genetic_disorders.csv
    curl -o train_genetic_disorders.csv https://raw.githubusercontent.com/dereknfernandez/workflow-demo/main/data/train_genetic_disorders.csv
    """
}
