ulimit -n 4096
rm -rf ../data/mnist/federated_data
mkdir ../data/mnist/federated_data
poetry run python ./generate_dataset.py --config baseline_02.json
poetry run python ./run_test.py --config baseline_02.json
