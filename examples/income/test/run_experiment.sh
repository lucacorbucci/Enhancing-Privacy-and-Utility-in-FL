rm -rf ../data/income/federated_data/*
poetry run python generate_dataset.py --config baseline.json
poetry run python ./run_test.py --config baseline.json