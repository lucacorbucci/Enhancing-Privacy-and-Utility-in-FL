rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config test.json
 
poetry run python ./run_test.py --config test.json