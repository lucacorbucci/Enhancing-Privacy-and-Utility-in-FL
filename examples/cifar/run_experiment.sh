ulimit -n 4096
rm -rf ../data/cifar/federated_split
poetry run python ./generate_dataset.py --config test.json
 
poetry run python ./run_test.py --config test.json