ulimit -n 4096
rm -rf ../data/celeba/federated_split
poetry run python ./generate_dataset.py --config baseline.json

# poetry run python ./run_test.py --config p2p+private.json
# poetry run python ./run_test.py --config inverted_majority_p2p_only_8_3.json



