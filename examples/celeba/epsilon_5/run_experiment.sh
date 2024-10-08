
rm -rf ../data/celeba/federated_split
poetry run python ./generate_dataset.py --config p2p+server_0.2.json
# poetry run python ./run_test.py --config p2p+server_0.2.json