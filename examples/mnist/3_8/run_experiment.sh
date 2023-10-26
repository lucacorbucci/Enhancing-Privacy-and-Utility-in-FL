ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config p2p+server.json

# poetry run python ./run_test.py --config baseline.json
# poetry run python ./run_test.py --config private.json
poetry run python ./run_test.py --config p2p+server.json