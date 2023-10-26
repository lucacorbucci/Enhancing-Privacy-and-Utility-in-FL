ulimit -n 4096
rm -rf ../data/imaginette/federated_split
poetry run python ./generate_dataset.py --config baseline.json

poetry run python ./run_test.py --config baseline.json
poetry run python ./run_test.py --config private_2.json
poetry run python ./run_test.py --config p2p+privacy_server_5.json
# poetry run python ./run_test.py --config p2p+privacy_server_6.json
# poetry run python ./run_test.py --config p2p+privacy_server_7.json
# poetry run python ./run_test.py --config p2p+privacy_server_8.json
# poetry run python ./run_test.py --config p2p+privacy_server_9.json
# poetry run python ./run_test.py --config p2p+privacy_server_10.json

