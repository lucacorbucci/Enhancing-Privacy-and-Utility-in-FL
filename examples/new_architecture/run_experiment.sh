ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config private_1.json

# poetry run python ./run_test.py --config baseline.json
# poetry run python ./run_test.py --config private_1.json
# poetry run python ./run_test.py --config private_2.json
poetry run python ./run_test.py --config p2p+privacy_server.json

# poetry run python ./run_test.py --config privacy_server.json
# poetry run python ./run_test.py --config p2p+privacy_server.json
# poetry run python ./run_test.py --config privacy_server_p2p_equal.json
# poetry run python ./run_test.py --config privacy_server_different.json

# ulimit -n 4096
# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config mnist_baseline_2.json
 
# poetry run python ./run_test.py --config mnist_baseline_2.json
# poetry run python ./run_test.py --config privacy_server_2.json
# poetry run python ./run_test.py --config privacy_server_p2p_equal_2.json
# poetry run python ./run_test.py --config privacy_server_different_2.json


# # ulimit -n 4096
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config p2p+privacy_server_test.json
 
# poetry run python ./run_test.py --config p2p+privacy_server_test.json


