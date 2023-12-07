ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config p2p+server_0.2.json

# poetry run python run_test.py --config private_08.json --batch_size_p2p=497 --local_training_epochs_p2p=2 --lr_p2p=0.01904348329495041 --optimizer=adam
# poetry run python ./run_test.py --config p2p+server_0.8.json
# poetry run python ./run_test.py --config inverted_majority_p2p_only_8_3.json



