# server

rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config private_03.json

poetry run python /home/l.corbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_03.json --batch_size_server=34 --clipping=1 --local_training_epochs_server=3 --lr_server=0.08776337793359408 --optimizer=sgd

# p2p

# list of sweeps to call

rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config p2p+server_0.3.json
poetry run python /home/l.corbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.3.json --batch_size_p2p=34 --batch_size_server=248 --clipping=9 --fl_rounds_P2P=10 --local_training_epochs_p2p=4 --local_training_epochs_server=3 --lr_p2p=0.0825192074423933 --lr_server=0.009359272295198564 --optimizer=adam