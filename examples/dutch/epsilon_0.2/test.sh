
rm -rf ../data/dutch/federated_data
mkdir ../data/dutch/federated_data
poetry run python ./generate_dataset.py --config baseline_02.json
poetry run python run_test.py --config baseline_02.json --batch_size_server=202 --local_training_epochs_server=4 --lr_server=0.00659118218338426 --optimizer=adam