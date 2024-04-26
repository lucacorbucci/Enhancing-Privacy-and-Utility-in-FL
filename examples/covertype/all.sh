
cd ./epsilon_1_paper

# p2p_public_02_epsilon_1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=152 --batch_size_server=66 --clipping=1 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.0719329259774692 --lr_server=0.01272869703785372 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=152 --batch_size_server=66 --clipping=1 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.0719329259774692 --lr_server=0.01272869703785372 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=152 --batch_size_server=66 --clipping=1 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.0719329259774692 --lr_server=0.01272869703785372 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=152 --batch_size_server=66 --clipping=1 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.0719329259774692 --lr_server=0.01272869703785372 --optimizer=adam --seed 3

# fl_public_02_epsilon_1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_02.json --batch_size_server=240 --clipping=2 --local_training_epochs_server=3 --lr_server=0.04614756755204003 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_02.json --batch_size_server=240 --clipping=2 --local_training_epochs_server=3 --lr_server=0.04614756755204003 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_02.json --batch_size_server=240 --clipping=2 --local_training_epochs_server=3 --lr_server=0.04614756755204003 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_02.json --batch_size_server=240 --clipping=2 --local_training_epochs_server=3 --lr_server=0.04614756755204003 --optimizer=adam --seed 2



# fl_public_03_epsilon_1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_03.json --batch_size_server=208 --clipping=2 --local_training_epochs_server=1 --lr_server=0.05453640797866734 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_03.json --batch_size_server=208 --clipping=2 --local_training_epochs_server=1 --lr_server=0.05453640797866734 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_03.json --batch_size_server=208 --clipping=2 --local_training_epochs_server=1 --lr_server=0.05453640797866734 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_03.json --batch_size_server=208 --clipping=2 --local_training_epochs_server=1 --lr_server=0.05453640797866734 --optimizer=adam --seed 0

# p2p_public_03_epsilon_1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=103 --batch_size_server=200 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.05575248010936919 --lr_server=0.05332468733742696 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=103 --batch_size_server=200 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.05575248010936919 --lr_server=0.05332468733742696 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=103 --batch_size_server=200 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.05575248010936919 --lr_server=0.05332468733742696 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=103 --batch_size_server=200 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.05575248010936919 --lr_server=0.05332468733742696 --optimizer=adam --seed 3

# fl_public_04_epsilon_1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_04.json --batch_size_server=48 --clipping=3 --local_training_epochs_server=4 --lr_server=0.013801867455305384 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_04.json --batch_size_server=48 --clipping=3 --local_training_epochs_server=4 --lr_server=0.013801867455305384 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_04.json --batch_size_server=48 --clipping=3 --local_training_epochs_server=4 --lr_server=0.013801867455305384 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config private_04.json --batch_size_server=48 --clipping=3 --local_training_epochs_server=4 --lr_server=0.013801867455305384 --optimizer=adam --seed 3

# p2p_public_04_epsilon_1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=208 --batch_size_server=248 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06913864763538252 --lr_server=0.07133592778805956 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=208 --batch_size_server=248 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06913864763538252 --lr_server=0.07133592778805956 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=208 --batch_size_server=248 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06913864763538252 --lr_server=0.07133592778805956 --optimizer=adam --seed 4
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_1_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=208 --batch_size_server=248 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06913864763538252 --lr_server=0.07133592778805956 --optimizer=adam --seed 5
