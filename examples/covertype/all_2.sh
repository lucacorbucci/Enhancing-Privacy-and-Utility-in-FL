

cd ./epsilon_05_paper

# fl_public_02_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_02.json --batch_size_server=183 --clipping=1 --local_training_epochs_server=2 --lr_server=0.07112261697131027 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_02.json --batch_size_server=183 --clipping=1 --local_training_epochs_server=2 --lr_server=0.07112261697131027 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_02.json --batch_size_server=183 --clipping=1 --local_training_epochs_server=2 --lr_server=0.07112261697131027 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_02.json --batch_size_server=183 --clipping=1 --local_training_epochs_server=2 --lr_server=0.07112261697131027 --optimizer=adam --seed 3

# p2p_public_02_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=230 --batch_size_server=92 --clipping=2 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.0733838283030458 --lr_server=0.06399715446127491 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=230 --batch_size_server=92 --clipping=2 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.0733838283030458 --lr_server=0.06399715446127491 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=230 --batch_size_server=92 --clipping=2 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.0733838283030458 --lr_server=0.06399715446127491 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.2.json --batch_size_p2p=230 --batch_size_server=92 --clipping=2 --fl_rounds_P2P=5 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.0733838283030458 --lr_server=0.06399715446127491 --optimizer=adam --seed 3

# fl_public_03_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_03.json --batch_size_server=132 --clipping=3 --local_training_epochs_server=2 --lr_server=0.09279340734188246 --optimizer=sgd --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_03.json --batch_size_server=132 --clipping=3 --local_training_epochs_server=2 --lr_server=0.09279340734188246 --optimizer=sgd --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_03.json --batch_size_server=132 --clipping=3 --local_training_epochs_server=2 --lr_server=0.09279340734188246 --optimizer=sgd --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_03.json --batch_size_server=132 --clipping=3 --local_training_epochs_server=2 --lr_server=0.09279340734188246 --optimizer=sgd --seed 3

# p2p_public_03_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=95 --batch_size_server=144 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=4 --lr_p2p=0.023299787948072644 --lr_server=0.02287692068684516 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=95 --batch_size_server=144 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=4 --lr_p2p=0.023299787948072644 --lr_server=0.02287692068684516 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=95 --batch_size_server=144 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=4 --lr_p2p=0.023299787948072644 --lr_server=0.02287692068684516 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.3.json --batch_size_p2p=95 --batch_size_server=144 --clipping=3 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=4 --lr_p2p=0.023299787948072644 --lr_server=0.02287692068684516 --optimizer=adam --seed 2

# fl_public_04_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_04.json --batch_size_server=139 --clipping=2 --local_training_epochs_server=3 --lr_server=0.0913889647545607 --optimizer=sgd --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_04.json --batch_size_server=139 --clipping=2 --local_training_epochs_server=3 --lr_server=0.0913889647545607 --optimizer=sgd --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_04.json --batch_size_server=139 --clipping=2 --local_training_epochs_server=3 --lr_server=0.0913889647545607 --optimizer=sgd --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config private_04.json --batch_size_server=139 --clipping=2 --local_training_epochs_server=3 --lr_server=0.0913889647545607 --optimizer=sgd --seed 2

# p2p_public_04_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=138 --batch_size_server=49 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.06431943929176077 --lr_server=0.01679274932655839 --optimizer=adam --seed 1
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=138 --batch_size_server=49 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.06431943929176077 --lr_server=0.01679274932655839 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=138 --batch_size_server=49 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.06431943929176077 --lr_server=0.01679274932655839 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/covertype/epsilon_05_paper/run_test.py --config p2p+server_0.4.json --batch_size_p2p=138 --batch_size_server=49 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=2 --local_training_epochs_server=2 --lr_p2p=0.06431943929176077 --lr_server=0.01679274932655839 --optimizer=adam --seed 3
