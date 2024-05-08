
cd ./epsilon_0.5

# fl_public_02_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_02.json --batch_size_server=73 --clipping=1 --local_training_epochs_server=4 --lr_server=0.05919886415586026 --optimizer=sgd --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_02.json --batch_size_server=73 --clipping=1 --local_training_epochs_server=4 --lr_server=0.05919886415586026 --optimizer=sgd --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_02.json --batch_size_server=73 --clipping=1 --local_training_epochs_server=4 --lr_server=0.05919886415586026 --optimizer=sgd --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_02.json --batch_size_server=73 --clipping=1 --local_training_epochs_server=4 --lr_server=0.05919886415586026 --optimizer=sgd --seed 4

# p2p_public_02_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.2.json --batch_size_p2p=45 --batch_size_server=191 --clipping=1 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=1 --lr_p2p=0.050728572252821395 --lr_server=0.0483909609685357 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.2.json --batch_size_p2p=45 --batch_size_server=191 --clipping=1 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=1 --lr_p2p=0.050728572252821395 --lr_server=0.0483909609685357 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.2.json --batch_size_p2p=45 --batch_size_server=191 --clipping=1 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=1 --lr_p2p=0.050728572252821395 --lr_server=0.0483909609685357 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.2.json --batch_size_p2p=45 --batch_size_server=191 --clipping=1 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=1 --lr_p2p=0.050728572252821395 --lr_server=0.0483909609685357 --optimizer=adam --seed 4

# fl_public_03_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_03.json --batch_size_server=229 --clipping=2 --local_training_epochs_server=3 --lr_server=0.06340392748137641 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_03.json --batch_size_server=229 --clipping=2 --local_training_epochs_server=3 --lr_server=0.06340392748137641 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_03.json --batch_size_server=229 --clipping=2 --local_training_epochs_server=3 --lr_server=0.06340392748137641 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_03.json --batch_size_server=229 --clipping=2 --local_training_epochs_server=3 --lr_server=0.06340392748137641 --optimizer=adam --seed 4

# p2p_public_03_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.3.json --batch_size_p2p=226 --batch_size_server=195 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=3 --lr_p2p=0.016055960728161497 --lr_server=0.057510981025277054 --optimizer=sgd --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.3.json --batch_size_p2p=226 --batch_size_server=195 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=3 --lr_p2p=0.016055960728161497 --lr_server=0.057510981025277054 --optimizer=sgd --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.3.json --batch_size_p2p=226 --batch_size_server=195 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=3 --lr_p2p=0.016055960728161497 --lr_server=0.057510981025277054 --optimizer=sgd --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.3.json --batch_size_p2p=226 --batch_size_server=195 --clipping=2 --fl_rounds_P2P=2 --local_training_epochs_p2p=1 --local_training_epochs_server=3 --lr_p2p=0.016055960728161497 --lr_server=0.057510981025277054 --optimizer=sgd --seed 4

# fl_public_04_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_04.json --batch_size_server=51 --clipping=1 --local_training_epochs_server=4 --lr_server=0.06356529248294597 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_04.json --batch_size_server=51 --clipping=1 --local_training_epochs_server=4 --lr_server=0.06356529248294597 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_04.json --batch_size_server=51 --clipping=1 --local_training_epochs_server=4 --lr_server=0.06356529248294597 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config private_04.json --batch_size_server=51 --clipping=1 --local_training_epochs_server=4 --lr_server=0.06356529248294597 --optimizer=adam --seed 4

# p2p_public_04_epsilon_05
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.4.json --batch_size_p2p=79 --batch_size_server=189 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=1 --local_training_epochs_server=4 --lr_p2p=0.08435756082060261 --lr_server=0.03389068878750167 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.4.json --batch_size_p2p=79 --batch_size_server=189 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=1 --local_training_epochs_server=4 --lr_p2p=0.08435756082060261 --lr_server=0.03389068878750167 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.4.json --batch_size_p2p=79 --batch_size_server=189 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=1 --local_training_epochs_server=4 --lr_p2p=0.08435756082060261 --lr_server=0.03389068878750167 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_test.py --config p2p+server_0.4.json --batch_size_p2p=79 --batch_size_server=189 --clipping=1 --fl_rounds_P2P=4 --local_training_epochs_p2p=1 --local_training_epochs_server=4 --lr_p2p=0.08435756082060261 --lr_server=0.03389068878750167 --optimizer=adam --seed 4
