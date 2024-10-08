cd ./epsilon_0.2

# # fl_public_02_epsilon_02
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam --seed 0
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam --seed 2
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam --seed 3
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam --seed 4

# # p2p_public_02_epsilon_02
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam --seed 0
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam --seed 2
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam --seed 3
# poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam --seed 4

# fl_public_03_epsilon_02
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_03.json --batch_size_server=83 --clipping=1 --local_training_epochs_server=2 --lr_server=0.05129179056623046 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_03.json --batch_size_server=83 --clipping=1 --local_training_epochs_server=2 --lr_server=0.05129179056623046 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_03.json --batch_size_server=83 --clipping=1 --local_training_epochs_server=2 --lr_server=0.05129179056623046 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_03.json --batch_size_server=83 --clipping=1 --local_training_epochs_server=2 --lr_server=0.05129179056623046 --optimizer=adam --seed 4

# p2p_public_03_epsilon_02
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.3.json --batch_size_p2p=38 --batch_size_server=84 --clipping=2 --fl_rounds_P2P=6 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.042856431564180446 --lr_server=0.011228718579719577 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.3.json --batch_size_p2p=38 --batch_size_server=84 --clipping=2 --fl_rounds_P2P=6 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.042856431564180446 --lr_server=0.011228718579719577 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.3.json --batch_size_p2p=38 --batch_size_server=84 --clipping=2 --fl_rounds_P2P=6 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.042856431564180446 --lr_server=0.011228718579719577 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.3.json --batch_size_p2p=38 --batch_size_server=84 --clipping=2 --fl_rounds_P2P=6 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.042856431564180446 --lr_server=0.011228718579719577 --optimizer=adam --seed 4

# fl_public_04_epsilon_02
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_04.json --batch_size_server=79 --clipping=2 --local_training_epochs_server=4 --lr_server=0.04104652324247766 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_04.json --batch_size_server=79 --clipping=2 --local_training_epochs_server=4 --lr_server=0.04104652324247766 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_04.json --batch_size_server=79 --clipping=2 --local_training_epochs_server=4 --lr_server=0.04104652324247766 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config private_04.json --batch_size_server=79 --clipping=2 --local_training_epochs_server=4 --lr_server=0.04104652324247766 --optimizer=adam --seed 4

# p2p_public_04_epsilon_02
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.4.json --batch_size_p2p=36 --batch_size_server=70 --clipping=4 --fl_rounds_P2P=8 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06660522184202793 --lr_server=0.011562607786561731 --optimizer=adam --seed 0
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.4.json --batch_size_p2p=36 --batch_size_server=70 --clipping=4 --fl_rounds_P2P=8 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06660522184202793 --lr_server=0.011562607786561731 --optimizer=adam --seed 2
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.4.json --batch_size_p2p=36 --batch_size_server=70 --clipping=4 --fl_rounds_P2P=8 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06660522184202793 --lr_server=0.011562607786561731 --optimizer=adam --seed 3
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2/run_test.py --config p2p+server_0.4.json --batch_size_p2p=36 --batch_size_server=70 --clipping=4 --fl_rounds_P2P=8 --local_training_epochs_p2p=2 --local_training_epochs_server=1 --lr_p2p=0.06660522184202793 --lr_server=0.011562607786561731 --optimizer=adam --seed 4

