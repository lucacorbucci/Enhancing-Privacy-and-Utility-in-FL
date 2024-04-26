

# fl_public_02_epsilon_02
cd ./epsilon_0.2_paper
poetry run python /home/l.corbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam
poetry run python /home/l.corbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam
poetry run python /home/l.corbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam
poetry run python /home/l.corbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config private_02.json --batch_size_server=190 --clipping=1 --local_training_epochs_server=2 --lr_server=0.045243085434709354 --optimizer=adam

# p2p_public_02_epsilon_02
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam
poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.2_paper/run_paper.py --config p2p+server_0.2.json --batch_size_p2p=188 --batch_size_server=83 --clipping=3 --fl_rounds_P2P=8 --local_training_epochs_p2p=3 --local_training_epochs_server=3 --lr_p2p=0.08947377511996717 --lr_server=0.016396381866457185 --optimizer=adam

# baseline_public_02_epsilon_02

poetry run python /home/lcorbucci/pistacchio-fl-simulator/examples/dutch/epsilon_0.5/run_paper.py --config baseline_02.json --batch_size_server=66 --local_training_epochs_server=4 --lr_server=0.08925155574926462 --optimizer=adam