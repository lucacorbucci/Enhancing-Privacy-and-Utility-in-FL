PROJECT_NAME="mnist_Sweeps" # swap out globally

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"
  
  # Run the wandb sweep command and store the output in a temporary file
  poetry run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  # Remove the temporary output file
#   rm temp_output.txt
  
  # Run the wandb agent command
  poetry run wandb agent $SWEEP_ID --count 20 --project "$PROJECT_NAME"
}


# list of sweeps to call
ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config baseline_02.json


run_sweep_and_agent "config_privacy_02_epsilon_3"
run_sweep_and_agent "config_privacy_02_epsilon_5"
run_sweep_and_agent "config_privacy_02_epsilon_8"
run_sweep_and_agent "config_privacy_02_epsilon_10"

run_sweep_and_agent "config_p2p+server_02_epsilon_3"
run_sweep_and_agent "config_p2p+server_02_epsilon_5"
run_sweep_and_agent "config_p2p+server_02_epsilon_8"
run_sweep_and_agent "config_p2p+server_02_epsilon_10"

run_sweep_and_agent "config_baseline_02"


# list of sweeps to call
ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config private_03.json


run_sweep_and_agent "config_privacy_03_epsilon_3"
run_sweep_and_agent "config_privacy_03_epsilon_5"
run_sweep_and_agent "config_privacy_03_epsilon_8"
run_sweep_and_agent "config_privacy_03_epsilon_10"

run_sweep_and_agent "config_p2p+server_03_epsilon_3"
run_sweep_and_agent "config_p2p+server_03_epsilon_5"
run_sweep_and_agent "config_p2p+server_03_epsilon_8"
run_sweep_and_agent "config_p2p+server_03_epsilon_10"

run_sweep_and_agent "config_baseline_03"


# list of sweeps to call
ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config private_04.json


run_sweep_and_agent "config_privacy_04_epsilon_3"
run_sweep_and_agent "config_privacy_04_epsilon_5"
run_sweep_and_agent "config_privacy_04_epsilon_8"
run_sweep_and_agent "config_privacy_04_epsilon_10"

run_sweep_and_agent "config_p2p+server_04_epsilon_3"
run_sweep_and_agent "config_p2p+server_04_epsilon_5"
run_sweep_and_agent "config_p2p+server_04_epsilon_8"
run_sweep_and_agent "config_p2p+server_04_epsilon_10"

run_sweep_and_agent "config_baseline_04"


# list of sweeps to call
ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config private_05.json

run_sweep_and_agent "config_privacy_05_epsilon_3"
run_sweep_and_agent "config_privacy_05_epsilon_5"
run_sweep_and_agent "config_privacy_05_epsilon_8"
run_sweep_and_agent "config_privacy_05_epsilon_10"


run_sweep_and_agent "config_p2p+server_05_epsilon_3"
run_sweep_and_agent "config_p2p+server_05_epsilon_5"
run_sweep_and_agent "config_p2p+server_05_epsilon_8"
run_sweep_and_agent "config_p2p+server_05_epsilon_10"

run_sweep_and_agent "config_baseline_05"


# list of sweeps to call
ulimit -n 4096
rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config private_06.json



run_sweep_and_agent "config_privacy_06_epsilon_3"
run_sweep_and_agent "config_privacy_06_epsilon_5"
run_sweep_and_agent "config_privacy_06_epsilon_8"
run_sweep_and_agent "config_privacy_06_epsilon_10"

run_sweep_and_agent "config_p2p+server_06_epsilon_3"
run_sweep_and_agent "config_p2p+server_06_epsilon_5"
run_sweep_and_agent "config_p2p+server_06_epsilon_8"
run_sweep_and_agent "config_p2p+server_06_epsilon_10"

run_sweep_and_agent "config_baseline_06"

