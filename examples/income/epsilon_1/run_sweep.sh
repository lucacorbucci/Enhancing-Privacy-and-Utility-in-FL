PROJECT_NAME="income_p2p" # swap out globally

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
  poetry run wandb agent $SWEEP_ID --count 10 --project "$PROJECT_NAME"
}


poetry run python ./generate_dataset.py --config baseline_02.json
poetry run python ./generate_dataset.py --config baseline_03.json
poetry run python ./generate_dataset.py --config baseline_04.json


# # list of sweeps to call

# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config baseline_02.json

# run_sweep_and_agent "config_baseline_02"
# rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config private_02.json

# # run_sweep_and_agent "config_privacy_02"
# # rm -rf ./wandb/


# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config p2p+server_0.2.json

# # run_sweep_and_agent "config_p2p_02"
# # rm -rf ./wandb/



# # list of sweeps to call

# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config baseline_03.json

# run_sweep_and_agent "config_baseline_03"
# rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config private_03.json

# # run_sweep_and_agent "config_privacy_03"
# # rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config p2p+server_0.3.json

# # run_sweep_and_agent "config_p2p_03"
# # rm -rf ./wandb/



# # list of sweeps to call

# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config baseline_04.json

# run_sweep_and_agent "config_baseline_04"
# rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config private_04.json

# # run_sweep_and_agent "config_privacy_04"
# # rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config p2p+server_0.4.json

# # run_sweep_and_agent "config_p2p_04"

# # rm -rf ./wandb/


# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config baseline_05.json

# # run_sweep_and_agent "config_baseline_05"
# # rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config private_05.json

# # run_sweep_and_agent "config_privacy_05"
# # rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config p2p+server_0.5.json

# # run_sweep_and_agent "config_p2p_05"

# # rm -rf ./wandb/


# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config baseline_06.json

# # run_sweep_and_agent "config_baseline_06"
# # rm -rf ./wandb/


# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config private_06.json

# # run_sweep_and_agent "config_privacy_06"
# # rm -rf ./wandb/

# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config p2p+server_0.6.json

# # run_sweep_and_agent "config_p2p_06"

# # rm -rf ./wandb/















