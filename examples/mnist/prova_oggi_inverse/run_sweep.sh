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
  poetry run wandb agent $SWEEP_ID --count 30 --project "$PROJECT_NAME"
}


# # list of sweeps to call
# 
# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config only_p2p_0.2.json

# run_sweep_and_agent "config_only_p2p_02"

# run_sweep_and_agent "config_only_p2p_02_private"

# run_sweep_and_agent "config_server_p2p_0.2"

# # list of sweeps to call
# 
# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config only_p2p_0.2.json

# run_sweep_and_agent "config_only_p2p_03"

# run_sweep_and_agent "config_only_p2p_03_private"

# run_sweep_and_agent "config_server_p2p_0.3"



# list of sweeps to call

rm -rf ../data/mnist/federated_split
poetry run python ./generate_dataset.py --config only_p2p_0.4.json

run_sweep_and_agent "config_only_p2p_04"

run_sweep_and_agent "config_only_p2p_04_private"

run_sweep_and_agent "config_server_p2p_0.4"



# # list of sweeps to call
# 
# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config only_p2p_0.6.json

# # run_sweep_and_agent "config_only_p2p_06"

# # run_sweep_and_agent "config_only_p2p_06_private"

# run_sweep_and_agent "config_server_p2p_0.6"



# # list of sweeps to call
# 
# rm -rf ../data/mnist/federated_split
# poetry run python ./generate_dataset.py --config only_p2p_0.8.json

# # run_sweep_and_agent "config_only_p2p_08"

# # run_sweep_and_agent "config_only_p2p_08_private"

# run_sweep_and_agent "config_server_p2p_0.8"




# # # list of sweeps to call
# # 
# # rm -rf ../data/mnist/federated_split
# # poetry run python ./generate_dataset.py --config only_p2p_full.json

# # run_sweep_and_agent "config_only_p2p_full"

# # run_sweep_and_agent "config_only_p2p_private_full"