PROJECT_NAME="p2p_covertype" # swap out globally

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

# list of sweeps to call

# rm -rf ../data/covertype/federated_split
# poetry run python ./generate_dataset.py --config private_02.json

# run_sweep_and_agent "config_privacy_02"
# run_sweep_and_agent "config_p2p_02"


# rm -rf ../data/covertype/federated_split
# poetry run python ./generate_dataset.py --config private_03.json
# run_sweep_and_agent "config_privacy_03"
# run_sweep_and_agent "config_p2p_03"

rm -rf ../data/covertype/federated_split
poetry run python ./generate_dataset.py --config private_04.json
# run_sweep_and_agent "config_privacy_04"
run_sweep_and_agent "config_p2p_04"