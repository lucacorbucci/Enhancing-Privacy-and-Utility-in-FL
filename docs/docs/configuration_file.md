---
sidebar_position: 2
---

# Configuration File

The configuration file is a JSON file that contains all the parameters needed to run a test. The configuration file is divided into 5 main parts:

-   General configuration
-   Configuration of the data split
-   Configuration of the P2P training
-   Configuration of the server training
-   Hyperparameters

## An Example

```json
{
	"dataset": "mnist",
	"mode": "semi_p2p",
	"debug": true,
	"save_model": false,
	"wandb": true,
	"data_split_config": {
		"split_type": "percentage",
		"num_nodes": 3,
		"num_clusters": 3,
		"noniid_nodes_distribution": false,
		"server_test_set": "server_validation_split",
		"percentage_configuration": {
			"cluster_0": {
				"0": 80,
				"1": 80,
				"2": 80,
				"3": 80,
				"4": 20,
				"5": 20,
				"9": 20
			},
			"cluster_1": {
				"4": 80,
				"5": 80,
				"6": 80,
				"7": 20,
				"8": 20,
				"3": 20
			},
			"cluster_2": {
				"7": 80,
				"8": 80,
				"9": 80,
				"6": 20,
				"0": 20,
				"1": 20,
				"2": 20
			}
		}
	},
	"p2p_config": {
		"differential_privacy_pre_training_cluster": false,
		"differential_privacy_mixed_mode_cluster": false,
		"local_training_epochs_in_cluster": 1,
		"num_communication_round_pre_training": [0],
		"num_communication_round_mixed_mode": 2
	},
	"server_config": {
		"differential_privacy_server": false,
		"mixed_mode": false,
		"local_training_epochs_with_server": 1,
		"num_communication_round_with_server": 40,
		"total_mixed_iterations": 1
	},
	"hyperparameters": {
		"batch_size": 32,
		"lr": 0.001,
		"MAX_PHYSICAL_BATCH_SIZE": 128,
		"DELTA": 1e-5,
		"noise_multiplier": [2.0],
		"noise_multiplier_P2P": [2.0],
		"max_grad_norm": [1.2],
		"weight_decay": 0,
		"min_improvement": 0.001,
		"patience": 10,
		"min_accuracy": 0.8
	}
}
```

Let's see the parameters one by one.

-   **Dataset:** the name of the dataset to use. We need to specify one of the dataset supported by the library. If you want to use a personalized dataset you need to add it. You can find more information in the [dataset section](/custom_dataset).
-   **Mode:** the mode we want to use during this test. The possible values are "semi_p2p", "classic" and "p2p". If you want to know more about the different modes you can read the [modes section](/Tasks).
-   **debug:** if true, some additional print statements will be printed during the execution of the test.
-   **wandb:** if true, the results of the test will be saved on [wandb](https://wandb.ai/).
-   **save_model:** if true, the model will be saved on wandb

With the **data_split_config** parameter we can specify all the parameters related to the data split:

-   **split_type:** the type of data split we want to use. The possible values are "percentage" and "stratified".
-   **num_nodes:** The number of nodes we want to use in the test.
-   **num_clusters:** The number of clusters we want to use in the test.
-   **noniid_nodes_distribution:** if true, the data split will be non-iid for each node. Otherwise it will be iid for each node inside each cluster.
-   **server_test_set:** the dataset we want to use as validation set on the server. The default value is "server_validation_split". We assume that the server has some data to make a test at each iteration.
-   **percentage_configuration:** A dictionary with the percentage of data that we want to assign to each cluster. The keys of the dictionary are the cluster names. The values are dictionaries with the percentage of data for each class. The keys of the inner dictionary are the class names and the values are the percentage of data for that class.

With the **p2p_config** parameter we can specify all the parameters related to the P2P training:

-   **differential_privacy_pre_training_cluster:** if true, the differential privacy will be applied during the pre-training phase inside each cluster.
-   **differential_privacy_mixed_mode_cluster:** if true, the differential privacy will be applied by each node of the cluster during the mixed mode phase.
-   **local_training_epochs_in_cluster:** the number of local training epochs that we want to perform on each client before sharing the model with the other nodes of the cluster.
-   **num_communication_round_pre_training:** the number of communication rounds (exchanges of model weights) that we want to perform during the pre-training phase inside the cluster.
-   **num_communication_round_mixed_mode:** the number of communication rounds (exchanges of model weights) that we want to perform during the mixed mode phase inside the cluster before switching to the training phase with the server.

With the **server_config** parameter we can specify all the parameters related to the server training:

-   **differential_privacy_server:** if true, the differential privacy will be applied during the training phase with the server.
-   **mixed_mode:** if true, the mixed mode will be used during the training phase with the server. This means that we will alternate between training inside the cluster and training with the server.
-   **local_training_epochs_with_server:** the number of local training epochs that we want to perform on each client before sharing the model with the server.
-   **num_communication_round_with_server:** the number of communication rounds (exchanges of model weights) that we want to perform during the training phase with the server.
-   **total_mixed_iterations:** the number of times we want to perform the mixed mode phase with the server. For instance if we set this to 10, we will alternate between training inside the cluster and training with the server for 10 times.

With the **hyperparameters** parameter we can specify all the parameters related to the hyperparameters:

-   **batch_size:** the batch size that we want to use during the training.
-   **lr:** the learning rate that we want to use during the training.
-   **MAX_PHYSICAL_BATCH_SIZE:** the maximum physical batch size that we want to use during the training. This is related with Opacus, the library that we use to inject differential privacy in the training.
-   **DELTA:** the delta value that we want to use during the training. This is related with Opacus, the library that we use to inject differential privacy in the training.
-   **noise_multiplier:** the noise multiplier that we want to use during the training with the server. This represents the amount of noise that we want to inject in the training. This is related with Opacus, the library that we use to inject differential privacy in the training.
-   **noise_multiplier_P2P:** the noise multiplier that we want to use during the training inside the cluster. This represents the amount of noise that we want to inject in the training. This is related with Opacus, the library that we use to inject differential privacy in the training.
-   **max_grad_norm:** the maximum gradient norm that we want to use during the training. This is related with Opacus, the library that we use to inject differential privacy in the training.
-   **weight_decay:** the weight decay that we want to use during the training.
-   **min_improvement:** the minimum improvement that we want to see in the validation accuracy before stopping the training.
-   **patience:** the number of epochs that we want to wait before stopping the training if the validation accuracy does not improve.
-   **min_accuracy:** the minimum accuracy that we want to reach before stopping the training.

## Execute a test
