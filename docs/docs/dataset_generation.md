---
sidebar_position: 3
---

# Dataset Generation

We implemented a tool to generate datasets with a specific data distribution. The tool is able to generate datasets with the following data distributions:

-   [Classic Stratified Sampling](#classic-stratified-sampling)
-   [Stratified Sampling with maximum number of samples per node](#stratified-sampling-with-maximum-number-of-samples-per-node)
-   [Stratified Sampling with maximum number of samples per nodes and underrepresented classes](#stratified-sampling-with-maximum-number-of-samples-per-nodes-and-underrepresented-classes)
-   [Stratified Sampling with different amount of samples per nodes and underrepresented classes only in some nodes](#stratified-sampling-with-different-amount-of-samples-per-nodes-and-underrepresented-classes-only-in-some-nodes)
-   [Classic Percentage split](#classic-percentage-split)
-   [Percentage split with maximum number of samples per nodes](#percentage-split-with-maximum-number-of-samples-per-nodes)
-   [Percentage split with maximum number of samples per nodes and underrepresented classes](#percentage-split-with-maximum-number-of-samples-per-nodes-and-underrepresented-classes)

## Stratified Sampling

### Classic Stratified Sampling

This is the easiest data distribution that we can generate. Given the original dataset we split it in N parts where N is the number of nodes that we want to have (N=8 in the following example). Each node will have the same distribution of the original dataset.

The following example shows how to generate a dataset with 8 nodes, and 10 classes.

```json
"data_split_config": {
        "split_type": "stratified",
        "num_nodes": 1,
        "num_clusters": 8,
}
```

This is the output of the script:

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/stratified.json

10 7500 [(0, 725), (1, 832), (2, 753), (3, 789), (4, 714), (5, 680), (6, 745), (7, 768), (8, 757), (9, 737)]
10 7500 [(0, 769), (1, 805), (2, 731), (3, 731), (4, 787), (5, 687), (6, 779), (7, 749), (8, 731), (9, 731)]
10 7500 [(0, 736), (1, 863), (2, 711), (3, 764), (4, 728), (5, 732), (6, 711), (7, 785), (8, 716), (9, 754)]
10 7500 [(0, 737), (1, 845), (2, 792), (3, 744), (4, 758), (5, 663), (6, 725), (7, 761), (8, 695), (9, 780)]
10 7500 [(0, 778), (1, 864), (2, 709), (3, 751), (4, 725), (5, 675), (6, 756), (7, 799), (8, 729), (9, 714)]
10 7500 [(0, 702), (1, 844), (2, 775), (3, 805), (4, 683), (5, 664), (6, 730), (7, 813), (8, 725), (9, 759)]
10 7500 [(0, 732), (1, 835), (2, 755), (3, 777), (4, 712), (5, 675), (6, 730), (7, 782), (8, 767), (9, 735)]
10 7500 [(0, 744), (1, 854), (2, 732), (3, 770), (4, 735), (5, 645), (6, 742), (7, 808), (8, 731), (9, 739)]
```

Note that in this example we consider 8 as the number of clusters and 1 as the number of the nodes because we wanted to use only 8 nodes. We can also choose a different parameter for the number of nodes so that we can have a first split among the clusters and a second split among the nodes in each cluster.

```json
"data_split_config": {
        "split_type": "stratified",
        "num_nodes": 3,
        "num_clusters": 8,
}
```

This is the output of the script:

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/stratified_cluster_and_nodes.json

10 2500 [(0, 257), (1, 254), (2, 248), (3, 245), (4, 244), (5, 238), (6, 264), (7, 264), (8, 256), (9, 230)]
10 2500 [(0, 242), (1, 300), (2, 269), (3, 276), (4, 220), (5, 237), (6, 205), (7, 246), (8, 275), (9, 230)]
10 2500 [(0, 226), (1, 278), (2, 236), (3, 268), (4, 250), (5, 205), (6, 276), (7, 258), (8, 226), (9, 277)]
10 2500 [(0, 268), (1, 242), (2, 263), (3, 253), (4, 268), (5, 234), (6, 250), (7, 243), (8, 248), (9, 231)]
10 2500 [(0, 237), (1, 279), (2, 231), (3, 249), (4, 245), (5, 228), (6, 273), (7, 248), (8, 249), (9, 261)]
10 2500 [(0, 264), (1, 284), (2, 237), (3, 229), (4, 274), (5, 225), (6, 256), (7, 258), (8, 234), (9, 239)]
10 2500 [(0, 254), (1, 300), (2, 227), (3, 255), (4, 227), (5, 242), (6, 242), (7, 264), (8, 230), (9, 259)]
10 2500 [(0, 270), (1, 292), (2, 232), (3, 245), (4, 229), (5, 253), (6, 229), (7, 241), (8, 249), (9, 260)]
10 2500 [(0, 212), (1, 271), (2, 252), (3, 264), (4, 272), (5, 237), (6, 240), (7, 280), (8, 237), (9, 235)]
10 2500 [(0, 251), (1, 306), (2, 264), (3, 247), (4, 256), (5, 209), (6, 235), (7, 253), (8, 225), (9, 254)]
10 2500 [(0, 251), (1, 286), (2, 268), (3, 238), (4, 252), (5, 213), (6, 247), (7, 267), (8, 240), (9, 238)]
10 2500 [(0, 235), (1, 253), (2, 260), (3, 259), (4, 250), (5, 241), (6, 243), (7, 241), (8, 230), (9, 288)]
10 2500 [(0, 246), (1, 289), (2, 239), (3, 266), (4, 253), (5, 238), (6, 240), (7, 272), (8, 220), (9, 237)]
10 2500 [(0, 252), (1, 276), (2, 236), (3, 240), (4, 253), (5, 220), (6, 274), (7, 251), (8, 256), (9, 242)]
10 2500 [(0, 280), (1, 299), (2, 234), (3, 245), (4, 219), (5, 217), (6, 242), (7, 276), (8, 253), (9, 235)]
10 2500 [(0, 238), (1, 288), (2, 279), (3, 250), (4, 235), (5, 214), (6, 250), (7, 247), (8, 234), (9, 265)]
10 2500 [(0, 234), (1, 267), (2, 258), (3, 285), (4, 227), (5, 232), (6, 236), (7, 293), (8, 233), (9, 235)]
10 2500 [(0, 230), (1, 289), (2, 238), (3, 270), (4, 221), (5, 218), (6, 244), (7, 273), (8, 258), (9, 259)]
10 2500 [(0, 245), (1, 281), (2, 244), (3, 254), (4, 226), (5, 228), (6, 246), (7, 262), (8, 262), (9, 252)]
10 2500 [(0, 231), (1, 279), (2, 243), (3, 255), (4, 242), (5, 217), (6, 241), (7, 267), (8, 267), (9, 258)]
10 2500 [(0, 256), (1, 275), (2, 268), (3, 268), (4, 244), (5, 230), (6, 243), (7, 253), (8, 238), (9, 225)]
10 2500 [(0, 262), (1, 302), (2, 251), (3, 239), (4, 239), (5, 224), (6, 246), (7, 253), (8, 227), (9, 257)]
10 2500 [(0, 234), (1, 276), (2, 231), (3, 243), (4, 277), (5, 219), (6, 264), (7, 277), (8, 247), (9, 232)]
10 2500 [(0, 248), (1, 276), (2, 250), (3, 288), (4, 219), (5, 202), (6, 232), (7, 278), (8, 257), (9, 250)]
```

### Stratified Sampling with maximum number of samples per node

Setting the following parameters in the configuration file, we are able to generate a dataset where we specify the amount of samples that we want to assign to each node. The distribution of each node will be stratified and it will be the same as the original dataset.

```json
"data_split_config": {
        "split_type": "stratified",
        "max_samples_per_cluster": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
}
```

The following example shows how to generate a dataset with 8 nodes, and 10 classes. The number of samples per class changes from 1000 to 8000, with a step of 1000.

```bash
>>> python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/stratified_custom_max_samples.json

10 1000 [(0, 107), (1, 120), (2, 88), (3, 97), (4, 96), (5, 102), (6, 106), (7, 89), (8, 103), (9, 92)]
10 2000 [(0, 208), (1, 200), (2, 209), (3, 193), (4, 188), (5, 189), (6, 197), (7, 220), (8, 221), (9, 175)]
10 3000 [(0, 280), (1, 323), (2, 304), (3, 335), (4, 281), (5, 262), (6, 282), (7, 318), (8, 308), (9, 307)]
10 4000 [(0, 398), (1, 431), (2, 415), (3, 417), (4, 417), (5, 361), (6, 410), (7, 384), (8, 373), (9, 394)]
10 5000 [(0, 501), (1, 563), (2, 468), (3, 478), (4, 519), (5, 453), (6, 529), (7, 506), (8, 483), (9, 500)]
10 6000 [(0, 606), (1, 707), (2, 571), (3, 607), (4, 559), (5, 581), (6, 567), (7, 613), (8, 563), (9, 626)]
10 7000 [(0, 681), (1, 795), (2, 719), (3, 690), (4, 729), (5, 628), (6, 682), (7, 739), (8, 662), (9, 675)]
10 8000 [(0, 795), (1, 886), (2, 782), (3, 820), (4, 781), (5, 726), (6, 813), (7, 830), (8, 761), (9, 806)]
```

Alternatively, you can specify only an integer and use it as the maximum number of samples per node. In this case, the number of samples per class will be the same for all the nodes.

```json
"data_split_config": {
        "split_type": "stratified",
        "max_samples_per_cluster": 1000,
        "num_nodes": 1,
        "num_clusters": 8,
}
```

This is the output of the script:

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/stratified_max_sample.json

10 1000 [(0, 107), (1, 120), (2, 88), (3, 97), (4, 96), (5, 102), (6, 106), (7, 89), (8, 103), (9, 92)]
10 1000 [(0, 94), (1, 93), (2, 99), (3, 105), (4, 101), (5, 93), (6, 106), (7, 114), (8, 104), (9, 91)]
10 1000 [(0, 114), (1, 107), (2, 110), (3, 88), (4, 87), (5, 96), (6, 91), (7, 106), (8, 117), (9, 84)]
10 1000 [(0, 95), (1, 109), (2, 98), (3, 112), (4, 90), (5, 81), (6, 91), (7, 104), (8, 109), (9, 111)]
10 1000 [(0, 89), (1, 125), (2, 122), (3, 119), (4, 90), (5, 103), (6, 75), (7, 97), (8, 98), (9, 82)]
10 1000 [(0, 96), (1, 89), (2, 84), (3, 104), (4, 101), (5, 78), (6, 116), (7, 117), (8, 101), (9, 114)]
10 1000 [(0, 93), (1, 125), (2, 119), (3, 105), (4, 87), (5, 85), (6, 102), (7, 89), (8, 79), (9, 116)]
10 1000 [(0, 90), (1, 123), (2, 89), (3, 97), (4, 113), (5, 92), (6, 110), (7, 101), (8, 98), (9, 87)]
```

### Stratified Sampling with maximum number of samples per nodes and underrepresented classes

An alternative to the previous data distribution is a data distribution where we want to reduce the amount of samples of one or more classes. At the same time we want a variable number of samples per node.

Setting the following parameters in the configuration file, we are able to generate a dataset where we specify the amount of samples that we want to assign to each node and we reduce the number of data for a specific class. The distribution of each node will be stratified and it will be the same as the original dataset. In this case we specify the number of samples of the class 8 that we want in the original dataset.

```json
"data_split_config": {
        "split_type": "stratified",
        "underrepresented_class": [8],
        "num_samples_underrepresented_classes": [2000],
        "max_samples_per_cluster": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
}
```

Alternatively, we can specify the percentage of samples of one or more class that we want to remove from the dataset:

```json
"data_split_config": {
        "split_type": "stratified",
        "underrepresented_class": [8],
        "percentage_underrepresented_class": [0.3],
        "max_samples_per_cluster": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
}
```

The following example shows how to generate a dataset with 8 nodes, and 10 classes. The number of samples per class changes from 1000 to 8000, with a step of 1000. The class 8 has only 2000 samples in the original dataset and so the amount of samples of this class in each node is reduced.

```bash
>>> python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/stratified_custom_max_samples_underrepresented_classes.json

10 1000 [(0, 112), (1, 125), (2, 110), (3, 120), (4, 83), (5, 99), (6, 85), (7, 125), (8, 41), (9, 100)]
10 2000 [(0, 227), (1, 251), (2, 199), (3, 239), (4, 191), (5, 176), (6, 215), (7, 210), (8, 80), (9, 212)]
10 3000 [(0, 310), (1, 372), (2, 301), (3, 336), (4, 281), (5, 307), (6, 306), (7, 364), (8, 112), (9, 311)]
10 4000 [(0, 439), (1, 455), (2, 410), (3, 435), (4, 434), (5, 378), (6, 434), (7, 448), (8, 121), (9, 446)]
10 5000 [(0, 487), (1, 602), (2, 544), (3, 531), (4, 496), (5, 511), (6, 559), (7, 542), (8, 179), (9, 549)]
10 6000 [(0, 636), (1, 735), (2, 629), (3, 622), (4, 637), (5, 598), (6, 638), (7, 689), (8, 193), (9, 623)]
10 7000 [(0, 717), (1, 836), (2, 726), (3, 755), (4, 747), (5, 663), (6, 735), (7, 780), (8, 255), (9, 786)]
10 8000 [(0, 875), (1, 964), (2, 828), (3, 874), (4, 832), (5, 766), (6, 800), (7, 905), (8, 299), (9, 857)]
```

### Stratified Sampling with different amount of samples per nodes and underrepresented classes only in some nodes

In this data distribution we specify:

-   The maximum amount of samples that each node will use during the training
-   The amount of samples of a specific class that we want to reduce in the dataset. In this example we will use only the 30% of class 8 removing the other 70% of the samples of this class.
-   The number of nodes where we want to have the underrepresented class

```json
"data_split_config": {
        "split_type": "stratified",
        "underrepresented_class": [8],
        "percentage_underrepresented_class": [0.7],
        "num_reduced_nodes": 3,
        "max_samples_per_cluster": 5000,
}
```

This is the output of the script:

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/stratified_custom_max_samples_underrepresented_classes_percentage_only_some_nodes.json

10 5000 [(0, 499), (1, 554), (2, 517), (3, 521), (4, 464), (5, 475), (6, 469), (7, 510), (8, 531), (9, 460)]
10 5000 [(0, 494), (1, 520), (2, 499), (3, 521), (4, 518), (5, 439), (6, 526), (7, 501), (8, 474), (9, 508)]
10 5000 [(0, 501), (1, 563), (2, 468), (3, 478), (4, 519), (5, 453), (6, 529), (7, 506), (8, 483), (9, 500)]
10 5000 [(0, 524), (1, 592), (2, 459), (3, 500), (4, 456), (5, 495), (6, 471), (7, 505), (8, 479), (9, 519)]
10 5000 [(0, 463), (1, 577), (2, 516), (3, 511), (4, 528), (5, 446), (6, 475), (7, 533), (8, 462), (9, 489)]
10 5000 [(0, 511), (1, 599), (2, 569), (3, 561), (4, 498), (5, 491), (6, 512), (7, 565), (8, 156), (9, 538)]
10 5000 [(0, 528), (1, 547), (2, 567), (3, 582), (4, 528), (5, 480), (6, 533), (7, 564), (8, 158), (9, 513)]
10 5000 [(0, 502), (1, 639), (2, 498), (3, 509), (4, 550), (5, 469), (6, 520), (7, 590), (8, 190), (9, 533)]
```

## Percentage

### Classic Percentage Split

This is the easiest way to generate a percentage split. In the configuration file we need to specify the number of nodes and, for each node, the percentage of samples that we want to use. Note that the percentages of each node don't sum to 100. The percentage of each class that we assign to each node must adds up to 100.

```json
"data_split_config": {
        "split_type": "percentage",
        "num_classes": 10,
        "num_nodes": 3,
        "num_clusters": 3,
        "noniid_nodes_distribution": false,
        "server_validation_set": "server_validation_split",
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
```

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/classic_percentage.json

7 7748 [(0, 1593), (1, 1744), (2, 1590), (3, 1673), (4, 406), (5, 351), (9, 391)]
7 7747 [(0, 1557), (1, 1793), (2, 1611), (3, 1632), (4, 375), (5, 385), (9, 394)]
7 7747 [(0, 1588), (1, 1856), (2, 1565), (3, 1599), (4, 387), (5, 348), (9, 404)]
6 5799 [(3, 402), (4, 1507), (5, 1426), (6, 1636), (7, 432), (8, 396)]
6 5798 [(3, 422), (4, 1562), (5, 1442), (6, 1555), (7, 430), (8, 387)]
6 5798 [(3, 403), (4, 1605), (5, 1469), (6, 1543), (7, 391), (8, 387)]
7 6455 [(0, 400), (1, 449), (2, 408), (6, 392), (7, 1639), (8, 1580), (9, 1587)]
7 6454 [(0, 395), (1, 480), (2, 373), (6, 397), (7, 1670), (8, 1578), (9, 1561)]
7 6454 [(0, 390), (1, 420), (2, 411), (6, 395), (7, 1703), (8, 1523), (9, 1612)]
```

### Percentage Split with maximum number of samples per nodes

For this data distribution we can decide to specify a single maximum amount of samples for all the nodes or we can specify a different maximum amount of samples for each node.

```json
"data_split_config": {
    "split_type": "percentage_max_samples",
        "max_samples_per_cluster": 5000,
        "num_clusters": 8,
    }
```

```json
    "data_split_config": {
        "split_type": "percentage_max_samples",
        "max_samples_per_cluster": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "num_clusters": 8,
    }
```

The following example shows how to generate a dataset with 8 nodes, and 10 classes. The number of samples per class changes from 1000 to 8000, with a step of 1000. The maximum number of samples per node is 5000.

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/percentage_custom_max_samples.json
10 1000 [(0, 100), (1, 100), (2, 100), (3, 100), (4, 100), (5, 100), (6, 100), (7, 100), (8, 100), (9, 100)]
10 2000 [(0, 200), (1, 200), (2, 200), (3, 200), (4, 200), (5, 200), (6, 200), (7, 200), (8, 200), (9, 200)]
10 3000 [(0, 300), (1, 300), (2, 300), (3, 300), (4, 300), (5, 300), (6, 300), (7, 300), (8, 300), (9, 300)]
10 4000 [(0, 400), (1, 400), (2, 400), (3, 400), (4, 400), (5, 400), (6, 400), (7, 400), (8, 400), (9, 400)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
10 6000 [(0, 600), (1, 600), (2, 600), (3, 600), (4, 600), (5, 600), (6, 600), (7, 600), (8, 600), (9, 600)]
10 7000 [(0, 700), (1, 700), (2, 700), (3, 700), (4, 700), (5, 700), (6, 700), (7, 700), (8, 700), (9, 700)]
10 8000 [(0, 800), (1, 800), (2, 800), (3, 800), (4, 800), (5, 800), (6, 800), (7, 800), (8, 800), (9, 800)]
```

### Percentage split with maximum number of samples per nodes and underrepresented classes

This data distribution allows us to specify the amount of samples that each node will have. For each node we can also specify the percentage of samples of each class that we want to assign to that node.

In this case we have 8 clusters (nodes) and we decided that we wanted to have cluster_0 with only a few samples of the class 8 and cluster_1 with a lot of samples of the class 8. We also decided that we wanted to have a maximum of 5000 samples per node.

````json

```json
"split_type": "percentage_max_samples",
        "max_samples_per_cluster": 5000,
        "num_nodes": 1,
        "num_clusters": 8,
        "percentage_configuration": {
            "cluster_0": {
                "0": 11,
                "1": 11,
                "2": 11,
                "3": 11,
                "4": 11,
                "5": 11,
                "6": 11,
                "7": 10,
                "8": 3,
                "9": 10
            },
            "cluster_1": {
                "0": 9,
                "1": 9,
                "2": 9,
                "3": 9,
                "4": 9,
                "5": 9,
                "6": 9,
                "7": 9,
                "8": 20,
                "9": 8
            },
        }
````

This is the output of the script.

```bash
>>> pyenv exec python DataSplit/generate_dataset.py --config Examples/DatasetGeneration/config_file/percentage_max_samples_underrepresented_class.json

10 5000 [(0, 550), (1, 550), (2, 550), (3, 550), (4, 550), (5, 550), (6, 550), (7, 500), (8, 150), (9, 500)]
10 5000 [(0, 450), (1, 450), (2, 450), (3, 450), (4, 450), (5, 450), (6, 450), (7, 450), (8, 1000), (9, 400)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
10 5000 [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
```
