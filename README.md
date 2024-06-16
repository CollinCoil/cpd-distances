# About

This repository contains code that corresponds with the paper "Assessing Distance Metrics for Change Point Detection in Continual Learning Scenarios." 

Use the following link to access the paper: [https://link.springer.com/chapter/10.1007/978-3-031-62700-2_23](https://link.springer.com/chapter/10.1007/978-3-031-62700-2_23)
Our approach generalized WATCH, which is accessible at https://github.com/lifelonglab/watch and was used in trials using the Wasserstein distance metric. 

Some datasets used in the paper can be accessed at https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios/tree/main 

The code used to run the change point detection experiments is in the `cpd_distance_trials.py` file. Experimental results beyond those presented the paper can be found the files `experimental_results.csv` and `CPD_Full_Tables.pdf`. The former contains information on the hyperparameters and results, and the latter contains cleaned tables of cover and F1 scores associated with the distance metrics in each trial. 

# Citation
A recommended citation is 

@InProceedings{10.1007/978-3-031-62700-2_23,
author="Coil, Collin
and Corizzo, Roberto",
editor="Appice, Annalisa
and Azzag, Hanane
and Hacid, Mohand-Said
and Hadjali, Allel
and Ras, Zbigniew",
title="Assessing Distance Measures for Change Point Detection in Continual Learning Scenarios",
booktitle="Foundations of Intelligent Systems",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="260--270",
abstract="Detecting relevant change points in time-series data is a necessary task in various applications. Change point detection methods are effective techniques for discovering abrupt changes in data streams. Although prior work has explored the effectiveness of different algorithms on real-world data, little has been done to explore the impact of different distance measures on change detection performance. In this paper, we modify the architecture of a change point detection workflow to assess the impact of distance measure choices on change detection accuracy and efficiency in continual learning scenarios, where the goal is detecting transitions between tasks or concepts. An experimental evaluation of 41 distance measure across several benchmark datasets demonstrated that the change detection accuracy depends on the distance measure selected. Furthermore, our analysis showed performance patterns for distance measures in the same family.",
isbn="978-3-031-62700-2"
}
