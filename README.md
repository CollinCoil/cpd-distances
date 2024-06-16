# About

This repository contains code that corresponds with the paper "Assessing Distance Metrics for Change Point Detection in Continual Learning Scenarios." 

Use the following link to access the paper: [https://link.springer.com/chapter/10.1007/978-3-031-62700-2_23](https://link.springer.com/chapter/10.1007/978-3-031-62700-2_23)

Our approach generalized WATCH, which is accessible at https://github.com/lifelonglab/watch and was used in trials using the Wasserstein distance metric. 

Some datasets used in the paper can be accessed at https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios/tree/main 

The code used to run the change point detection experiments is in the `cpd_distance_trials.py` file. Experimental results beyond those presented the paper can be found the files `experimental_results.csv` and `CPD_Full_Tables.pdf`. The former contains information on the hyperparameters and results, and the latter contains cleaned tables of cover and F1 scores associated with the distance measure in each trial. 

# Citation
A recommended citation is 

@InProceedings{10.1007/978-3-031-62700-2_23,

  author="Coil, Collin and Corizzo, Roberto",
  
  editor="Appice, Annalisa and Azzag, Hanane and Hacid, Mohand-Said and Hadjali, Allel and Ras, Zbigniew",
  
  title="Assessing Distance Measures for Change Point Detection in Continual Learning Scenarios",
  
  booktitle="Foundations of Intelligent Systems",
  
  year="2024",
  
  publisher="Springer Nature Switzerland",
  
  address="Cham",
  
  pages="260--270",
  
  isbn="978-3-031-62700-2"
  
}
