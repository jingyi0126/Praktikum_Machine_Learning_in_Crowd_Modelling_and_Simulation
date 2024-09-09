This final project of MLCMS is a reproduction of the thesis Prediction of Pedestrian Speed with Artificial Neural Networks (https://arxiv.org/abs/1801.09782). The dataset is downloaded at https://zenodo.org/records/1054017.

/data stores the raw data.
/outputs stores the results of algorithms and weights of models.
/processed_data stores the preprocessed data, which is too large to upload. If the folder doesn't exist, create an empty folder with the name 'processed_data'.
/src stores the source codes of the project.
	/src/ANN.ipynb is the implementation of ANN model.
	/src/comparison.ipynb shows the comparison between ANN and Weidmann model.
	/src/data_processing.ipynb deals with preprocessing. The output path is ../processed_data.
	/src/model.py stores a python class for ANN model.
	/src/utils.py stores the utilized functions in this project.
	/src/Weidmann.ipynb is the implementation of Weidmann model.

Authors: Kejia Gao, Jingyi Zhang, Maximilian Mayr, Yizhi Liu, Felipe Antonio Diaz Laverde