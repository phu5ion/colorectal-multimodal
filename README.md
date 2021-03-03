# colorectal-multimodal

This project showcases the methods used in our paper: Predictive models for colorectal cancer recurrence using multi-modal healthcare data, available at (link). We are in the process of updating this repo to include a synthetic dataset with similar features, as well as a Jupyter Notebook documenting our code and methodology. We will be releasing the following types of models:

1. Baseline standard ML models that utilise pre-extracted time-series features combined with tabular data
2. Deep-learning hybrid models that receives and processes dual input (time-series and tabular data), integrates learnt features to output a decision
3. Denoising autoencoder that learns a latent representation of each data modality in an unsupervised manner. The representations can be combined and fed through a standard ML classifier such as SVM to learn the classification task.
