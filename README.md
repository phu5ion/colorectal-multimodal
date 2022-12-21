# colorectal-multimodal

This project showcases the methods used in our paper: Predictive models for colorectal cancer recurrence using multi-modal healthcare data, available at [this link](https://dl.acm.org/doi/10.1145/3450439.3451868). We are in the process of updating this repo to include a synthetic dataset with similar features, as well as a Jupyter Notebook documenting how to run our models. We will be releasing the following types of models:

1. Baseline standard ML models that utilise pre-extracted time-series features combined with tabular data
2. Deep-learning hybrid models that receives and processes dual input (time-series and tabular data), integrates learnt features to output a decision  
- Our own modified transformer (details in our paper)
- LSTM
- Temporal convolutional network ([Wavenet](https://arxiv.org/abs/1609.03499))
3. Denoising autoencoder that learns a latent representation of each data modality in an unsupervised manner. The representations can be combined and fed through a standard ML classifier such as SVM to learn the classification task.
