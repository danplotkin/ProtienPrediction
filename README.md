# Amino Acid Sequence Function Prediction with TensorFlow and Prot5 Embeddings

## About 

Predicting protein function from a sequence of amino acids is very important for understanding biology, enabling drug discovery, advancing biotechnology, personalizing medicine, analyzing genomic data, and elucidating protein structure-function relationships. In this project, using the dataset from the [CAFA 5 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) Kaggle competition, I will use TensorFlow and ProtT5 embeddings to predict the biological functions of a protein given its amino acid sequence. Due to RAM constraints, we will be predicting 5000 function, rather than the full 31,466 in the dataset.

## Embedding Source

We will be using the pre-fit embeddings for this dataset from [Sergei Fironov](https://www.kaggle.com/datasets/sergeifironov/t5embeds/data), where the [prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) from Roastlab was fit on the amino acid sequences in the `Train/train_sequences.fasta` file. Using these emebddings saves us the effort of running this script, where large computational resources are required. The code for this process is [linked](https://www.kaggle.com/code/sergeifironov/t5embeds-calculation-only-few-samples).

## Store Colab Secrets

We will be using the CAFA 5 dataset and the T_5 embeds datasets from Kaggle for our model. In order to get this into Google Colab and replicate this code, we utilize Colab's secrets feature to store our Kaggle username and api key. This [discussion](https://www.kaggle.com/discussions/general/74235) shows you can store and retreive your secrets from Kaggle.

## Training

### GPU
This model was trained using a V100 GPU with premium RAM due to the large dataset used to train. However, a less powerfull GPU (or maybe even CPU) can be used for training.

### Metrics 
Since this is a multi-label classification problem, we use `tf.keras.losses.BinaryCrossentropy()` as our loss function, and `tf.keras.metrics.BinaryAccuracy()` and `tf.keras.metrics.AUC()` as our metrics. 

### Train-validation split
We grabbed a random 20% subset of our dataset shaped ((142246, 1024), (142246, 5000)) for validation.

### Results
<img src='https://github.com/danplotkin/ProtienPrediction/assets/116699460/9bdf470a-29a9-4998-90f0-65ee0a5e6b65' width='500' height='500'>

## Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danplotkin/ProtienPrediction/blob/main/ProtienPrediction.ipynb)

## Citation
@misc{cafa-5-protein-function-prediction,
    author = {Iddo Friedberg, Predrag Radivojac, Clara De Paolis, Damiano Piovesan, Parnal Joshi, Walter Reade, Addison Howard},
    title = {CAFA 5 Protein Function Prediction},
    publisher = {Kaggle},
    year = {2023},
    url = {https://kaggle.com/competitions/cafa-5-protein-function-prediction}
}
