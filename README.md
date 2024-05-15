# DeepSound: Genre Recognition through Deep Learning

## Students
- Thanh Dang
- Chi Nguyen
- Mohammad Fanous


## Abstract
Music genre classification is a crucial task in various music-related applications such as recommendation systems and search engines. This project aims to implement a deep learning model for accurately predicting the genre of a given piece of music. Deep learning methods, including LSTM, CNN, and Transformer architectures, will be explored for feature learning and classification.

## Steps
1. **Data Collection and Preprocessing**:
   - Collect the GTZAN dataset.
   - Preprocess audio files, including normalization, segmentation, and feature extraction using MFCCs and VGGish techniques.
2. **Data Splitting**:
   - Split the dataset into training, validation, and test sets.
3. **Model Architecture**:
   - Design or adapt LSTM, CNN, and/or Transformer architectures for music genre classification.
4. **Model Training**:
   - Train the model on the training set using an appropriate loss function and optimizer.
   - Start with two music genres and gradually increase the number of genres.
   - Shorten the length of analyzed music clips as needed.
5. **Hyperparameter Tuning**:
   - Tune hyperparameters such as learning rate, number of layers, and number of filters to improve performance on the validation set.
6. **Model Evaluation**:
   - Evaluate the performance of the model on the test set using metrics such as accuracy, precision, recall, and F1 score.
7. **Visualization**:
   - Visualize the model's output to demonstrate its ability to accurately classify music by genre.

## Milestone 1
For the first milestone of our project, we conduct data preprocessing with 2 feature extraction methods, MFCCs and VGGish in the corresponding Jupyter notebooks. The notebook contains training, validation and test sets as input (X) and output (Y) matrices, which can be later used for training deep neural networks.

## Milestone 2

Corresponding to each data preprocessing method, MFCCs and VGGish, we produce a corresponding model and obtain preliminary accuracies.

- MFCCs Model:
For the MFCCs feature extraction, we have developed a deep neural network model that includes several convolutional and pooling layers, followed by dense layers for classification. This model has achieved an accuracy of around 51% on the test set.

- VGGish Model:
For the VGGish feature extraction, we have adapted the VGGish architecture, which is a pre-trained model for audio event classification. By fine-tuning this model on our music genre dataset, we have obtained a preliminary accuracy of 81% on the test set.

These preliminary results demonstrate the effectiveness of deep learning approaches for music genre recognition. Further optimization of the model architectures and hyperparameters is ongoing to improve the performance of the models.

## Final Submission

We improved the VGGish model by changing the kernel size, which increased its accuracy to 85%. We also evaluated the model in more detail using a confusion matrix.

To test the model with data it wasn't trained on (Out-Of-Distribution data), we used 31 songs from a Kaggle dataset. The VGGish model got an accuracy of 51% on this new data, while the MFCCs model had a lower accuracy of 29%.

## Instructions for Reproduction

Download GTZAN dataset from Kaggle (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and save it to your Google Drive.

Run the corresponding Jupyter Notebook for the reproduction of each model.

## Citations
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.
- Hershey, S., Chaudhuri, S., Ellis, D. P., Gemmeke, J. F., Jansen, A., Moore, R. C., ... & Wilson, K. (2017). CNN architectures for large-scale audio classification. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 131-135). IEEE.
- Pons, J., Lienen, T., & Serra, X. (2020). Downbeat detection for drum samples using deep learning on spectrograms. In Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR 2020).




