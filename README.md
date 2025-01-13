# Sentiment Analysis Project

This project demonstrates how to perform sentiment analysis on the IMDB movie review dataset using the DistilBERT model, and the Trainer class from the Hugging Face Transformers library. The goal is to classify movie reviews as either positive or negative.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Results and Performance](#results-and-performance)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a piece of text. In this project, we use the DistilBERT model, and the Trainer class to classify movie reviews from the IMDB dataset as positive or negative.

## Installation

To install the necessary dependencies, run the first cell in the sentiment-analysis.ipynb notebook

## Dataset

The dataset used in this project is the IMDB movie review dataset. It contains 50,000 movie reviews labeled as either positive or negative.

## Model

This project used the DistilBERT model for sequence classification. DistilBERT is a smaller, faster, and cheaper version of BERT, making it suitable for this task.

## Training

The training process involves feeding the dataset into the model and optimizing it using backpropagation. Training parameters and configurations are specified in the notebook.

## Results and Performance

The model achieves an evaluation accuracy of approximately 80.87% on the test set. Actual performance metrics will vary based on the specific training run and dataset size

## Limitations

• Limited to binary classification (positive/negative)
• Training dataset size reduced for demonstration purposes
• May not capture nuanced or ambiguous sentiments
• Performance dependent on review text length and complexity

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the following libraries and resources:

• Hugging Face Transformers
• IMDB movie review dataset
• PyTorch
• Scikit-learn

Special thanks to the developers and contributors of these libraries for their excellent work.
