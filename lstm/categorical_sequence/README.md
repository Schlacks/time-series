# Categorical Sequences

### Intro
Time series anaylses can be applied at places one least expects them. In this example, we want
to predict the gender of web page user based on their browsing behaviour. The sample data consists
of users, their visited pages and the corrsponding timestamp. The url's of the visited pages are uuid 
encoded.

Treating the browsing behaviour as a sequence of categorical data points, we can employ techniques 
that are based on NPL processing such as sentiment analysis. Adapting an appropriate code 
(e.g. https://www.kaggle.com/code/arunmohan003/sentiment-analysis-using-lstm-pytorch/notebook) the resulting LSTM based algorithm 
performs more than satisfactory, reaching an accuracy of 98% on the validation data.

### Setup

in source run:
```
conda create -n categorical_sequence
source activate categorical_sequence
conda install pip
pip install -r requirements.txt
```

unzip the user_path_data.zip in /data

In config.ini feel free to play around with the hyperparameters of the LSTM model.

in /src run:

``` 
python3 main.py
```

