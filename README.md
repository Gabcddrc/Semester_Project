# Chinese Abbreviations Translation 
In this project, we attempts to translate Chinese Internet Slang with the form of taking the first letter of the Pinyin representation of each Chinese character
in a Chinese phrase/word (head abbreviations). We train with three different type of models, namely a seq2seq model, a seq2word model and a multiple choice model.

## Models
The models can be found in the 'model' folder: multiple_choice, seq2seq and seq2word.

To train the model, the user simply has to run:
````
python model/multiple_choice.py
````

There is also scipts for evaluate the trained model inside the 'evaluate' folder.

After trained the mutiple choice model, 'run_choice.py' can be used to run the model for inference (user has to set the correct path for the trained model in the scipt), 'sample.png' contains some examples.

'run_choice_auto.py' is an automated version of 'run_choice.py', where the user no longer need to supply the choices during inference, however the user has supply a dictionary which maps head abbreviations to possible Chinese words, the sample dictionary is 'nbnnssh.pkl', the path need to be set within the script.

## Dataset
The dataset uses data from 'https://github.com/brightmart/nlp_chinese_corpus', 'dataset_choice.zip' includes the processed dataset for multiple_choice. For the other two model the training dataset can be created by taking a chinese paragraph, and randomly convert a word to its head abbreviation form, and use the original paragraph/word as the ground truth.
