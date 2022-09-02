# Take it easy -bot

[]: # Language: markdown
[]: # Path: Take-it-easy\take-it-easy-ml\README.md
Machine learning project, which uses a neural network to predict the optimal game strategy in the board game [Take it easy](https://en.wikipedia.org/wiki/Take_It_Easy_(game).

## Libraries

Game environment is implemented using [gym](https://gym.openai.com/).
Neural network is implemented using [keras](https://keras.io/) and [keras-rl](https://keras-rl.readthedocs.io/en/latest/).

## Installation
install required libraries using ´´´pip install -r requirements.txt´´´

## Training

### Quick start
To train the model, run the following command in the terminal:

```python take-it-easy.py```

This will train the model and in case of improvement, save it to the file `dqn_take-it-easy-weights.dfh5`.



You can use either the Jupyter notebook or the pure python script to train and test the model. Using the .py script is more straightforward, but the notebook may be more convenient for the user.

Best-performing model is automatically saved in the file `dqn_take-it-easy_weights.hdf5`. If you want to train the model from scratch, delete the files `dqn_take-it-easy_weights.hdf5` and `highscore.txt`.



## Testing and practical use

You can not yet use the model in the real game, but you can use the `test-model.ipynb` notebook to test the model and get graphical feedback on the performance. Keep in mind that the rewards used by the model are double the ones it would get playing the real board game when training, but real when using the testing notebook.


