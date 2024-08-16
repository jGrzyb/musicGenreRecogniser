# musicGenreRecogniser
### It is a python project that uses Tensorflow, Keras, and Librosa libraries to recognise music genre with recursive neural network.

To train model you need to downlowad dataset from [here](https://huggingface.co/datasets/marsyas/gtzan/blob/main/data/genres.tar.gz) and run [music.ipynb](https://github.com/jGrzyb/musicGenreRecogniser/blob/main/music.ipynb) script to preprocess data. Then you need to run [musicGenreTrainer.py](https://github.com/jGrzyb/musicGenreRecogniser/blob/main/musicGenreTrainer.py) script and train the model. It will save ml model to musicGenreRecogniser.keras file

To recognise music genre run [musicGenreRecogniser.py](https://github.com/jGrzyb/musicGenreRecogniser/blob/main/musicGenreRecogniser.py) script (you must train model first)
