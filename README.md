# speech-to-text
This framework provides python scripts to train neural networks for speech recognition.

## Requirements
- Python 3.6+

## Prerequisites
### Installing Dependencies
Install the required python dependencies listed in the `requirements.txt`:
```shell
pip install -r requirements.txt
```

### Providing training data
To run a training, training data is required.
A Training accepts a file, which contains metadata about the training data. 
The file itself is a JSON file consisting of an array.
Each element has the following properties:

- `path`: Absolute path to the audio file
- `text`: Transcription of the audio file

A training data file could look like this:
```json
[
  {
    "path": "path/to/audio/file1.wav",
    "text": "hello world"
  },
  {
    "path": "path/to/audio/file2.wav",
    "text": "goodbye world"
  }
]
```
You can find a downloader for the voxforge corpus at [https://github.com/KevNetG/speech-to-text-voxforge](https://github.com/KevNetG/speech-to-text-voxforge). 
This repo also includes a `generator.py` file, which creates a training data file containing the required metadata for trainings.

Currently, only WAVE files are supported.

## Usage

### Configuring a training
The idea is, that you don't write you training configuration into the command line, 
but instead into a file, which you can modify and reuse for other trainings.
You can find a sample configuration under `examples/training.config.json` and adjust it to your needs.
A configuration file has the following properties:

- `epochs`: Number of epochs to train
- `batchSize`: Batch Size
- `trainingDataQuantity`: Amount of training data that is taken from the provided sources
- `net`: Name of the model. Models are specified in the `models.py` file
- `trainingData`: Absolute Path to a training data file. You can specify multiple sources to simply scale your amount of available training data
- `alphabetPath`: Path to an alphabet file

```json
{
  "epochs": 10,
  "batchSize": 20,
  "trainingDataQuantity": 50000,
  "net": "graves",
  "trainingData": [
    "speech-to-text/training_data.json"
  ],
  "alphabetPath": "speech-to-text/examples/english.json"
}
```

You can use the english alphabet available under `examples/english.json` or create one yourself for any other language.
The alphabet file is a simple JSON file consisting of an array containing the characters from the alphabet:

### Running a training
To run a training execute the `train.py` and provide two arguments:

- `path`: Where to store the training. You don't have to specify a file extension
- `plan`: The path to a training configuration

like this:

```shell
python train.py "trainings/graves" "training_data.json" 
```

Trainings are saved after each epoch.

### Continuing an interrupted training
If you had to stop a training prematurely, you can continue it from the last checkpoint. 
Simply execute the `continuetraining.py` and pass the path to the training save file. 
You don't have to specify the file extension:

```
python continuetraining.py "trainings/graves.json"
```

### Making a prediction
In order to create a transcription of an audio file, use the `predict.py` script.
Pass the following arguments: 

- `path`: Path to a training save file
- `weights`, Path to a weights matrix
- `audio`: Path to the audio file which shall be transcribed

For example:
```shell
python predict.py "trainings/graves.json" "trainings/graves.weights-20-65.68075.h5" "media/audio.wav"
```

### Displaying training statistics
If you want to display the training loss and the validation loss of a training, execute the `statistics.py` script:
```shell
python statistics.py "trainings/graves.statistics.json"
```
