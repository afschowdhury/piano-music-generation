 # Piano Music Composer 

## Overview

In this task, your goal is to train a transformer model, called **piano music composer**, to generate piano music.

## Data

The piano data (in MIDI format) is downloaded in from the following link: 
```
https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip
```

It has been unzipped to this dir: `maestro-v1.0.0`.

### Supporting Files

- **`midi2seq.py`**: Contains a set of functions that help to process the MIDI data and convert the data to sequences of events.
- **`model_base.py`**: Contains the base classes that you should inherit when implementing the following model classes.

## Composer Implementation

Implement a class called **`Composer`**. It should be a subclass of the class `ComposerBase`. You must use the exact class name.

In this task, you should implement a language model as a composer. You should explore different transformer model architectures to obtain a good composer.

### Requirements

When the `compose` member function is called, it should return a sequence of events that can be translated into piano music:

- The music should play for **at least 20 seconds**.
- **Randomness is required** in the implementation of the compose function such that each call to the function should generate a different sequence.

The function `seq2piano` in `midi2seq.py` can be used to convert the sequence into a MIDI object, which can be written to a MIDI file and played on a computer. Train the language model using the downloaded piano plays.

## Task

Develop and train your model so that your model can compose reasonable piano music pieces at least 20 seconds long.

### Additional Guidelines

- **Do not modify** the file `model_base.py`
- Put all your code in a single file named **`hw2.py`** (you must use this file name)
- Import the `ComposerBase` class in `hw2.py`


### Testing
I will test your implementation using code similar to the following:

```python
from hw2 import Composer
piano_seq = torch.from_numpy(process_midi_seq())
loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz)

cps = Composer()
for i in range(epoch):
    for x in loader:
        cps.train(x[0].cuda(0).long())
        
cps2 = Composer(load_trained=True)
midi = cps2.compose()
midi = seq2piano(midi)
midi.write('piano1.midi')
```

<!-- **Important**: I will do the testing in Google Colab. Make sure code can run in Colab on a GPU node. -->



In the above code, `cps2` should be a `Composer` model with the trained weights loaded. We should be able to call `cps2.compose()` without training it and obtain a piano sequence from the weights of trained model. 