# WhisperX for LJ Speech

Automatically create datasets in [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) format to use for training TTS (
text-to-speech) models. LJ Speech is a common standard and is used in TTS frameworks such
as [Tortoise](https://github.com/neonbjb/tortoise-tts) or [Piper](https://github.com/rhasspy/piper).

Segments detected in VAD step in [WhisperX](https://github.com/m-bain/whisperX) are used to create short samples in
`.wav` format and WhisperX ASR is used to
create the corresponding transcriptions.

## Install

`pip install -r requirements.txt`

## Usage

1. Put your (possibly long) audio files containing spoken audio into `input_audio`
2. Run `python create_dataset.py --model base --gpu 0 --input input_audio --output output` if you have a GPU
   available, or on CPU: `python create_dataset.py --model tiny --cpu --input input_audio --output output`
3. Output:
    * Processed audio samples are saved as `.wav` files in the `output/audio` directory
    * A `metadata.csv` file is generated, containing entries in the format
      ```
      000001_000001|Transcribed text of first audiosample.     
      000001_000002|Transcribed text of the second audiosample. 
      ...
      ```

## Docker

* Build Dockerimage: `docker build -t whisperx4ljspeech .`
* Run
  `docker run --gpus '"device=0"' -v $(pwd)/input_audio:/app/input_audio -v $(pwd)/output:/app/output whisperx4ljspeech --input input_audio/ --output output --gpu 0 --model large-v3 --language de`

