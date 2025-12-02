from transformers import Wav2Vec2Processor
import soundfile as sf
import librosa

SAMPLING_RATE = 16000

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")


def preprocess_audio(audio_file_path):
    # Load an audio file
    audio_input, sr = sf.read(audio_file_path)

    # Resample
    if sr != SAMPLING_RATE:
        audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=SAMPLING_RATE)

    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)

    # Use the processor to prepare the inputs
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_values

    return input_values