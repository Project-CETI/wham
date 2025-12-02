import os
import random
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from wam import IMPULSE_DIR, CODA_DIR

MIN_NUM_CLICKS = 4
MAX_NUM_CLICKS = 8
SAMPLE_LEN = 2000
CLICK_DURATION = 50

def create_silent_wave_with_clicks():
    # Create a 2-second silent wave file
    silent_wave = AudioSegment.silent(duration=SAMPLE_LEN)  # 2 seconds of silence
    
    # Randomly decide how many clicks (between 4 and 8)
    num_clicks = random.randint(MIN_NUM_CLICKS, MAX_NUM_CLICKS)
    
    # Randomly place clicks within the 2-second window
    for _ in range(num_clicks):
        click_time = random.randint(0, SAMPLE_LEN)  # Random time for the click (in milliseconds)
        click = Sine(1000).to_audio_segment(duration=CLICK_DURATION)  # Generate a short click sound (50ms duration)
        silent_wave = silent_wave.overlay(click, position=click_time)  # Overlay the click at the random position
    
    return silent_wave

def add_random_recording_and_save(folder_path, save_path):
    # Get all audio files in the provided folder
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    if not audio_files:
        raise ValueError(f"No .wav files found in {folder_path}")
    
    # Pick a random file from the folder
    random_file = random.choice(audio_files)
    random_file_path = os.path.join(folder_path, random_file)
    
    # Load the audio file
    recording = AudioSegment.from_wav(random_file_path)
    
    # Cut or pad the audio file to 2 seconds
    if len(recording) > SAMPLE_LEN:
        recording = recording[:SAMPLE_LEN]  # Cut it to 2 seconds
    elif len(recording) < SAMPLE_LEN:
        silence_needed = SAMPLE_LEN - len(recording)
        padding = AudioSegment.silent(duration=silence_needed)
        recording = recording + padding  # Add silence to make it 2 seconds
    
    # Create the silent wave with clicks
    silent_wave = create_silent_wave_with_clicks()
    
    # Prepend the recording to the silent wave
    final_wave = recording + silent_wave
    
    # Save the final wave to the specified folder
    save_file_path = os.path.join(save_path, f"final_audio_{random.randint(1000, 9999)}.wav")
    final_wave.export(save_file_path, format="wav")
    print(f"Saved: {save_file_path}")

def generate_audio_files(input_folder, output_folder, num_files=40):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for _ in range(num_files):
        add_random_recording_and_save(input_folder, output_folder)

# Set your input folder (where the recordings are) and output folder (where to save the generated files)
input_folder = CODA_DIR
output_folder = IMPULSE_DIR

# Generate the audio files
generate_audio_files(input_folder, output_folder)
