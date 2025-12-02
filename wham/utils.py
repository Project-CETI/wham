import os
import shutil

def pathify(string):
    bad_chars = [" ", ".", "/", ":"]
    return "".join([c if c not in bad_chars else "_" for c in string])


def move_wav_files(source_dir, destination_dir):
    """
    Recursively moves all .wav files from the source directory to the destination directory.
    Renames the files numerically starting from 1.

    This function is necessary because the vampnet generation script saves generations in a sequence of subdirectories,
    To simplify the code for evaluating embeddings, we require all .wav files to be saved under the same folder.

    Parameters:
        source_dir (str): The directory to search for .wav files.
        destination_dir (str): The directory to move .wav files to.
    """
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Initialize a counter for renaming files
    file_counter = 1

    # Walk through the directory tree
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.wav'):
                source_path = os.path.join(root, file)
                new_file_name = f"{file_counter}.wav"
                destination_path = os.path.join(destination_dir, new_file_name)

                try:
                    shutil.move(source_path, destination_path)
                    file_counter += 1
                except Exception as e:
                    print(f"Error moving {file}: {e}")

