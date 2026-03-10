from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="wham",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Project-CETI/wham",
    license="MIT",
    packages=find_packages(),
    package_dir={},
    install_requires=[
        "descript-audiotools @ git+https://github.com/hugofloresgarcia/audiotools.git",
        "argbind",
        "pandas",
        "pathlib",
        "pydub",
        "ffmpeg-python",
        "tqdm",
        "scikit-learn",
        "wandb",
        "gdown",  # For fetching large files from Google Drive
        "soundfile",
        "transformers",
        "torch",
        "Cython",
        "fadtk",
        "urllib3==2.0"
    ],
)
