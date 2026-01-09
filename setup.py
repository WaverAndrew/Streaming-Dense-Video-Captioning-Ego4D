from setuptools import setup, find_packages

setup(
    name="streaming_dvc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentencepiece",
        "protobuf",
        "numpy",
        "openai-clip",
        "ffmpeg-python",
    ],
)
