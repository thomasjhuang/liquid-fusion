from setuptools import setup, find_packages

setup(
    name="liquid-fusion",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "rouge-score",
        "tqdm"
    ],
)