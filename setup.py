from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="sentiment_classification",
    version=version,
    packages=["sentiment_classification"],
)
