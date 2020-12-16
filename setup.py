import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NLP-CSE583", # Replace with your own username
    version="0.0.1",
    author="Angel Burr, Neilly Herrera Tan, Zhan Shi",
    author_email="",
    description="A small NLP wrapper for mxnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neillytan/NLP-Project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)