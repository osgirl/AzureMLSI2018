import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="InitializationScript",
    version="0.0.1",
    author="Caleb",
    author_email="Wharton_caleb@bah.com",
    description="Initialization script for the Azure ML demonstration system running outside the Azure cloud",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/booz-allen-hamilton/AzureMLSI2018",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)