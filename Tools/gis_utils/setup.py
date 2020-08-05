import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gis-utils-pkg-dillhicks", # Replace with your own username
    version="0.0.4",
    author="Dillon Hicks",
    author_email="sdhicks@ucsd.edu",
    description="GIS Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UCSD-E4E/mangrove/tree/master/Tools/gis_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)