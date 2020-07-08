import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="superpoint", # Replace with your own username
    version="0.0.1",
    author="You-Yi Jau",
    author_email="eric.yyjau@gmail.com",
    description="Superpoint in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eric-yyjau/pytorch-superpoint",
    # packages=setuptools.find_packages(),
    # py_modules=['datasets', 'models', 'utils', 'train4', 'helloworld'],
    py_modules=['superpoint'],
    package_dir={'.': '.'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

