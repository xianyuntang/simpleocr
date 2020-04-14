import setuptools

setuptools.setup(
    name="quickocr",  #
    version="0.0.1",
    author="xt1800i",
    author_email="xt1800i@gmail.com",
    description="A small example package",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=['quickocr'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "opencv-python",
        "tensorflow-gpu",
        "requests"
    ],

)
