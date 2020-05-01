import setuptools

long_description = open('Readme.md', 'r').read()

setuptools.setup(
    name="simpleocr",  #
    version="0.0.8",
    author="xt1800i",
    author_email="xt1800i@gmail.com",
    description="A ocr tool for traditional chinese",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xianyuntang/simpleocr",
    packages=setuptools.find_packages(include=['simpleocr']),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "opencv-python",
        "tensorflow-cpu",
        "requests"
    ],
    include_package_data=True,

)
