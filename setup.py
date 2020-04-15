import setuptools

setuptools.setup(
    name="simple-ocr",  #
    version="0.0.3",
    author="xt1800i",
    author_email="xt1800i@gmail.com",
    description="A small example package",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/xianyuntang/quickocr",
    packages=setuptools.find_packages(include=['quickocr']),

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
