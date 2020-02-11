import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distgfs", 
    version="0.0.1",
    author="Ivan Raikov",
    author_email="ivan.g.raikov@gmail.com",
    description="Distributed global function search via dlib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iraikov/distgfs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=[
        'mpi4py', 'numpy', 'h5py', 'distwq'
    ],

)
