import setuptools

setuptools.setup(
    name="SymPDE", # Replace with your own username
    version="0.1",
    author="Katharine Long",
    author_email="katharine.long@ttu.edu",
    description="PDEs",
    long_description="Symbolic description of PDEs",
    long_description_content_type="text/markdown",
    url="https://github.com/krlong014/PyNonlin",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: LGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
