import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Auto Differentiation",
    version="0.1.0",
    author="Abdullah Alnami",
    author_email="bdallhsydalnmy@gmail.com",
    description="A tiny scalar-valued auto differentiation engine with a small PyTorch-like neural network library on top.",
    url="https://github.com/Abdullah1Allnami/Auto_Diffrentiation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
