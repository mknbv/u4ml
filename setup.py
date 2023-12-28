from distutils.core import setup

setup(
    name="u4ml",
    version="0.1dev",
    description="Utilities for machine learning projects",
    author="Mikhail Konobeev",
    author_email="konobeev.michael@gmail.com",
    url="https://github.com/MichaelKonobeev/u4ml/",
    license="MIT",
    packages=["u4ml"],
    install_requires=[
        "ipython>=7.31.1",
        "matplotlib>=3.5.1",
        "numpy>=1.16.4",
        "scipy>=1.5.0",
        "tqdm",
    ],
    extras_require={
        "torch": ["torch>=1.5.1"],
        "tensorboard": ["tensorboard>=1.15"],
        "tensorflow": ["tensorflow>=2.8.0"],
    },
)
