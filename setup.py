import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name="interact",
    version="0.3",
    packages=setuptools.find_packages(),
    url="https://github.com/rystrauss/interact",
    license="LICENSE",
    author="Ryan Strauss",
    author_email="ryanrstrauss@icloud.com",
    description="A reinforcement learning library written in TensorFlow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "tensorflow==2.4.1",
        "gym",
        "tensorflow-probability==0.11",
        "tqdm",
        "click",
        "opencv-python",
        "ray",
        "gin-config>=0.4",
    ],
    extras_require={"dev": ["pytest"]},
)
