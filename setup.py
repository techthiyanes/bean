from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="bean",
    version="0.0.1",
    author="Ogundepo Odunayo", "Iyanuoluwa Shode",
    author_email="oogundep@uwaterloo.ca", "shodei1@montclair.edu",
    description="BEnchmarking Africa NLP (BeAN 🫘), A framework for easy evaluation of language models on several Africa NLP datasets",
    packages=find_packages(),
    url="",
    install_requires=requirements,
    package_data={"bean": [
        "datasets/config/*.yaml",
     ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
