# setup.py

from setuptools import find_packages, setup

setup(
    name="SynapseSearch",
    version="0.1.0",
    description="A mini search engine built with Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Harshit Prashant Dhanwalkar",
    url="https://github.com/Harshit-Dhanwalkar/SynapseSearch.git",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=3.0.0",
        "flask-caching>=2.3.0",
        "flask-limiter>=3.0.1",
        "bleach>=4.0.0",
        "symspellpy>=6.7.0",
        "nltk>=3.9.0",
        "scipy>=1.10.1",
        "numpy==1.26.1",
        "scikit-learn==1.3.2",
        "faiss-cpu==1.7.4",
        "sentence-transformers==2.2.2",
        "pillow>=11.3.0",
        'redis>=4.0.0; platform_system!="Windows"',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires="==3.11",
    entry_points={
        "console_scripts": [
            "synapsesearch=synapsesearch.cli:main",
        ],
    },
)
