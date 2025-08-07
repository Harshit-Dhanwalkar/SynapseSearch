from setuptools import setup

setup(
    name="SynapseSearch",
    install_requires=[
        "flask>=2.0.0",
        "bleach>=4.0.0",
        "flask-limiter>=2.0.0",
        "symspellpy>=6.7.0",
        "nltk>=3.6.0",
        "scipy>=1.10.1",
        "flask-caching>=2.0.0",
        'redis>=4.0.0; platform_system!="Windows"',
    ],
)
