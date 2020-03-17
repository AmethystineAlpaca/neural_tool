from setuptools import setup

setup(
    name='neural_tools',
    version='1.0',
    description='A useful module',
    author='ming',
    # packages=['torch'],  # same as name
    # external packages as dependencies
    install_requires=['torch', 'pandas', 'sklearn', 'transformers', 'nltk'],
)
