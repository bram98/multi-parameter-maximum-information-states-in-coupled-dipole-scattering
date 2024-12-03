from setuptools import setup, find_packages

setup(
    name='scripts',
    version='0.1.0',
    py_modules=find_packages(),  # List all your modules here
    install_requires=['numpy', 'matplotlib', 'scipy'],  # Add dependencies if needed
    description='Collection of libraries for the project',
)