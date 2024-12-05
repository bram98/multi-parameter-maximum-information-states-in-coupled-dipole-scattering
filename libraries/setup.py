from setuptools import find_packages, setup

setup(
    name="libraries",
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages and subpackages
    install_requires=[],  # Add dependencies if needed
    description="Collection of libraries for the project",
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
)
