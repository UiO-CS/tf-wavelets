from setuptools import setup

# Install python package
setup(
    name="tfwavelets",
    version=0.1,
    author="Kristian Monsen Haug, Mathias Lohne",
    author_email="mathialo@ifi.uio.no",
    license="MIT",
    description="TensorFlow implementation of descrete wavelets",
    url="https://github.com/UiO-CS/tf-wavelets",
    install_requires=["tensorflow", "numpy"],
    packages=["tfwavelets"],
    zip_safe=False)
