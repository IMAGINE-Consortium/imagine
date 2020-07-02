# Built-in imports
from codecs import open
import re

# Package imports
from setuptools import find_packages, setup

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Read the __version__.py file
with open('imagine/__version__.py', 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

setup(name="imagine",
      version=version,
      description="Interstellar MAGnetic field INference Engine",
      license="GPLv3",
      url="https://github.com/IMAGINE-Consortium/imagine",
      author="IMAGINE Consortium",
      author_email="jiaxin.wang@sjtu.edu.cn, luizfelippesr@gmail.com",
      maintainer="Jiaxin Wang, Luiz Felippe S. Rodrigues",
      maintainer_email="jiaxin.wang@sjtu.edu.cn, luizfelippesr@gmail.com",
      packages=find_packages(),
      include_package_data=True,
      platforms="UNIX",
      python_requires='>=3.5',
      install_requires=requirements,
      zip_safe=False,
      classifiers=["Development Status :: 3 - Alpha",
                   "Intended Audience :: Science/Research",
                   "Topic :: Utilities",
                   "License :: OSI Approved :: GNU General Public License v3 "
                   "or later (GPLv3+)"],)
