import os
from setuptools import setup, find_packages

exec(open('imagine/version.py').read())

setup(name = "imagine",
      version = __version__,
      description = ("Intestellar MAny field inference enGINE"),
      license = "GPLv3",
      url = "https://gitlab.mpcdf.mpg.de/ift/IMAGINE",
      packages=find_packages(),
      #package_data={'imagine.simulators.hammurabi': ['input/*'],},
      #package_dir={"imagine": "imagine"},
      dependency_links=['git+https://gitlab.mpcdf.mpg.de/ift/nifty.git/@NIFTy_5#egg=ift_nifty-5.0.0'],
      python_requires='>=3.5',
      install_requires=['nifty5>=5.0.0', 'simplejson'],
      zip_safe=False,
      classifiers=["Development Status :: 4 - Beta",
                   "Topic :: Utilities",
                   "License :: OSI Approved :: GNU General Public License v3 "
                   "or later (GPLv3+)"],
)
