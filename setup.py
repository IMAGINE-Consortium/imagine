from setuptools import setup, find_packages

setup(name="imagine",
      version="1.0.0",
      description="Intestellar MAny field inference enGINE",
      license="GPLv3",
      url="https://bitbucket.org/hammurabicode/imagine",
      author="Jiaxin Wang, Theo Steininger",
      author_email="jiaxin.wang@sjtu.edu.cn",
      maintainer="Jiaxin Wang",
      maintainer_email="jiaxin.wang@sjtu.edu.cn",
      packages=find_packages(),
      include_package_data=True,
      platforms="any",
      python_requires='>=3.5',
      install_requires=['numpy', 'mpi4py', 'h5py'],
      zip_safe=False,
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Topic :: Utilities",
                   "License :: OSI Approved :: GNU General Public License v3 "
                   "or later (GPLv3+)"],)
