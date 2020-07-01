from setuptools import setup, find_packages

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(name="imagine",
      version="2.0.0-alpha",
      description="Intestellar MAny field inference enGINE",
      license="GPLv3",
      url="https://github.com/IMAGINE-Consortium/imagine/",
      author="IMAGINE Consortium",
      author_email="jiaxin.wang@sjtu.edu.cn, luizfelippesr@gmail.com",
      maintainer="Jiaxin Wang, Luiz Felippe S. Rodrigues",
      maintainer_email="jiaxin.wang@sjtu.edu.cn, luizfelippesr@gmail.com",
      packages=find_packages(),
      include_package_data=True,
      platforms="any",
      python_requires='>=3.5',
      install_requires=requirements,
      #dependency_links=[
        #'git+https://bitbucket.org/hammurabicode/hamx.git#egg=hampyx'],
      zip_safe=False,
      classifiers=["Development Status :: 3 - Alpha",
                   "Intended Audience :: Science/Research",
                   "Topic :: Utilities",
                   "License :: OSI Approved :: GNU General Public License v3 "
                   "or later (GPLv3+)"],)
