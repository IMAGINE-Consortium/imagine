FROM continuumio/miniconda3
ARG BRANCH=master
ARG REPO=IMAGINE-Consortium/imagine

RUN echo REPO=$REPO  BRANCH=$BRANCH

# Install build environment (for hamx)
RUN apt-get update
RUN apt-get install -y --fix-missing git wget vim
RUN apt-get install -y --fix-missing build-essential autoconf automake cmake gfortran
RUN apt-get install -y --fix-missing gsl-bin libgsl-dev

# IMAGINE download
RUN mkdir /home/lab
WORKDIR /home/lab
RUN echo BRANCH: $BRANCH && echo REPO $REPO
RUN git clone -b $BRANCH https://github.com/$REPO.git imagine

# IMAGINE dependencies
WORKDIR /home/lab/imagine
RUN conda env create --file=imagine_conda_env.yml
# Make RUN commands use the 'imagine' environment by default:
SHELL ["conda", "run", "-n", "imagine", "/bin/bash", "-c"]

# Python extra dependencies (for installation, testing and tutorials)
# nomkl slightly reduces the size of the image
RUN conda install -y -c conda-forge pip seaborn nomkl pytest-cov pytest-mpl nbval astroquery requests jupyterlab
RUN python -m ipykernel install --user --name imagine --display-name 'Python (imagine)'
WORKDIR /home/lab

### ---- for hammurabi X -----
# FFTW
RUN mkdir /home/lab/fftw
WORKDIR fftw
RUN wget http://www.fftw.org/fftw-3.3.8.tar.gz && tar -xzf fftw-3.3.8.tar.gz --strip-components 1
RUN ./configure --enable-threads --enable-openmp --enable-shared --prefix=/tmp/local/ && make && make install
WORKDIR /home/lab

# HAMMURABI X
RUN mkdir /home/lab/hamx
WORKDIR /home/lab/hamx
RUN wget https://github.com/hammurabi-dev/hammurabiX/archive/v2.4.1.tar.gz && tar -xzf v2.4.1.tar.gz --strip-components 1 && rm v2.4.1.tar.gz
# Deactivates testing (and together with it, the google tests dependency)
RUN sed -E 's/(OPTION.ENABLE_TESTING .+) ON/\1 OFF/g' CMakeLists.txt > CMakeLists_new.txt
RUN mv CMakeLists_new.txt CMakeLists.txt
RUN mkdir /home/lab/hamx/build
WORKDIR /home/lab/hamx/build
RUN cmake .. && make install
WORKDIR /home/lab/hamx
RUN pip install -e .
### ----- end of hammurabi X -----

# INSTALL IMAGINE!
WORKDIR /home/lab/imagine
RUN pip install -e .
WORKDIR /home/lab

# IMAGINE datasets extension (installs latest version)
RUN pip install git+https://github.com/IMAGINE-Consortium/imagine-datasets.git
RUN mkdir /home/lab/.imagine_datasets_cache
ENV IMAGINE_DATASETS_CACHE_DIR /home/lab/.imagine_datasets_cache

# CLEAN PACKAGES
RUN conda clean -afy && find /opt/conda/ -follow -type f -name '*.pyc' -delete
RUN rm -f *.tar.gz
RUN rm -rf /home/lab/fftw
RUN apt-get autoremove && apt-get clean

# SET PATH
ENV PATH /tmp/local/hammurabi/bin:${PATH}
ENV LD_LIBRARY_PATH /tmp/local/lib:/tmp/local/hammurabi/lib:${LD_LIBRARY_PATH}


# MISC
# Activates imagine conda environment by default
RUN sed -i 's/conda activate base/conda activate imagine/g' ~/.bashrc
# Alias to make mpirun work
RUN echo 'alias mpirun='\''mpirun --mca btl ^vader --allow-run-as-root'\''' >> ~/.bashrc
# Launching script to simplify starting jupyterlab
RUN echo '#!/bin/bash' >> ~/jupyterlab.bash
RUN echo 'source ~/.bashrc && jupyter-lab --notebook-dir=/home/lab --ip='\''*'\'' --port=8888 --no-browser --allow-root'   >> ~/jupyterlab.bash
RUN chmod +x ~/jupyterlab.bash
# Alias to simplify launching the jupyterlab in interactive mode
RUN echo "alias jupyter-lab='~/jupyterlab.bash'"  >> ~/.bashrc
