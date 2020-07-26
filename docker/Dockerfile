FROM continuumio/miniconda3

MAINTAINER jiaxin.wang@sjtu.edu.cn

RUN apt-get update
RUN apt-get install -y --fix-missing build-essential autoconf automake cmake gfortran
RUN apt-get install -y --fix-missing gsl-bin libgsl-dev
RUN apt-get install -y --fix-missing git wget vim

RUN mkdir /home/lab
WORKDIR /home/lab

# IMAGINE download
RUN git clone https://github.com/IMAGINE-Consortium/imagine.git imagine

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

# GOOGLE TEST
RUN git clone https://github.com/google/googletest.git googletest
WORKDIR /home/lab/googletest
RUN mkdir /home/lab/googletest/build
WORKDIR /home/lab/googletest/build
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/tmp/local .. && make && make install
RUN mkdir /tmp/local/src && cp ../googletest/src/* /tmp/local/src
WORKDIR /home/lab

# HAMMURABI X
RUN git clone https://github.com/hammurabi-dev/hammurabiX.git hamx
WORKDIR /home/lab/hamx
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

# CLEAN PACKAGES
RUN conda clean -afy && find /opt/conda/ -follow -type f -name '*.pyc' -delete
RUN rm -f *.tar.gz
RUN rm -rf /home/lab/googletest
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
RUN echo 'source ~/.bashrc && jupyter-lab --notebook-dir=/home/lab --ip='\''*'\'' --port=8888 --no-browser --allow-root'   >> ~/jupyterlab.bash
# Alias to simplify launching the jupyterlab in interactive mode
RUN echo "alias jupyter-lab='~/jupyterlab.bash'"  >> ~/.bashrc
