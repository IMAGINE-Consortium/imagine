FROM ubuntu:latest

# users may need python3-tk, please install it manually: apt-get install python3-tk

RUN apt-get update
RUN apt-get install -y --fix-missing build-essential time gfortran libblas-dev liblapack-dev gsl-bin libgsl-dev autoconf cmake
RUN apt-get install -y --fix-missing git wget unzip vim valgrind python3 python3-pip python3-dev
RUN pip3 install jupyter numpy scipy astropy healpy matplotlib progressbar corner

RUN mkdir /home/lab
WORKDIR /home/lab

#mpi4py
RUN apt-get install -y --fix-missing libopenmpi-dev openmpi-bin openmpi-doc
RUN pip3 install mpi4py

#MultiNest
RUN apt-get install -y --fix-missing libblas3 libblas-dev liblapack3 liblapack-dev libatlas-base-dev libatlas3-base
RUN git clone https://github.com/JohannesBuchner/MultiNest.git
WORKDIR /home/lab/MultiNest/build
RUN cmake .. && make && make install
WORKDIR /home/lab

# PyMultiNest
RUN git clone https://github.com/JohannesBuchner/PyMultiNest.git
WORKDIR /home/lab/PyMultiNest
RUN python3 setup.py install
WORKDIR /home/lab

#Dynesty
RUN pip3 install dynesty

# HDF5
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz && mkdir hdf5 && tar xzf hdf5-1.10.5.tar.gz -C hdf5 --strip-components 1
WORKDIR /home/lab/hdf5
RUN ./configure --enable-parallel --enable-shared --prefix=/usr/local && make && make install
WORKDIR /home/lab

# h5py
RUN git clone https://github.com/h5py/h5py.git h5py
WORKDIR /home/lab/h5py
RUN export CC=mpicc && python3 setup.py configure --mpi --hdf5=/usr/local/hdf5 && python3 setup.py build && python3 setup.py install
WORKDIR /home/lab

#IMAGINE
RUN git clone -b mpi https://ricphy@bitbucket.org/hammurabicode/imagine.git imagine
WORKDIR /home/lab/imagine
RUN python3 setup.py install
WORKDIR /home/lab

# CLEAN PACKAGES
RUN rm -f *.tar.gz
# SET PATH
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# ADJUSTMENTS
RUN echo "alias mpirun='mpirun --mca btl ^vader --allow-run-as-root'" >> ~/.bashrc
RUN echo "alias python='python3'" >> ~/.bashrc
