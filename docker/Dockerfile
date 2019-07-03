FROM ubuntu:latest

# users may need python3-tk, please install it manually: apt-get install python3-tk

RUN apt-get update
RUN apt-get install -y --fix-missing build-essential time gfortran libblas-dev liblapack-dev gsl-bin libgsl-dev autoconf cmake
RUN apt-get install -y --fix-missing git wget unzip vim valgrind python3 python3-pip python3-dev
RUN pip3 install jupyter numpy scipy astropy healpy matplotlib progressbar corner

RUN mkdir /home/lab
WORKDIR /home/lab

# CFITSIO
RUN mkdir /home/lab/cfitsio
RUN wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz && tar xzf cfitsio_latest.tar.gz -C cfitsio --strip-components 1
WORKDIR /home/lab/cfitsio
RUN ./configure --prefix=/usr/local/ && make && make install 
WORKDIR /home/lab

# FFTW
RUN mkdir /home/lab/fftw
RUN wget http://www.fftw.org/fftw-3.3.8.tar.gz && tar xzf fftw-3.3.8.tar.gz -C fftw --strip-components 1
WORKDIR /home/lab/fftw
RUN ./configure --enable-threads --enable-openmp --enable-shared --prefix=/usr/local/ && make && make install
WORKDIR /home/lab

# HEALPIX
RUN mkdir /home/lab/healpix
RUN wget http://downloads.sourceforge.net/project/healpix/Healpix_3.50/Healpix_3.50_2018Dec10.tar.gz && tar xzf Healpix_3.50_2018Dec10.tar.gz -C healpix --strip-components 1
WORKDIR /home/lab/healpix
# write config file on fly
RUN echo '4\n\
	/sur/local/bin\n\
	/usr/local/include\n\
	4\n\
	y\n\
	0\n'\
	> hlpx_config
RUN ./configure -L < hlpx_config && make 
WORKDIR /home/lab
ENV HEALPIX_TARGET optimized_gcc 
ENV HEALPIX /home/lab/healpix

# GOOGLE TEST
RUN git clone https://github.com/google/googletest.git googletest
WORKDIR /home/lab/googletest
RUN mkdir /home/lab/googletest/build
WORKDIR /home/lab/googletest/build
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. && make && make install && cp ../googletest/src/* /usr/local/src
WORKDIR /home/lab

#mpi4py
RUN apt-get install -y --fix-missing libopenmpi-dev openmpi-bin openmpi-doc
RUN pip3 install mpi4py

#(Py)MultiNest
RUN apt-get install -y --fix-missing libblas3 libblas-dev liblapack3 liblapack-dev libatlas-base-dev libatlas3-base

RUN git clone https://github.com/JohannesBuchner/MultiNest.git
WORKDIR /home/lab/MultiNest/build
RUN cmake .. && make && make install
WORKDIR /home/lab

RUN git clone https://github.com/JohannesBuchner/PyMultiNest.git
WORKDIR /home/lab/PyMultiNest
RUN python3 setup.py install
WORKDIR /home/lab

#Dynesty
RUN pip3 install dynesty

#hdf5
RUN apt-get install -y --fix-missing libhdf5-100 libhdf5-dev libhdf5-openmpi-100 libhdf5-openmpi-dev hdf5-tools

#h5py
RUN pip3 install h5py

#NIFTy
RUN pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_5
RUN pip3 install --user git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft

# HAMMURABI X
RUN git clone https://bitbucket.org/hammurabicode/hamx hamx
WORKDIR /home/lab/hamx
RUN mkdir /home/lab/hamx/build
WORKDIR /home/lab/hamx/build
RUN cmake .. && make install
# SET PATH
ENV PATH /usr/local/hammurabi/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/hammurabi/lib:${LD_LIBRARY_PATH}
WORKDIR /home/lab

#IMAGINE
RUN git clone https://ricphy@bitbucket.org/hammurabicode/imagine.git imagine
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
