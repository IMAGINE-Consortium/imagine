# -*- coding: utf-8 -*-
"""
developed by Joe Taylor
based on the initial work of Theo Steininger

warning:
working directory is set as the same directory as this file
it relies on subprocess to fork c++ routine and passing data through disk
so, it is not fast

methods:

# Import class
In []: import hampyx as hpx

# Initialize object
In []: object = hpx.hampyx (exe_path, xml_path)

hammurabiX executable path is by default '/usr/local/hammurabi/bin/hamx'
while xml file path is by default './'

# Modify parameter value from base xml file to temp xml file
In []: object.mod_par (keychain=['key1','key2',...], attrib={'tag':'content'})

# Add new parameter with or without attributes
In []: object.add_par (keychain=['key1','key2',...], subkey='keyfinal', attrib={'tag':'content'})

the new parameter subkey, will be added at the path defined by keychain

# Delete parameter
In []: object.del_par (keychain=['key1','key2',...])

if additional argument opt='all', then all matching parameters will be deleted

the strings 'key1', 'key2', etc represent the path to the desired parameter, going through the xml
the "tag" is the label for the parameter: eg. "Value" or "cue" or "type"
the "content" is the content under the tag: eg. the string for the tag "filename"

# Look through the parameter tree in python
In []: object.print_par(keychain=['key1','key2',...])

this will return the current value of the parameter in the XML associated with the path "key1/key2/.../keyfinal/"

# Run the executable
In []: object(verbose=True/False)

if additional verbose=True (by default is False) hampyx_run.log and hampyx_err.log will be dumped to disk
notice that dumping logs is not thread safe, use quiet mode in threading

after this main routine, object.sim_map will be filled with simulation outputs from hammurabiX
the structure of object.sim_map contains arrays under entries:
(we give up nested dict structure for the convenience of Bayesian analysis)
object.sim_map[('sync',str(freq),str(Nside),'I')] # synchrotron intensity map at 'frequency' 
object.sim_map[('sync',str(freq),str(Nside),'Q')] # synchrotron Q map at 'frequency' 
object.sim_map[('sync',str(freq),str(Nside),'U')] # synchrotron U map at 'frequency' 
object.sim_map[('sync',str(freq),str(Nside),'PI')] # synchrotron pol. intensity at 'frequency' 
object.sim_map[('sync',str(freq),str(Nside),'PA')] # synchrotron pol. angle at 'frequency' (IAU convention)
object.sim_map[('fd','nan',str(Nside),'nan')] # Faraday depth map
object.sim_map[('dm','nan',str(Nside),'nan')] # dispersion measure map

detailed caption of each function can be found with their implementation
"""

import os
import sys
import time
import subprocess
import healpy as hp
import xml.etree.ElementTree as et
import numpy as np
import tempfile as tf
import logging as log


class Hampyx(object):

    """
    default executable path is '/usr/local/hammurabi/bin/hamx'
    default executable path is './params.xml'
    """
    def __init__(self,
                 xml_path='./params.xml',
                 exe_path=None):
        log.debug('initialize Hampyx')
        # current working directory
        self.wk_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        log.debug('set working directory at %s' % self.wk_dir)
        # encapsulated below
        self.exe_path = exe_path
        self.xml_path = xml_path
        # assign tmp file name by base file name
        self.temp_file = self._base_file
        # read from base parameter file
        self.tree = et.parse(self._base_file)
        # simulation output
        self.sim_map_name = {}
        self.sim_map = {}
        # switches
        self._do_sync = False
        self._do_dm = False
        self._do_fd = False
    
    @property
    def exe_path(self):
        return self._exe_path
    
    @property
    def xml_path(self):
        return self._xml_path
    
    """
    by default hammurabiX executable "hamx" is in the same directory as this file
    """
    @exe_path.setter
    def exe_path(self, exe_path):
        if exe_path is None:  # search sys environ
            env = os.environ.get('PATH').split(os.pathsep)
            cnddt = [s for s in env if 'hammurabi' in s]
            for match in cnddt:
                if os.path.isfile(os.path.join(match, 'hamx')):
                    self._exe_path = os.path.join(match, 'hamx')
        else:  # if given
            assert isinstance(exe_path, str)
            self._exe_path = os.path.abspath(exe_path)
        self._executable = self._exe_path
        log.debug('set hammurabiX executable path %s' % str(self._executable))

    @xml_path.setter
    def xml_path(self, xml_path):
        assert isinstance(xml_path, str)
        self._xml_path = os.path.abspath(xml_path)
        self._base_file = self.xml_path
        log.debug('set hammurabiX base XML parameter file path %s' % str(self._base_file))

    @property
    def wk_dir(self):
        return self._wk_dir

    @wk_dir.setter
    def wk_dir(self, wk_dir):
        self._wk_dir = wk_dir

    @property
    def temp_file(self):
        return self._temp_file

    @temp_file.setter
    def temp_file(self, temp_file):
        self._temp_file = temp_file

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree
        log.debug('capture XML parameter tree from %s' % str(tree))

    @property
    def sim_map_name(self):
        return self._sim_map_name

    @sim_map_name.setter
    def sim_map_name(self, sim_map_name):
        try:
            self._sim_map_name.update(sim_map_name)
            log.debug('update simulation map name dict %s' % str(sim_map_name))
        except AttributeError:
            self._sim_map_name = sim_map_name
            log.debug('set simulation map name dict %s ' % str(sim_map_name))

    @property
    def sim_map(self):
        return self._sim_map

    @sim_map.setter
    def sim_map(self, sim_map):
        try:
            self._sim_map.update(sim_map)
            log.debug('update simulation map dict %s' % str(sim_map.keys()))
        except AttributeError:
            self._sim_map = sim_map
            log.debug('set simulation map dict %s' % str(sim_map.keys()))

    """
    the main routine for running hammurabiX executable
    """
    def __call__(self, verbose=False):
        # create new temp parameter file
        if self.temp_file is self._base_file:
            self._new_xml_copy()
        # if need verbose output
        if verbose is True:
            logfile = open('hammurabiX_run.log', 'w')
            errfile = open('hammurabiX_err.log', 'w')
            temp_process = subprocess.Popen([self._executable, self._temp_file],
                                            stdout=logfile,
                                            stderr=errfile)
            temp_process.wait()
            logfile.close()
            errfile.close()
        # if quiet, only print upon error
        else:
            temp_process = subprocess.Popen([self._executable, self._temp_file],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
            temp_process.wait()
            if temp_process.returncode != 0:
                last_call_log, last_call_err = temp_process.communicate()
                print(last_call_log)
                print(last_call_err)
        # grab output maps and delete temp files
        self._get_sims()
        self._del_xml_copy()

    """
    make a temporary parameter file copy and rename output file with random mark
    """
    def _new_xml_copy(self):
        # create a random file name which doesn't exist currently
        fd, new_path = tf.mkstemp(prefix='params_', suffix='.xml', dir=self._wk_dir)
        os.close(fd)
        rnd_idx = new_path[new_path.index('params_')+7:-4]
        self.temp_file = new_path
        # copy base_file to temp_file
        root = self.tree.getroot()
        # count number of sync output files
        if root.find("./observable/sync[@cue='1']") is not None:
            self._do_sync = True
            # for each sync output
            for sync in root.findall("./observable/sync[@cue='1']"):
                freq = str(sync.get('freq'))
                nside = str(sync.get('nside'))
                self.sim_map_name[('sync', freq, nside)] = os.path.join(self.wk_dir,
                                                                        'sync_'+freq+'_'+nside+'_'+rnd_idx+'.fits')
                sync.set('filename', self.sim_map_name[('sync', freq, nside)])
        fd = root.find("./observable/faraday[@cue='1']")
        if fd is not None:
            self._do_fd = True
            nside = str(fd.get('nside'))
            self.sim_map_name[('fd', 'nan', nside)] = os.path.join(self.wk_dir, 'fd_'+nside+'_'+rnd_idx+'.fits')
            fd.set('filename', self.sim_map_name[('fd', 'nan', nside)])
        dm = root.find("./observable/dm[@cue='1']")
        if dm is not None:
            self._do_dm = True
            nside = str(dm.get('nside'))
            self.sim_map_name[('dm', 'nan', nside)] = os.path.join(self._wk_dir, 'dm_'+nside+'_'+rnd_idx+'.fits')
            dm.set('filename', self.sim_map_name[('dm', 'nan', nside)])
        # automatically create a new file
        self.tree.write(self.temp_file)

    """
    grab simulation output from disk and delete corresponding fits file
    """
    def _get_sims(self):
        # locate the keys
        sync_key = list()
        dm_key = None
        fd_key = None
        for k in self.sim_map_name.keys():
            if k[0] == 'dm':
                dm_key = k
            elif k[0] == 'fd':
                fd_key = k
            elif k[0] == 'sync':
                sync_key.append(k)
            else:
                raise ValueError('mismatched key %s' % str(k))
        # read dispersion measure and delete file
        if self._do_dm is True:
            if os.path.isfile(self.sim_map_name[dm_key]):
                [DM] = self._read_fits_file(self.sim_map_name[dm_key])
                self.sim_map[(dm_key[0], dm_key[1], dm_key[2], 'nan')] = DM
                os.remove(self.sim_map_name[dm_key])
            else:
                raise ValueError('missing %s' % str(self.sim_map_name[dm_key]))
        # read faraday depth and delete file
        if self._do_fd is True:
            if os.path.isfile(self.sim_map_name[fd_key]):
                [Fd] = self._read_fits_file(self.sim_map_name[fd_key])
                self.sim_map[(fd_key[0], fd_key[1], fd_key[2], 'nan')] = Fd
                os.remove(self.sim_map_name[fd_key])
            else:
                raise ValueError('missing %s' % str(self.sim_map_name[fd_key]))
        # read synchrotron pol. and delete file
        if self._do_sync is True:
            for i in sync_key:
                # if file exists
                if os.path.isfile(self.sim_map_name[i]):
                    [Is, Qs, Us] = self._read_fits_file(self.sim_map_name[i])
                    self.sim_map[(i[0], i[1], i[2], 'I')] = Is
                    self.sim_map[(i[0], i[1], i[2], 'Q')] = Qs
                    self.sim_map[(i[0], i[1], i[2], 'U')] = Us
                    # polarisation intensity
                    self.sim_map[(i[0], i[1], i[2], 'PI')] = np.sqrt(np.square(Qs) + np.square(Us))
                    # polarisatioin angle, IAU convention
                    self.sim_map[(i[0], i[1], i[2], 'PA')] = np.arctan2(Us, Qs)/2.0
                    os.remove(self.sim_map_name[i])
                else:
                    raise ValueError('missing %s' % str(self.sim_map_name[i]))
    """
    read a single fits file with healpy
    """
    def _read_fits_file(self, path):
        rslt = []
        i = 0
        while True:
            try:
                loaded_map = hp.read_map(path, verbose=False, field=i)
                rslt += [loaded_map]
                i += 1
            except IndexError:
                break
        return rslt

    """
    delete temporary parameter file copy
    """
    def _del_xml_copy(self):
        if self.temp_file is self._base_file:
            raise ValueError('read only')
        else:
            os.remove(self._temp_file)
            self._temp_file = self._base_file

    """
    THE FOLLOWING FUNCTIONS ARE RELATED TO XML FILE MANIPULATION
    """

    """
    modify parameter in self.tree
    argument of type ['path','to','target'], {attrib}
    attrib of type {'tag': 'content'}
    if attribute 'tag' already exists, then new attrib will be assigned
    if attribute 'tag' is not found, then new 'tag' will be inserted
    """
    def mod_par(self, keychain=None, attrib=None):
        # input type check
        if type(attrib) is not dict or type(keychain) is not list:
            raise ValueError('wrong input %s %s' % (keychain, attrib))
        root = self.tree.getroot()
        path_str = '.'
        for key in keychain:
            path_str += '/' + key
        target = root.find(path_str)
        if target is None:
            raise ValueError('wrong path %s' % path_str)
        for i in attrib:
            target.set(i, attrib.get(i))

    """
    add new subkey under keychain in the tree
    argument of type ['path','to','target'], 'subkey', {attrib}
    or of type ['path','to','target'], 'subkey'
    """
    def add_par(self, keychain=None, subkey=None, attrib=None):
        # input type check
        if type(keychain) is not list or type(subkey) is not str:
            raise ValueError('wrong input %s %s %s' % (keychain, subkey, attrib))
        if attrib is not None and type(attrib) is dict:
            root = self.tree.getroot()
            path_str = '.'
            for key in keychain:
                path_str += '/' + key
            target = root.find(path_str)
            et.SubElement(target, subkey, attrib)
        elif attrib is None:
            root = self.tree.getroot()
            path_str = '.'
            for key in keychain:
                path_str += '/' + key
            target = root.find(path_str)
            et.SubElement(target, subkey)
        else:
            raise ValueError('wrong input %s %s %s' % (keychain, subkey, attrib))

    """        
    print a certain parameter
    argument of type ['path','to','key'] (e.g. ['grid','observer','x'])
    print all parameters down to the keychain children level
    """
    def print_par(self, keychain=None):
        # input type check
        if type(keychain) is not list:
            raise ValueError('wrong input %s' % keychain)
        root = self.tree.getroot()
        # print top parameter level if no input is given
        if keychain is None:
            for child in root:
                print(child.tag, child.attrib)
        else:
            path_str = '.' 
            for key in keychain:
                path_str += '/' + key
            for target in root.findall(path_str):
                print(target.tag, target.attrib)
                for child in target:
                    print('|--> ', child.tag, child.attrib)

    """        
    deletes an parameter and all of its children
    argument of type ['keys','to','target'] (e.g. ['grid','observer','x'])
    if opt='all', delete all parameters that match given keychain
    """
    def del_par(self, keychain=None, opt=None):
        # input type check
        if type(keychain) is not list:
            raise ValueError('wrong input %s' % keychain)
        root = self.tree.getroot()
        if keychain is not None:
            path_str = '.'
            par_path_str = '.'
            n = 1
            for key in keychain:
                path_str += '/' + key
                n += 1
                if n is len(keychain):
                    par_path_str = path_str
            target = root.find(path_str)
            parent = root.find(par_path_str)
            if target is None or parent is None:
                raise ValueError('wrong path %s' % path_str)
            if opt is None:
                parent.remove(target)
            elif opt is 'all':
                for i in root.findall(path_str):
                    parent.remove(i)
            else:
                raise ValueError('unsupported option at %s' % keychain)
        else:
            raise ValueError('empty keychain')
