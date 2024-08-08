import os
import sys
import inspect
import logging

# This class simply tries to see if AirSim
class SetupPath:
    @staticmethod
    def getDirLevels(path):
        path_norm = os.path.normpath(path)
        return len(path_norm.split(os.sep))
    
    @staticmethod
    def getCurrentPath():
        cur_filepath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        return os.path.dirname(cur_filepath)
    
    @staticmethod
    def getGrandParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 2:
            return os.path.dirname(os.path.dirname(cur_path))
        return ''
    
    @staticmethod
    def getParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 1:
            return os.path.dirname(cur_path)
        return ''
    
    @staticmethod
    def addAirSimModulePath():
        # if airsim module is installed, then don't do anything
        # import pkgutil
        # airsim_loader = pkgutil.find_loader('airsim')
        # if airsim_loader is not None:
        # return

        parent = SetupPath.getParentDir()
        if parent != '':
            airsim_path = os.path.join(parent, 'airsim')
            client_path = os.path.join(airsim_path, 'client.py')
            if os.path.exists(client_path):
                sys.path.insert(0, parent)
        else:
            logging.warning("airsim module not found in parent folder. Using installed package (pip install airsim)")

SetupPath.addAirSimModulePath()