import logging
import os
from copy import deepcopy
from time import strftime, localtime


class Logger:
    def __init__(self, ID=None, to_stdout=False, path=None):
        file_name = strftime("%b-%d-%Y_%H:%M:%S", localtime())
        self.training_start_time = deepcopy(file_name)
        if ID:
            file_name += "_" + ID + ".log"
        else:
            file_name += ".log"

        if path:
            if path not in os.listdir(os.getcwd()):
                os.makedirs(path)
            file_name = os.getcwd() + "/" + path + "/" + file_name

        logging.basicConfig(filename=file_name, level=logging.INFO)

        if to_stdout:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logging.getLogger().addHandler(ch)

        self.logger = logging.getLogger()

    def log(self, content, name=" "):
        self.logger.info("%s -> %s: %s" % (strftime("%b-%d-%Y %H:%M:%S", localtime()), name, content))

    def get_training_start_time(self):
        return self.training_start_time
