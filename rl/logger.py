# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import time
import logging
import config

FLAGS = config.flags.FLAGS


# Logger
logger = logging.getLogger('rl')
logger.setLevel(logging.INFO)
fh_agent = logging.FileHandler('./rl.log')
sh = logging.StreamHandler()
fm = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > [%(name)s] %(message)s')
fh_agent.setFormatter(fm)
sh.setFormatter(fm)
logger.addHandler(fh_agent)
logger.addHandler(sh)
logging.getLogger('rl.agent').setLevel(logging.DEBUG)
logging.getLogger('rl.env').setLevel(logging.DEBUG)


# Setting for result file
now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
file_name = FLAGS.agent + "-s-"+str(FLAGS.seed)+"-" + s_time
result = logging.getLogger('Result')
result.setLevel(logging.INFO)

if FLAGS.folder == "default":
    result_fh = logging.FileHandler("./results/eval/r-" + file_name + ".txt")
    nn_filename = "./results/nn/n-" + file_name
else:
    result_fh = logging.FileHandler("./results/eval/"+ FLAGS.folder +"/r-" + file_name + ".txt")
    nn_filename = "./results/nn/" + FLAGS.folder + "/n-" + file_name

result_fm = logging.Formatter('[%(filename)s:%(lineno)s] %(asctime)s\t%(message)s')
result_fh.setFormatter(result_fm)
result.addHandler(result_fh)
