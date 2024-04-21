# -*- encoding: utf-8 -*-
# Date: 20/Mar/2023
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""""""""""""""""""""""""""""""""""""""""""""""""""""
Description: logging module, logging to file
"""""""""""""""""""""""""""""""""""""""""""""""""""""
import logging

# levels
# CRITICAL 50
# ERROR 40
# WARNING 30
# INFO 20
# DEBUG 10
# NOTSET 0

# create and configure logger
LOG_FORMAT = "%(asctime)s,%(msecs)d %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s"
#LOG_FORMAT="%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=r'./test.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='w')

logger = logging.getLogger(__name__)


def main():
    # test the logger
    logger.info('hello')
    logger.debug('here should be 1')
    logger.warning('warning here')
    logger.error('here 0 error')
    print(logger.level, __name__)


if __name__ == "__main__":
    main()
