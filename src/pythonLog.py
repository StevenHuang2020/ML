"""python logging module testing
"""
import logging

# levels
# CRITICAL 50
# ERROR 40
# WARNING 30
# INFO 20
# DEBUG 10
# NOTSET 0

#create and configure logger
LOG_FORMAT="%(asctime)s,%(msecs)d %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s"
#LOG_FORMAT="%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=r'./test.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='w')

log = logging.getLogger(__name__)


#test the logger
log.info('hello')
log.debug('here should be 1')
log.warning('warning here')
log.error('here 0 error')


print(log.level, __name__)
