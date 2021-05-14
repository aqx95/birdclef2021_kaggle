import os
import logging

def log(config, name):
    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)
    log_file = config.LOG_PATH,/'log.txt'
    open(log_file, 'w+').close()

    console_log_format = "%(levelname)s %(message)s"
    file_log_format = "%(levelname)s: %(asctime)s: %(message)s"

    #Configure logger
    logging.basicConfig(level=logging.INFO, format=console_log_format)
    logger = logging.getLogger(name)

    #File handler
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)
    f_formatter = logging.Formatter(file_log_format)
    f_handler.setFormatter(f_formatter)

    #Stream handler
    # c_handler = logging.StreamHandler()
    # c_handler.setLevel(logging.INFO)
    # c_formatter = logging.Formatter(console_log_format)
    # c_handler.setFormatter(c_formatter)

    #Add handler to logger
    logger.addHandler(f_handler)
    #logger.addHandler(c_handler)

    return logger
