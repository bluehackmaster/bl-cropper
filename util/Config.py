import configparser

config = configparser.ConfigParser()
config.read('./config/config.ini')


def get_value(section, value):
    return config[section][value]
