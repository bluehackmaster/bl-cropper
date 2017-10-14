import os
import shutil

from .Config import get_value


def check_and_create_directory(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)


def delete_pod():
    try:
        name_space = get_value('KUBERNETES', 'NAMESPACE')
        pod_name = os.environ['HOSTNAME']

        print('KUBERNETES::' + name_space)
        print('HOSTNAME::' + pod_name)
        print(os.system('chmod 777 ./bin/kubectl'))
        print(os.system('kubectl --namespace=' + name_space + ' delete pod ' + pod_name))
    except Exception as ex:
        print('Failed to delete pod')
        print(ex)
