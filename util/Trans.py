import json

import boto3
import requests

from . import Config

AWS_ACCESS_KEY = Config.get_value('AWS', 'AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = Config.get_value('AWS', 'AWS_SECRET_ACCESS_KEY')


def download_file(save_path, file_url):
    """ Download file from http url link """

    r = requests.get(file_url)  # create HTTP response object

    with open(save_path, 'wb') as f:
        f.write(r.content)

    return save_path


def download_file_from_dropbox(save_path, link):
    """ Download file from dropbox
    It is need to get a app access token"""

    dbx = dropbox.Dropbox("Access Token")
    print(dbx.users_get_current_account())

    return dbx.sharing_get_shared_link_file(url=link, path=save_path)


def upload_file_to_bucket(bucket, file_path, key, is_public=False):
    """ Upload files to S3 Bucket """
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    with open(file_path, 'rb') as data:
        s3.upload_fileobj(data, bucket, key)

    if is_public:
        s3.put_object_acl(ACL='public-read', Bucket=bucket, Key=key)

    file_url = '%s/%s/%s' % ('https://s3.ap-northeast-2.amazonaws.com', bucket, key)
    return file_url


def request_service(method, url, params):
    """ Request to whatsit API server"""
    if method == 'PUT':
        resp = requests.put(url=url, json=params, timeout=60)
    elif method == 'POST':
        resp = requests.post(url=url, params=params, timeout=60)
    else:
        resp = requests.get(url=url, params=params, timeout=60)
    print(resp.status_code)
    return json.loads(resp.text)
