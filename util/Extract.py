import os
import shutil
import zipfile

import numpy as np
from moviepy.editor import VideoFileClip

from util import Trans, Comm


class Extract:
    def __init__(self, path, dataset_id):
        self.__path = path
        self.__img_path = os.path.join(path, 'img')
        self.__dataset_id = dataset_id
        Comm.check_and_create_directory([path, self.__img_path])
        print('Created temp folders')

    def execute(self):
        response_data = Trans.request_service('GET', 'http://api.whatsit.net/datasets/' + self.__dataset_id, [])
        print(response_data)
        if response_data['data']['data'][0] is not None:
            data_set = response_data['data']['data'][0]
            video_name = data_set['name']
            source = data_set['source']

            sections = data_set['sections']
            video_path = Trans.download_file(os.path.join(self.__path, 'temp.mp4')
                                             , source)

            video = VideoFileClip(filename=video_path)
            zip = zipfile.ZipFile(os.path.join(self.__path, 'temp.zip'), 'w')
            image_files = []
            print('[Extracted image file from video]')
            for i in sections:
                for k in np.arange(i[0], i[1] + 0.2, 0.2):
                    image_inform = {'name': str(k) + '.jpg'}
                    image_files.append(image_inform)
                    image_file_path = os.path.join(self.__img_path, str(k) + '.jpg')
                    video.save_frame(filename=image_file_path, t=k)
                    zip.write(image_file_path, os.path.relpath(image_file_path, self.__img_path),
                              compress_type=zipfile.ZIP_DEFLATED)
                    print(image_file_path)

            zip.close()
            print('\n\nCreated zip file::' + zip.filename)
            file_url = Trans.upload_file_to_bucket('whatsit-dataset-video', zip.filename,
                                                   key=self.__dataset_id + '/' + video_name + '.zip', is_public=True)
            print('image_files' + image_files.__str__())
            print(file_url)
            print('Deleted temp directory::')
            shutil.rmtree(self.__path)
            params = {
                "type": "video",
                "data": [{
                    "frames": file_url
                    , "images": image_files
                }]
            }

            print(params)
            print('Requested::')
            print(Trans.request_service('PUT', 'http://api.whatsit.net/datasets/' + self.__dataset_id, params))
        else:
            print('no existing dataset ')
