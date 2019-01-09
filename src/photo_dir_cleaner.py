import os
from PIL import Image
import numpy as np
from numpy.random import choice
from shutil import move 


class ImageDirectoryCleaner():
    """
    This object will clean the directory containing the "Image Label" folders and their
    respective images. It starts by removing any URL variables ("?=...") and then attempts
    to open the images to verify their compatibility with Tensorflow training. If they don't
    open, they will be deleted.

    Inputs:
    top_directory = (string) The path to the top directory that contains the "Image Label" subdirectories.
    sub_directory = (string) (optional) A path to a single directory containing images (Overrides top_directory scan)
    
    Output: Subdirectories containing only images with testid filenames ('.jpeg, .jpg, .png, etc')
    """

        
    def __init__(self, top_directory=None, sub_directory=None):
        self.top_directory = top_directory
        self.sub_directory = sub_directory
    
    def clean(self):
        if self.sub_directory:
            self._remove_url_var(self.sub_directory)
            self._delete_non_images(self.sub_directory)
        elif self.top_directory:
            subdir_list_ = [d for d in os.listdir(self.top_directory)]
            for subdir in subdir_list_:
                path = '{}/{}'.format(self.top_directory,subdir)
                self._remove_url_var(path)
                self._delete_non_images(path)

    def _remove_url_var(self, file_path):
        file_list = [f for f in os.listdir(file_path)]
        for f in file_list:
            if '?' in f:
                old_path = "{}/{}".format(file_path,f)
                new_name = f[:f.find("?")]
                new_path = "{}/{}".format(file_path, new_name)
                os.rename(old_path, new_path)
                continue
            if '&' in f:
                old_path = "{}/{}".format(file_path,f)
                new_name = f[:f.find("&")]
                new_path = "{}/{}".format(file_path, new_name)
                os.rename(old_path, new_path)
                continue


    def _delete_non_images(self, file_path):
        file_list = [f for f in os.listdir(file_path)]
        for f in file_list:
            try:
                image = Image.open('{}/{}'.format(file_path, f))
                image.load()
                image.close()
            except:
                try:
                    os.remove('{}/{}'.format(file_path,f))
                except:
                    continue



class TrainTestSplit():
    def __init__(self,
                 base_path=None,
                 train_dir='train',
                 test_dir='test',
                 test_split=0.05):
        self.base_path = base_path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_path = os.path.join(self.base_path, self.train_dir)
        self.test_path = os.path.join(self.base_path, self.test_dir)
        self.test_split = test_split

    def create_split(self):
        subdir_list_ = [d for d in os.listdir(self.train_path)]
        for subdir in subdir_list_:
            self.make_test_dir(subdir)
            self.parse_image_folder(subdir)
    
    def make_test_dir(self, subdir):
        test_folder = os.path.join(self.test_path, subdir)
        os.mkdir(test_folder)

    def parse_image_folder(self, subdir):
        train_folder = os.path.join(self.train_path, subdir)
        file_list = [f for f in os.listdir(train_folder)]
        for file in file_list:
            move_it = choice([False, True], size=1, p=[1-self.test_split, self.test_split])[0]
            if move_it:
                self.move_file_to_dir(self.train_path, self.test_path, subdir, file)

    def move_file_to_dir(self,src_path, dest_path, subdir, filename):
        source = os.path.join(src_path, subdir, filename)
        dest = os.path.join(dest_path, subdir, filename)
        move(source, dest)

    def reverse_test_split(self):
        subdir_list_ = [d for d in os.listdir(self.test_path)]
        for subdir in subdir_list_:
            subdir_path = os.path.join(self.test_path, subdir)
            file_list = [f for f in os.listdir(subdir_path)]
            for file in file_list:
                self.move_file_to_dir(self.test_path, self.train_path, subdir, file)



if __name__ == '__main__':

    ## To Clean the directory
    # idc = ImageDirectoryCleaner(top_directory='./train')
    # idc.clean()


    ## To make the test splits
    base_path = '/home/shullback/Dropbox/fish_photos/data/'
    train_dir = 'train'
    test_dir = 'test'

    tvs = TrainTestSplit(base_path, train_dir, test_dir)
    # tvs.create_split()

