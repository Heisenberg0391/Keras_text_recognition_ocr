# coding = utf-8
import os


if __name__ == '__main__':
    # 此时位于main.py根目录
    # os.system('conda activate keras')
    # os.system('python hdf5/2_noisy_synthetic.py')
    os.system('python hdf5/2_build_train_hdf5.py')
    os.system('python 4_train_hdf5.py')
    # os.system('python 5_train_hdf5_mine.py')


