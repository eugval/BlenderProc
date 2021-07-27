import os
import fcntl

class Locker:
    def __init__(self, lockfile_path = None):

        if(lockfile_path is None):
            current_path = os.path.dirname(os.path.realpath(__file__))
            lockfile_path = os.path.join(current_path,'../../lockfile/lockfile.lck')

        if(os.path.isdir(lockfile_path)):
            lockfile_path = os.path.join(lockfile_path, 'lockfile.lck')

        if(not os.path.exists(lockfile_path)):
            f = open(lockfile_path,'a+')
            f.close()

        self.lockfile_path = lockfile_path


    def __enter__ (self):
        self.fp = open(self.lockfile_path)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


if __name__== '__main__':
    import time
    with Locker():
        while True:
            print('locking bitch')
            time.sleep(1.)