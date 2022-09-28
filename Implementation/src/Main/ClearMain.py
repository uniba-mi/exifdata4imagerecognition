import os
import shutil
import sys

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) < 1:
        print("please add the path to the directory to delete as an argument")
    else:
        directory = args[0]
        
        onlyDirectories = False
        if len(args) == 2:
            onlyDirectories = True

        if os.path.exists(directory) and os.path.isdir(directory):
            for fileName in os.listdir(directory):
                filePath = os.path.join(directory, fileName)
                try:
                    if os.path.isfile(filePath) or os.path.islink(filePath):
                        if not onlyDirectories:
                            os.unlink(filePath)
                    elif os.path.isdir(filePath):
                        shutil.rmtree(filePath)
                except:
                    print("error while deleting: " + directory)
        else:
            print("error: the given path does not exist or does not point to a directory")