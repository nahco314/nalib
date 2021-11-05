import shutil
import zipfile
import os


def one_filanize(path, name):
    shutil.make_archive('./local_zip', 'zip', root_dir=path)

    with open('./local_zip.zip', 'rb') as f1:
        res = f"""
import zipfile
import os
import sys


if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("./zip.zip", "bw") as f:
        f.write({f1.read()})
    
    with zipfile.ZipFile('./zip.zip') as existing_zip:
        existing_zip.extractall('./{name}')
        """

    os.remove("local_zip.zip")

    return res


if __name__ == '__main__':
    os.remove("res.txt")

    with open("res.txt", "w") as f:
        print(one_filanize("lib_path", "lib_name"), file=f)
