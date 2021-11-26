import base64
import shutil
import zipfile
import os


def one_filanize(path, name):
    shutil.make_archive('./local_xz', 'xztar', root_dir=path)

    with open('./local_xz.tar.xz', 'rb') as f1:
        res = f"""
import shutil
import os
import sys
import base64


if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    os.makedirs("{name}", exist_ok=True)
    os.chdir("{name}")

    with open("xz.tar.xz", "bw") as f:
        f.write(base64.b85decode("{base64.b85encode(f1.read()).decode()}"))

    shutil.unpack_archive("xz.tar.xz")
    os.chdir("../")
        """

    os.remove("local_xz.tar.xz")

    return res


if __name__ == '__main__':
    os.remove("res.txt")

    with open("res.txt", "w") as f:
        print(one_filanize("lib_path", "lib_name"), file=f)
