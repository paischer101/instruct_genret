import os
import zipfile
dirName = './'
import glob
files = glob.glob('**', recursive=True)
# Read all directory, subdirectories and file lists
for file in files:
    zf = zipfile.ZipFile("./genrec.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
    for name in files:
        if name.endswith('.py') or name.endswith('.yaml'): 
            zf.write(name, arcname=name)
    zf.close()