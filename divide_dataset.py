import os
import random
import sys
import shutil

photos = set(list(os.listdir("buf")))
maps = set(list(os.listdir("eje")))

#os.mkdir("img/training")
#os.mkdir("eje/training")

files = list(maps & photos)
random.shuffle(files)

n = int(sys.argv[1])

for file in files[0:int(n*0.80)]:
	if os.path.isfile("buf/"+ file):
		shutil.move("buf/"+ file, "buf/train/"+ file)
		shutil.move("eje/"+ file, "eje/train/"+ file)

#os.mkdir("img/validation")
#os.mkdir("eje/validation")

for file in files[int(n*0.8):int(n*0.9)]:
	if os.path.isfile("buf/"+ file):
		shutil.move("buf/"+ file, "buf/val/"+ file)
		shutil.move("eje/"+ file, "eje/val/"+ file)
    
#os.mkdir("img/test")
#os.mkdir("eje/test")

for file in files[int(n*0.9):n]:
	if os.path.isfile("buf/"+ file):
		shutil.move("buf/"+ file, "buf/test/"+ file)
		shutil.move("eje/"+ file, "eje/test/"+ file)