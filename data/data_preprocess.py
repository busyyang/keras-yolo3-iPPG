import os

labels = os.listdir('labels')
labels = [l.split('.')[0] for l in labels]

images = os.listdir('images')
images = [i.split('.')[0] for i in images]
f = open('training_list.txt', 'w')
for i in images:
    if i not in labels:
        os.remove('images/{}.jpg'.format(i))
    else:
        f.writelines('D:/busy/Desktop/keras-yolo3/data/images/{}.jpg '.format(i))
        with open('labels/{}.txt'.format(i), 'r') as lab:
            ll = lab.readlines()
        for l in ll:
            l = l.split()
            f.writelines('{},{},{},{},{} '.format(l[1], l[2], l[3], l[4], l[0]))
        f.write('\n')
f.close()
