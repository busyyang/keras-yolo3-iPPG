from glob import glob
import json

cls_path = 'model_data/pulse_classes.txt'
with open(cls_path, 'r') as f:
    cls = f.readlines()
cls = dict([(c.strip(), i) for i, c in enumerate(cls)])

ann_path = r'data\wrist_images\*.json'
anns = glob(ann_path)
for ann in anns:
    with open(ann, 'r') as f:
        tmp = json.load(f)
        line = ''
        for shape in tmp['shapes']:
            c = cls[shape['label']]
            point = shape['points']
            xmin = min(point[0][0], point[1][0])
            xmax = max(point[0][0], point[1][0])
            ymin = min(point[0][1], point[1][1])
            ymax = max(point[0][1], point[1][1])
            line += '{},{},{},{},{} '.format(int(xmin), int(ymin), int(xmax), int(ymax), c)
    with open('data/pulse_training_list.txt', 'a') as ptl:
        ptl.writelines('{} {}\n'.format(ann[:-5] + '.jpg', line))
