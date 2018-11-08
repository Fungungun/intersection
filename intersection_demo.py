'''
 Created by 
 Chenghuan Liu (Github: Fungungun) 
 3/11/18
'''
import configparser
import cv2
import skvideo.io
import argparse
import logging

from detector import Detector
from mot_tracker import MOTTracker

logging.basicConfig(filename='log_intersection', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='a')
logging.info('--------------------------New Session--------------------------')

import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolo3_darknet53_coco', type=str, help=
                    'Some choices: faster_rcnn_resnet101_v1d_coco, yolo3_darknet53_coco, ssd_512_mobilenet1.0_coco')
parser.add_argument('--display', action='store_true', default=False)
parser.add_argument('--write', action='store_true', default=False)
args = parser.parse_args()

main_config = configparser.ConfigParser()
main_config.read('../MOT/cfg/main.cfg')
video_dir_path = main_config['path']['video_dir']

vid_name = '5A_0751_0822.wmv'


logging.info(vid_name)
logging.info(args.model)


cap = cv2.VideoCapture(os.path.join(video_dir_path, vid_name))
vid_fps = cap.get(cv2.CAP_PROP_FPS)

writer = skvideo.io.FFmpegWriter(vid_name.split('.')[0] + '_output.mp4')

detector = Detector(model=args.model)
mot_tracker = MOTTracker(max_age=100, min_hits=1,
                         time_since_counted_threshold=99)

display_flag = args.display
write_flag = args.write

line_width = 2
draw_colour = (255, 0, 0)
font_size = 1

lane_count = [0, 0, 0]

num_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Get detection
        class_id, bounding_box, show_img = detector.process(frame)
    else:
        break
    # MOT update
    ret, tracklets = mot_tracker.update(bounding_box, class_id)

    # Counting update
    for tracklet in tracklets:
        if tracklet.current_region == 'B' and not tracklet.count_flag:
            tracklet.count_flag = True
            lane_count[tracklet.vehicle_lane] += 1

    # Visualization on image
    for tracklet in tracklets:
        x1, y1, x2, y2 = tracklet.bbox[-1]
        cv2.rectangle(show_img, (x1, y1), (x2, y2), draw_colour, thickness=2)

    cv2.putText(show_img, 'Count:  ' + str(lane_count[0]) + '      ' + str(lane_count[1]) + '      ' + str(lane_count[2]),
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, draw_colour, thickness=line_width)

    cv2.putText(show_img, 'Lane:   1      2      3',
                (50, 40), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, draw_colour, thickness=line_width)

    # Show in window
    if display_flag:
        cv2.imshow('Test', show_img)
        if cv2.waitKey(1) == 113:
            exit()

    if write_flag:
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        writer.writeFrame(show_img)

    num_frame += 1
    if num_frame % (vid_fps*60) == 0:
        logging.info('Count: ' + str(lane_count[0]) + ' ' + str(lane_count[1]) + ' ' + str(lane_count[2]))

logging.info('Video Finished!')
logging.info('Count: ' + str(lane_count[0]) + ' ' + str(lane_count[1]) + ' ' + str(lane_count[2]))

cap.release()
