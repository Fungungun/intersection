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
from counter import Counter

import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='faster_rcnn_resnet101_v1d_coco', type=str, help=
'Some choices: faster_rcnn_resnet101_v1d_coco, yolo3_darknet53_coco, ssd_512_mobilenet1.0_coco')
parser.add_argument('--site', default='4B', type=str, choices={'4B', '5A'}, help='Nmae of the site.')
parser.add_argument('--display', action='store_true', default=False, help='Display the processed video')
parser.add_argument('--write', action='store_true', default=False, help='Write the processed video to file')
args = parser.parse_args()

logging.basicConfig(handlers=[logging.FileHandler('log_' + args.site), logging.StreamHandler()],
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logging.info('--------------------------New Session--------------------------')

main_config = configparser.ConfigParser()
main_config.read('path.cfg')
video_dir_path = main_config['path']['video_dir']

if args.site == '5A':
    vid_name = '5A_0751_0822.wmv'
elif args.site == '4B':
    vid_name = '4B_1648_1718.wmv'
else:
    raise NotImplementedError

logging.info(vid_name)
logging.info(args.model)

cap = cv2.VideoCapture(os.path.join(video_dir_path, vid_name))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
# v_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# v_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = skvideo.io.FFmpegWriter(vid_name.split('.')[0] + '_output.mp4')

detector = Detector(model=args.model)
mot_tracker = MOTTracker(max_age=100, min_hits=1,
                         time_since_counted_threshold=99)
# Tis count_group should be read from cfg file for  each site
counter = Counter(site=args.site, count_group=3)

display_flag = args.display
write_flag = args.write

line_width = 2
draw_colour = (255, 0, 0)
font_size = 1

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

    lane_count = counter.process(tracklets)


    # All the display settings need to be read from cfg file for each site
    # Here they are hardcoded fro site 5A and 4B
    # Visualization on image
    for tracklet in tracklets:
        x1, y1, x2, y2 = tracklet.bbox[-1]
        cv2.rectangle(show_img, (x1, y1), (x2, y2), draw_colour, thickness=2)
        # for box in tracklet.bbox:
        #     cv2.circle(show_img, (int((box[0]+box[2])/2), box[3]),
        #                radius=1, color=draw_colour, thickness=-1)

    cv2.putText(show_img,
                'Count:  ' + str(lane_count[0]) + '      ' + str(lane_count[1]) + '      ' + str(lane_count[2]),
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, draw_colour, thickness=line_width)

    cv2.putText(show_img, 'Group:   1      2      3',
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
    if num_frame % (vid_fps * 60) == 0:
        logging.info(str(round(1.0 * num_frame / total_frames, 2)) + '%')
        logging.info('Count: ' + str(lane_count[0]) + ' ' + str(lane_count[1]) + ' ' + str(lane_count[2]))

logging.info('Video Finished!')
logging.info('Count: ' + str(lane_count[0]) + ' ' + str(lane_count[1]) + ' ' + str(lane_count[2]))

cap.release()
