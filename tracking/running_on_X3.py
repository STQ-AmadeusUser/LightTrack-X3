import _init_paths
import time
import argparse
import cv2
import numpy as np
import sys
from easydict import EasyDict as edict
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
from lib.models.model_deploy import ModelDeploy
from lib.utils.deploy_helper import (load_yaml,
                                     print_properties,
                                     bgr2nv12_opencv,
                                     cxy_wh_2_rect,
                                     get_frames)
from lib.tracker.lighttrack_deploy import Lighttrack_deploy


def parse_args():
    parser = argparse.ArgumentParser(description='LightTrack Demo')
    parser.add_argument('--cfg', type=str, default='../experiments/LightTrack.yaml',
                        help='yaml configure file name')
    parser.add_argument('--inference', default='../bin/LightTrack.bin',
                        help='LightTrack algorithm in Siamese Tracking algorithms')
    parser.add_argument('--video', default='../video/ChasingDrones', type=str,
                        help='video or image files directory')
    args = parser.parse_args()
    args.init_rect = [653, 221, 55, 40]
    return args


def track(inputs):
    video_player = inputs['player']
    tracking_tracker = inputs['tracker']
    tracking_net = inputs['network']
    video_shower = inputs['shower']
    args = inputs['args']
    init_rect = args.init_rect
    start_frame, lost, boxes, toc = 0, 0, [], 0
    writer = None

    for idx, im in enumerate(video_player):
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        print("frame {} is ok!".format(idx))

        tic = cv2.getTickCount()
        if idx == start_frame:  # init
            # initialize video writer
            writer = cv2.VideoWriter('../video/' + args.video_name + '_result' + '.mp4',
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     30,
                                     (im.shape[1], im.shape[0]))
            # initialize video tracker
            cx = init_rect[0] + (init_rect[2] - 1) / 2  # center_x
            cy = init_rect[1] + (init_rect[3] - 1) / 2  # center_y
            w, h = init_rect[2], init_rect[3]
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            state = tracking_tracker.init(im, target_pos, target_sz, tracking_net)  # init tracker
            # write the first frame
            bbox = list(map(int, init_rect))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
            writer.write(im)
            # im_nv12 = bgr2nv12_opencv(im)
            # video_shower.set_img(im_nv12.tobytes())
            time.sleep(2)

        elif idx > start_frame:  # tracking
            state = tracking_tracker.track(state, im)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            bbox = list(map(int, location))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
            writer.write(im)
            # im_nv12 = bgr2nv12_opencv(im)
            # video_shower.set_img(im_nv12.tobytes())
            time.sleep(2)
        toc += cv2.getTickCount() - tic
    writer.release()
    video_shower.close()


def main():
    print('===> load argparse and configs <====')
    args = parse_args()
    configs = edict(load_yaml(args.cfg))

    print('===> load dnn models <====')
    inference = dnn.load(args.inference)
    dnn_model = {'inference': inference}
    print_properties(inference[0].inputs[0].properties)
    print_properties(inference[0].inputs[1].properties)
    print_properties(inference[0].outputs[0].properties)
    print_properties(inference[0].outputs[1].properties)

    print('===> create inference model <====')
    net = ModelDeploy(dnn_model)

    print('===> load videos or images <====')
    if args.video is not None:
        video_name = args.video.split('/')[-1].split('.')[0]
        args.video_name = video_name
    video_player = get_frames(args.video)
    # Get HDMI display object
    video_shower = srcampy.Display()
    # For the meaning of parameters, please refer to the relevant documents of HDMI display
    # video_shower.display(0, configs.DEMO.DISPLAY.width, configs.DEMO.DISPLAY.height)

    print('===> create tracking model <====')
    # build tracker
    tracker = Lighttrack_deploy(configs)

    print('===> start tracking! <====')
    inputs = {'player': video_player,
              'tracker': tracker,
              'network': net,
              'shower': video_shower,
              'args': args,
              'config': configs,
              }
    track(inputs)


if __name__ == '__main__':
    main()
