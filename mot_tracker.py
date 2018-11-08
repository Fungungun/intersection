

from __future__ import print_function
import numpy as np
from tracklet import KalmanBoxTracker
from data_association import associate_detections_to_trackers


class MOTTracker:
    def __init__(self, max_age=1, min_hits=3, time_since_counted_threshold=30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.time_since_counted_threshold = time_since_counted_threshold

        self.trajectory_database = []

    def update(self, dets, obj_classes):
        """
        Params:
        dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            # print(pos)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                print('isnan')
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            print(t)
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, iou_threshold=0.25)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    trk.tracklet_class = int(obj_classes[d])

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.tracklet_class = int(obj_classes[i])
                self.trackers.append(trk)

        # update the variables used for MR tasks
        for trk in reversed(self.trackers):
            trk.statistics_update()

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])

            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                ret.append(trk)
            i -= 1
            # remove dead tracklet and counted dead tracklet
            if (trk.time_since_update > self.max_age) or (trk.count_flag and trk.time_since_counted > self.time_since_counted_threshold):
                self.trajectory_database.append(trk.trajectory)
                self.trackers.pop(i)

        if len(ret) > 0:
            return True, ret
        else:
            return False, ret
