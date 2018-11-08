import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # statistics variables
        self.current_pos = []
        self.current_region = None
        self.previous_region = None

        self.speed = 0.0
        self.trajectory = []
        self.size = []
        self.bbox = []

        self.count_flag = False  # true: already counted
        self.time_since_counted = 0
        self.vehicle_lane = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if bbox != []:
            self.kf.update(convert_bbox_to_z(bbox))

            # if len(self.history) >= 2:
            #     pre_pos_x = (self.history[-2][0][0] + self.history[-2][0][2]) / 2.0
            #     pre_pos_y = (self.history[-2][0][1] + self.history[-2][0][3]) / 2.0
            #     cur_pos_x = (self.history[-1][0][0] + self.history[-1][0][2]) / 2.0
            #     cur_pos_y = (self.history[-1][0][1] + self.history[-1][0][3]) / 2.0
            #     self.speed = np.math.sqrt((pre_pos_x - cur_pos_x) ** 2 + (pre_pos_y - cur_pos_y) ** 2) * 40

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1][0]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)[0]

    def statistics_update(self):
        state = self.get_state()
        d = state.reshape(1, -1)
        d = d[0].astype(np.int32)
        self.bbox.append(d)

        # current position of the tracklet
        self.current_pos = d

        # Add point to tracklet
        # self.trajectory.append((int((d[0] + d[2])/2), int((d[1] + d[3])/2)))
        self.trajectory.append((d[2], d[3]))



def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
