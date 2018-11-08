'''
 Created by 
 Chenghuan Liu (Github: Fungungun) 
 8/11/18
'''
import logging


# TODO: Fix the resize problem in detector and also the rule used here.

class Counter():
    def __init__(self, site, count_group):
        self.site = site
        self.count_res = [0] * count_group

    def process(self, tracklets):
        '''
        This function returns the current counting results
        '''
        if self.site == '5A':
            self.process_5a(tracklets)
        elif self.site == '4B':
            self.process_4b(tracklets)
        else:
            raise NotImplementedError

        return self.count_res

    def process_5a(self, tracklets):
        # ---------------------------------------------------#
        # Process each tracklet to get which group it belongs to
        # Here we hardcode some rules for counting by lanes
        for tracklet in tracklets:
            if not tracklet.count_flag:
                if tracklet.trajectory[-1][1] >= 330:
                    tracklet.current_region = 'B'
                    if tracklet.trajectory[-1][0] < 250:
                        tracklet.vehicle_lane = 0
                    elif 250 <= tracklet.trajectory[-1][0] < 430:
                        tracklet.vehicle_lane = 1
                    else:
                        tracklet.vehicle_lane = 2
            else:
                tracklet.time_since_counted += 1
        # ---------------------------------------------------#

        # Counting results update
        for tracklet in tracklets:
            if tracklet.current_region == 'B' and not tracklet.count_flag:
                tracklet.count_flag = True
                self.count_res[tracklet.vehicle_lane] += 1

    def process_4b(self, tracklets):
        # ---------------------------------------------------#
        '''
        Divide the whole frame into 4 regions
        A -> C: Group 1
        B -> C: Group 2
        D -> D: Group 3
                    |
             A      |   B
                    |
        -------------------
             C      |   D
        '''
        for tracklet in tracklets:
            if not tracklet.count_flag:
                tracklet.previous_region = tracklet.current_region
                if (tracklet.bbox[-1][0] + tracklet.bbox[-1][2]) / 2 < 480:  # (x1+x2)/2 < 480
                    if tracklet.bbox[-1][3] < 330:
                        tracklet.current_region = 'A'
                    else:
                        tracklet.current_region = 'C'
                else:
                    if tracklet.bbox[-1][3] < 330:
                        tracklet.current_region = 'B'
                    else:
                        tracklet.current_region = 'D'
            else:
                tracklet.time_since_counted += 1
        # ---------------------------------------------------#

        # Counting results update
        for tracklet in tracklets:
            if (tracklet.previous_region is not None) and \
                    (tracklet.current_region is not None) and \
                    (not tracklet.count_flag):
                if tracklet.previous_region == 'A' and tracklet.current_region == 'C':
                    self.count_res[0] += 1
                    tracklet.count_flag = True
                elif tracklet.previous_region == 'B' and tracklet.current_region == 'C':
                    self.count_res[1] += 1
                    tracklet.count_flag = True
                elif (tracklet.previous_region == 'D' and tracklet.current_region == 'C') or \
                    (tracklet.previous_region == 'D' and tracklet.current_region == 'D'):
                    self.count_res[2] += 1
                    tracklet.count_flag = True
