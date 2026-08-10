"""Composite camera pose combining multiple pose sources."""
from utils.registry import CAMPOSE_REGISTRY


@CAMPOSE_REGISTRY.register()
class CompositePose:
    def __init__(self, conf):
        self.conf = conf
        self.pose_list = []
        for sub_conf in conf['list']:
            clas = sub_conf['type']
            try:
                cam_cls = CAMPOSE_REGISTRY.get(clas)
            except KeyError:
                raise ValueError(f'Unknown CamPose type: {clas}')
            pose_generator = cam_cls(sub_conf)
            self.pose_list += pose_generator.pose_list
        self.img_num = len(self.pose_list)
