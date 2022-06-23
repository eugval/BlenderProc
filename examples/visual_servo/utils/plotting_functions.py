import matplotlib.pyplot as plt
import numpy as np
from .. import se3_tools as se3
import os

class PlotVisualServo_errors(object):
    def __init__(self):
        self.camera_pos_errors = []
        self.camera_pos_errors_y = []
        self.camera_pos_errors_x = []
        self.camera_pos_errors_z = []
        self.rot_errors = []
        self.average_flows_y = []
        self.average_flows_x = []
        self.average_flows = []

    def plot_errors(self, bottleneck_pose, camera_pose, flattened_correspondences, starting_pixels, save_dir):

        delt = bottleneck_pose[:3, 3] - camera_pose[:3, 3]
        camera_pos_error = np.linalg.norm(delt)
        camera_pos_error_x = np.abs(delt[0])
        camera_pos_error_y = np.abs(delt[1])
        camera_pos_error_z = np.abs(delt[2])
        self.camera_pos_errors.append(camera_pos_error)
        self.camera_pos_errors_y.append(camera_pos_error_y)
        self.camera_pos_errors_x.append(camera_pos_error_x)
        self.camera_pos_errors_z.append(camera_pos_error_z)
        plt.plot(self.camera_pos_errors, label='norm')
        plt.plot(self.camera_pos_errors_x, label='x')
        plt.plot(self.camera_pos_errors_y, label='y')
        plt.plot(self.camera_pos_errors_z, label='z')
        plt.legend()
        plt.ylim((np.min(self.camera_pos_errors + self.camera_pos_errors_x + self.camera_pos_errors_y + self.camera_pos_errors_z),
                  np.max(self.camera_pos_errors + self.camera_pos_errors_x + self.camera_pos_errors_y + self.camera_pos_errors_z)))
        plt.savefig(save_dir+ '/plot_position_error.png')
        plt.close()

        camera_rot_delta = bottleneck_pose[:3, :3] @ camera_pose[:3, :3].T
        camera_rotvec_delta = se3.rot2rotvec(camera_rot_delta)
        rot_error = np.linalg.norm(camera_rotvec_delta)
        self.rot_errors.append(rot_error)
        plt.plot(self.rot_errors)
        plt.ylim((np.min(self.rot_errors), np.max(self.rot_errors)))
        plt.savefig(save_dir+ '/plot_rotation_error.png')
        plt.close()


        f = flattened_correspondences - starting_pixels
        self.average_flows_y.append(np.mean(f[:, 0]))
        self.average_flows_x.append(np.mean(f[:, 1]))
        self.average_flows.append(np.mean(np.linalg.norm(f, axis=1)))
        plt.plot(self.average_flows_y, label='y')
        plt.plot(self.average_flows_x, label='x')
        plt.plot(self.average_flows, label='xy_norm')
        plt.ylim((np.min(self.average_flows_y + self.average_flows_x + self.average_flows),
                  np.max(self.average_flows_y + self.average_flows_x + self.average_flows)))
        plt.legend()
        plt.savefig(save_dir + '/plot_flow.png')
        plt.close()


    def save_image(self, base_dir, dir_name, num_frame, image):
        if(not os.path.exists(base_dir + '/{}'.format(dir_name))):
            os.makedirs(base_dir + '/{}'.format(dir_name))

        plt.imsave(base_dir+ '/{}/{}.png'.format(dir_name, num_frame), image)