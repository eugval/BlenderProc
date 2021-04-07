import os

import bpy
import h5py
import numpy as np

from src.main.GlobalStorage import GlobalStorage
from src.writer.WriterInterface import WriterInterface
from src.utility.Utility import Utility


class FolderStructureWriter(WriterInterface):
    """ Make a folder structure.

    **Configuration**:

    .. list-table::
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type

    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)

    def run(self):

      output_dir = self.config.get_string("output_dir")
      run_number = self.config.get_int('run_number',0)

      main_path = os.path.join(output_dir,'{}'.format(run_number))

      if(not os.path.exists(main_path)):
          os.makedirs(main_path)

          os.makedirs(os.path.join(main_path,'rgb'))
          os.makedirs(os.path.join(main_path,'cam_poses'))
          os.makedirs(os.path.join(main_path, 'correspondances'))
          os.makedirs(os.path.join(main_path, 'corrspondance_semantic_maps'))
          os.makedirs(os.path.join(main_path, 'depth_images'))
          os.makedirs(os.path.join(main_path, 'segmentation_masks'))


