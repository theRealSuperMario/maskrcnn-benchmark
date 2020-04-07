import pytest
import os


from python_pascal_voc import voc_utils, pascal_part
from python_pascal_voc import pascal_part_annotation
from matplotlib import pyplot as plt
import numpy as np

from maskrcnn_benchmark.config import paths_catalog
from maskrcnn_benchmark.data.datasets.voc import VOCPartsCropped


DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


class Test_PascalPartCropped:
    def test_bounding_boxes(self, tmpdir):
        args = paths_catalog.DatasetCatalog.get("pascal_part_2010_train")["args"]
        import pudb

        pudb.set_trace()
        csv_dir = tmpdir.mkdir("csv")
        args["dir_cropped_csv"] = csv_dir
        dset = VOCPartsCropped(**args)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        for i in range(8):
            ax = axes[i]
            image, boxlist, idx = dset[i]

            overlay = voc_utils.overlay_boxes_without_labels(
                np.array(image), boxlist.bbox.numpy().astype(np.int32)
            )

            ax.imshow(overlay)

        plt.show()
