import numpy as np
import seg_metrics.seg_metrics as sg
import os
import nibabel as nib
import glob
import numpy as np

# During post-processing, the background get assigned as foreground and the foreground as background
# this finction will
def switchbackgroundandforeground():
    pred_label = os.path.normpath(
        "/home/mathildef/PycharmProjects/segmentation-from-david/not_pretrain_US_DA")
    i=0
    test_images = sorted(glob.glob(os.path.join(pred_label, "*.nii.gz")))
    for image in test_images:
        ori = nib.load(image)
        affine = ori.affine
        data = ori.get_fdata()

        data_new = np.copy(data)
        data_new[data > 0.5] = 0
        data_new[data <= 0.5] = 1
        nib.save(nib.Nifti1Image(data_new, affine=affine, dtype="ubyte"), "/home/mathildef/PycharmProjects/results_SSL/original_image_size/not_pretrained_US_DA/BT_US_BEFORE_0"+str(i+26)+".nii.gz")
        i=+1


#switchbackgroundandforeground()


def calc_hd95():
    gdth = '/home/mathildef/PycharmProjects/nnUNetv2_BT/data/test_data/115_test_images/labelsTs'
    pred = '/home/mathildef/PycharmProjects/results_SSL/original_image_size/not_pretrained_US_DA'
    csv_file = '/home/mathildef/PycharmProjects/results_SSL/original_image_size/SSL_NOTpretrainedUSDA_ori.csv'

    metrics = sg.write_metrics(labels=[1],  # exclude background
                               gdth_path=gdth,
                               pred_path=pred,
                               csv_file=csv_file,
                               TPTNFPFN=True)
    #print(metrics)  # a list of dictionaries which includes the metrics for each pair of image.

calc_hd95()
