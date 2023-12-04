import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from nibabel.processing import resample_to_output
from monai.config import print_config
from monai.handlers.utils import from_engine
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    AsDiscreted,
    LoadImaged,
    LoadImage,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SaveImaged,
    EnsureTyped,
    EnsureType,
    Invertd,
    ResampleToMatchd,
    Activationsd,
    RemoveSmallObjectsD,
    FillHolesD,
    FillHoles,
    RemoveSmallObjects,

)
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch

from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
import nibabel as nib

print_config()

test_image = os.path.normpath(
    "/home/mathildef/PycharmProjects/segmentation-from-david/segmentation-torch-brain-tumors/datasets/Labeled_data/imagesTs/")
test_label = os.path.normpath(
    "/home/mathildef/PycharmProjects/segmentation-from-david/segmentation-torch-brain-tumors/datasets/Labeled_data/labelsTs/")

test_images = sorted(glob.glob(os.path.join(test_image, "*.nii.gz")))
test_labels = sorted((glob.glob(os.path.join(test_label, "*.nii.gz"))))

test_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
image_ori=[]
affine=[]
spacing =[]
for image in test_images:
    ori = nib.load(image)
    image_ori.append(ori)
    affine.append(ori.affine)
    spacing.append(ori.header.get_zooms())


test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(0.2, 0.2, 0.2),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),  # -175, 250
        CropForegroundd(keys=["image"], source_key="image"),
        #ToTensord(keys=["image"]),
    ]
)

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=test_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        #device="cpu"

    ),
    #Activationsd(keys="pred", sigmoid=True),
    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    SaveImaged(keys="pred", output_dir="./output_pred_US_DA", output_postfix="seg_not_pretrained_USDA", resample=True)
])

test_ds = Dataset(data=test_dicts, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="conv",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)
model.load_state_dict(
    torch.load(os.path.normpath("logdir/best_metric_model_without_pretraining_US_DA1811.pth"), map_location=device))
model.to(device)
model.eval()

loader = LoadImage()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        test_input = data["image"].to(device)

        data["pred"] = sliding_window_inference(test_input, roi_size, sw_batch_size, model)
        data = [post_transforms(i) for i in decollate_batch(data)]
        test_output = from_engine(["pred"])(data)
        original_image = loader(test_output[0].meta["filename_or_obj"])[0].detach().cpu().astype(np.float32)

        nib.save(nib.Nifti1Image(test_output[0].detach().cpu().astype(np.uint8)[0,:,:,:], affine=affine[i]),
          "/home/mathildef/PycharmProjects/segmentation-from-david/not_pretrain_US_DA/BT_US_BEFORE_0"+str(i+24)+".nii.gz")
