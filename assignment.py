import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

# --- Dataset Class ---
class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None, image_size=512):
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_size = image_size
        self.img_ids = list(self.coco.imgs.keys())
        self.cat_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        mask = Image.new('L', image.size)

        for ann in anns:
            if 'segmentation' not in ann or ann['category_id'] not in self.cat_ids:
                continue
            seg = ann['segmentation']
            cat_id = ann['category_id']
            cls_idx = self.cat_ids.index(cat_id) + 1
            if isinstance(seg, list):
                for s in seg:
                    poly = Image.new('L', image.size, 0)
                    ImageDraw.Draw(poly).polygon(s, outline=cls_idx, fill=cls_idx)
                    mask = Image.composite(poly, mask, poly)

        if self.transforms:
            image = self.transforms(image)
            mask = T.functional.resize(mask, (self.image_size, self.image_size), interpolation=Image.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# --- DataModule ---
class COCODataModule(pl.LightningDataModule):
    def __init__(self, image_dir, annotation_path, batch_size=8, image_size=512):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.batch_size = batch_size
        self.image_size = image_size

        self.transforms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        dataset = COCOSegmentationDataset(self.image_dir, self.annotation_path, self.transforms, self.image_size)
        split = int(0.9 * len(dataset))
        self.train_ds, self.val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

# --- Model ---
class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes=81, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)['out']

    def compute_iou(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        ious = []
        for cls in range(self.hparams.num_classes):
            pred_inds = (preds == cls)
            target_inds = (targets == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        return np.nanmean(ious)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, masks)
        miou = self.compute_iou(preds, masks)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mIoU", miou, prog_bar=True)

        if batch_idx == 0:
            imgs_to_log = []
            for i in range(min(2, len(images))):
                img = images[i].cpu()
                gt = masks[i].cpu().unsqueeze(0).float() / self.hparams.num_classes
                pr = preds[i].cpu().unsqueeze(0).float() / self.hparams.num_classes
                imgs_to_log += [img, gt.repeat(3, 1, 1), pr.repeat(3, 1, 1)]

            grid = torchvision.utils.make_grid(imgs_to_log, nrow=3)
            self.logger.experiment.add_image("Samples", grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# --- Training Loop ---
if __name__ == "__main__":
    pl.seed_everything(42)

    image_dir = "/kaggle/input/coco-2017-dataset/train2017"
    ann_path = "/kaggle/input/coco-2017-dataset/annotations/instances_train2017.json"

    data = COCODataModule(image_dir, ann_path, batch_size=4)
    model = SegmentationModel(num_classes=81)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor="val_mIoU", mode="max", save_top_k=1)
    early_stop_cb = pl.callbacks.EarlyStopping(monitor="val_mIoU", mode="max", patience=3)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=pl.loggers.TensorBoardLogger("lightning_logs/")
    )

    trainer.fit(model, datamodule=data)
