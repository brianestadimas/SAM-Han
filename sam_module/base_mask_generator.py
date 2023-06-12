import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import time

from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    batch_iterator,
    build_all_layer_point_grids,
    generate_crop_boxes,
)

# for grid points prompt
class BaseMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
    

    @torch.no_grad()
    def get_masks(self, 
                  image: np.ndarray, 
                  points_per_side: Optional[int] = None,
                  multi_mask: bool = False,
                  get_highest: bool = False
    ) -> torch.Tensor:
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                self.crop_n_layers,
                self.crop_n_points_downscale_factor,
            )
        masks = self._get_masks(image, multi_mask, get_highest)

        return masks.numpy()
    

    def _get_masks(self, image: np.ndarray, multi_mask: bool, get_highest: bool) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        # Iterate over image crops
        masks_data = []
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            masks = self._process_crop(image, crop_box, layer_idx, orig_size, multi_mask, get_highest)
            masks_data.append(masks)
            
        return torch.cat(masks_data, dim=0)
    

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        multi_mask: bool, 
        get_highest: bool,
    ) -> torch.Tensor:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        masks_data = []
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            default_masks = self._process_batch(points, cropped_im_size, crop_box, orig_size, multi_mask, get_highest)
            masks_data.append(default_masks)
        self.predictor.reset_image()

        return torch.cat(masks_data, dim=0)


    def _process_batch( # * 64
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        multi_mask: bool, 
        get_highest: bool,
    ) -> torch.Tensor:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=multi_mask,
            return_logits=True,
        )

        if not multi_mask:
            default_mask = (masks.squeeze() > self.predictor.model.mask_threshold).cpu()
        elif multi_mask and not get_highest:
            default_mask = (masks.squeeze() > self.predictor.model.mask_threshold).cpu()
        else:
            idx = torch.argmax(iou_preds, dim=1)
            new_masks = []
            for i in range(len(masks)):
                new_masks.append(masks[i][idx[i]].unsqueeze(0))
            new_masks = torch.cat(new_masks, dim=0).unsqueeze(1)
            default_mask = (new_masks.squeeze() > self.predictor.model.mask_threshold).cpu()
            del new_masks

        return default_mask
        


