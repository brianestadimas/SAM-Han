
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
class Predictor:
    def __init__(
        self,
        model: Sam,
    ) -> None:
        self.predictor = SamPredictor(model)
        self.device = self.predictor.device
    
    # one point
    def predict_single_point(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray = None,
        multi_masks: bool = False,
        get_highest: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using sam, predict masks for one point prompt.

        Arguments:
            image (np.array): shaped HxWxC
            point_coords (np.array): shaped 1x2
            point_labels (np.array) shaped 1
            multi_masks (bool): If set True, you will get three masks
            get_highest (bool) Only valid when multi_masks=True. If set True, 
                you will get highest scored mask among three masks.
        """
        self.predictor.set_image(image)
        # If point labels is None, set foreground point
        if point_labels is None:
            point_labels = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multi_masks,
        )
        if multi_masks and get_highest:
            best_idx = np.argmax(scores)
            masks = masks[best_idx]
            scores = scores[best_idx]

        self.predictor.reset_image()

        masks = masks.squeeze()
        scores = scores.squeeze()
        
        return masks, scores


    # one box
    def predict_single_box(
        self,
        image: np.ndarray,
        input_box: np.ndarray,
        multi_masks: bool = False,
        get_highest: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using sam, predict masks for one box prompt.

        Arguments:
            image (np.array): shaped HxWxC
            input_box (np.array): shaped 1x4 (xyxy)
            multi_masks (bool): If set True, you will get three masks
            get_highest (bool) Only valid when multi_masks=True. If set True, 
                you will get highest scored mask among three masks.
        """
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=multi_masks,
        )
        if multi_masks and get_highest:
            best_idx = np.argmax(scores)
            masks = masks[best_idx]
            scores = scores[best_idx]

        self.predictor.reset_image()

        masks = masks.squeeze()
        scores = scores.squeeze()

        return masks, scores


    # multi points, masks for each point
    def predict_multi_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray = None,
        multi_masks: bool = False,
        get_highest: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using sam, predict mask for multiple points prompt.
        With this method, you can get masks for each point.

        Arguments:
            image (np.array): shaped HxWxC
            point_coords (np.array): shaped Nx2
            point_labels (np.array or None) shaped N
        """
        if point_labels is not None:
            assert len(point_coords) == len(point_labels)

        if len(point_coords) == 1:
            return self.predict_single_point(
                image, point_coords, point_labels, multi_masks, get_highest
            )
        else:
            self.predictor.set_image(image)
            transformed_points = self.predictor.transform.apply_coords(point_coords, image.shape[:2])
            in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
            if point_labels is not None:
                in_labels = torch.as_tensor(point_labels, dtype=torch.int, device=in_points.device)
            else:
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

            masks, iou_preds, _ = self.predictor.predict_torch(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=multi_masks,
            )

            if not multi_masks:
                masks = masks.squeeze().cpu().numpy()
                iou_preds = iou_preds.squeeze().cpu().numpy()

            elif multi_masks and not get_highest:
                masks = masks.squeeze().cpu().numpy()
                iou_preds = iou_preds.squeeze().cpu().numpy()

            else:
                idx = torch.argmax(iou_preds, dim=1)
                new_masks = []
                for i in range(len(masks)):
                    new_masks.append(masks[i][idx[i]].unsqueeze(0))
                new_masks = torch.cat(new_masks, dim=0)
                masks = new_masks.cpu().numpy()
                iou_preds = torch.max(iou_preds, dim=1)[0].cpu().numpy()
                del new_masks, idx

            return masks, iou_preds
            
    
    # multi points speciying object
    def predict_multi_points_specify(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using sam, predict mask for multiple points prompt.
        With this method, you can get only one mask specifying by multiple points.

        Arguments:
            image (np.array): shaped HxWxC
            point_coords (np.array): shaped Nx2
            point_labels (np.array or None) shaped N
        """
        self.predictor.set_image(image)
        if point_labels is None:
            point_labels = np.array([1]*len(point_coords))
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        
        self.predictor.reset_image()
        
        return masks, scores


    # multi boxes
    def predict_multi_boxes(
        self,
        image: np.ndarray,
        input_boxes: np.ndarray,
        multi_masks: bool = False,
        get_highest: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(input_boxes) == 1:
            return self.predict_single_box(
                image, input_boxes, multi_masks, get_highest
            )
        else:
            self.predictor.set_image(image)
            input_boxes = self.predictor.transform.apply_boxes(input_boxes, image.shape[:2])
            input_boxes = torch.as_tensor(input_boxes, device=self.predictor.device)
            masks, iou_preds, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=input_boxes,
                multimask_output=multi_masks,
            )

            if not multi_masks:
                masks = masks.squeeze().cpu().numpy()
                iou_preds = iou_preds.squeeze().cpu().numpy()

            elif multi_masks and not get_highest:
                masks = masks.squeeze().cpu().numpy()
                iou_preds = iou_preds.squeeze().cpu().numpy()

            else:
                idx = torch.argmax(iou_preds, dim=1)
                new_masks = []
                for i in range(len(masks)):
                    new_masks.append(masks[i][idx[i]].unsqueeze(0))
                new_masks = torch.cat(new_masks, dim=0)
                masks = new_masks.cpu().numpy()
                iou_preds = torch.max(iou_preds, dim=1)[0].cpu().numpy()
                del new_masks, idx

            return masks, iou_preds


    
    
