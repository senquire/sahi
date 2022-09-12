from sahi.model import DetectionModel
import argparse
import os
import sys
import time

import numpy as np

import cv2
import torch
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from PIL import Image

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
#from sahi.utils.import_utils import is_torch_available
from sahi.utils.torch import is_torch_cuda_available
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
logger = logging.getLogger(__name__)

class NanoSlice(DetectionModel): 
	def load_model(self):
		"""
		Detection model is initialized and set to self.model.
		"""
		# import pdb; pdb.set_trace()

		load_config(cfg, self.config_path)
		self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

		self.cfg = cfg
		# create model
		model = build_model(
			self.cfg.model
		)
		logger = Logger(-1, use_tensorboard=False)
		ckpt = torch.load(self.model_path, map_location=lambda storage, loc: storage)
		load_model_weight(model, ckpt, logger)
		self.model = model.eval()
		# set category_mapping
		if not self.category_mapping:
			category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
			self.category_mapping = category_mapping
		self.confidence_threshold = 0

	def perform_inference(self, image: np.ndarray, image_size: int = None):
		"""
		Prediction is performed using self.model and the prediction result is set to self._original_predictions.
		Args:
			image: np.ndarray
				A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
			image_size: int
				Inference input size.
		"""
		# Confirm model is loaded
		assert self.model is not None, "Model is not loaded, load it by calling .load_model()"
		#import pdb;pdb.set_trace()

		# perform inference
		# if isinstance(image, np.ndarray):
		#	  # https://github.com/obss/sahi/issues/265
		#	  image = image[:, :, ::-1]

		img_info = {"id": list(range(len(image)))}
		img_info["file_name"] = None
		height, width = image.shape[2:0:-1]
		img_info["height"] = height
		img_info["width"] = width 
		meta = dict(img_info=img_info, raw_img=image, img=[], warp_matrix=[])
		for img in image:
			meta_data = self.pipeline(None, {"img":img}, self.cfg.data.val.input_size)
			meta["img"].append(torch.from_numpy(np.asarray(meta_data["img"]).transpose(2,0,1)).to(self.device)
)
			meta["warp_matrix"].append(meta_data["warp_matrix"])

		meta = naive_collate([meta])
		meta["img"] = meta["img"][0]
		meta["warp_matrix"] = meta["warp_matrix"][0]
		meta["img_info"]["id"] = meta["img_info"]["id"][0]
		meta["img"] = stack_batch_img(meta["img"], divisible=32)
		with torch.no_grad():
			import pdb;pdb.set_trace()
			results = self.model.inference(meta)

		# compatibility with sahi v0.8.15
		if not isinstance(image, list):
			image = [image]
		self._original_predictions = results

	@property
	def num_categories(self):
		"""
		Returns number of categories
		"""
		if isinstance(self.cfg.class_names, str):
			num_categories = 1
		else:
			num_categories = len(self.cfg.class_names)
		return num_categories

	@property
	def has_mask(self):
		"""
		Returns if model output contains segmentation mask
		"""
		return False

	@property
	def category_names(self):
		if type(self.cfg.class_names) == str:
			return (self.cfg.class_names,)
		else:
			return self.cfg.class_names

	def _create_object_prediction_list_from_original_predictions(
		self,
		shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
		full_shape_list: Optional[List[List[int]]] = None,
	):
		"""
		self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
		self._object_prediction_list_per_image.
		Args:
			shift_amount_list: list of list
				To shift the box predictions from sliced image to full sized image, should
				be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
			full_shape_list: list of list
				Size of the full image after shifting, should be in the form of
				List[[height, width],[height, width],...]
		"""
		original_predictions = self._original_predictions
		category_mapping = self.category_mapping
		self._object_prediction_list = []
		# compatilibty for sahi v0.8.15
		shift_amount_list = fix_shift_amount_list(shift_amount_list)
		full_shape_list = fix_full_shape_list(full_shape_list)

		# parse boxes from predictions
		num_categories = self.num_categories
		object_prediction_list_per_image = []

		for image_ind, original_prediction in original_predictions.items():
			shift_amount = shift_amount_list[image_ind]
			full_shape = None
			print(f"SHIFT_AMOUNT = {shift_amount}")
			boxes = original_prediction

			object_prediction_list = []

			# process predictions
			for category_id in range(num_categories):
				category_boxes = boxes[category_id]
				
				num_category_predictions = len(category_boxes)

				for category_predictions_ind in range(num_category_predictions):
					bbox = category_boxes[category_predictions_ind][:4]
					score = category_boxes[category_predictions_ind][4]
					category_name = category_mapping[str(category_id)]

					# ignore low scored predictions
					if score < self.confidence_threshold:
						continue

					bool_mask = None

					# fix negative box coords
					bbox[0] = max(0, bbox[0])
					bbox[1] = max(0, bbox[1])
					bbox[2] = max(0, bbox[2])
					bbox[3] = max(0, bbox[3])

					# fix out of image box coords
					if full_shape is not None:
						bbox[0] = min(full_shape[1], bbox[0])
						bbox[1] = min(full_shape[0], bbox[1])
						bbox[2] = min(full_shape[1], bbox[2])
						bbox[3] = min(full_shape[0], bbox[3])

					# ignore invalid predictions
					if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
						logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
						continue
					object_prediction = ObjectPrediction(
						bbox=bbox,
						category_id=category_id,
						score=score,
						bool_mask=bool_mask,
						category_name=category_name,
						shift_amount=shift_amount,
						full_shape=full_shape,
					)
					self._object_prediction_list.append(object_prediction)
			object_prediction_list_per_image.append(object_prediction_list)
		self._object_prediction_list_per_image = object_prediction_list_per_image
		import pdb;pdb.set_trace()
"""
model = NanoSlice(
	model_path = "/home/thor/nanodet_model/model.ckpt",
	# model: Optional[Any] = None,
	config_path = "/home/thor/nanodet_model/config.yml",
	load_at_init = True,
	image_size = 416,
	device="cpu"
)
"""

