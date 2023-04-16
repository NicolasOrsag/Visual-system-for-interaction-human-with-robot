import logging

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from torch import nn

from fsdet.modeling.roi_heads import build_roi_heads

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.features = {}
        self.freeze = False

        self.max_size = [0, 0]

        self.min_widths = set()

        self.saved_images = set()

        self.freeze2 = False

        self.saved_images2 = set()

        if cfg.MODEL.BACKBONE.FREEZE:
            # self.freeze = True
            self.freeze2 = True
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        # print(images.tensor[0].shape)
        # self.min_widths.add(images.tensor.shape[3])
        # print(self.min_widths)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        # if self.freeze:
        #     # if len(batched_inputs == 1):
        #     #     if batched_inputs[0]["image_id"] not in self.features:
        #     #         features = self.backbone(images.tensor[0].unsqueeze(0))
        #     #             self.features[batched_inputs[0]["image_id"]] = features
        #     # min_width = str(images.tensor[0].shape[1])
        #     # name = batched_inputs[0]["image_id"] + "_" + min_width
        #     # features_path = "features/" + name + ".pt"
        #     # if name not in self.saved_images:
        #     #     features = self.backbone(images.tensor[0].unsqueeze(0))
        #     #     self.saved_images.add(name)
        #     #     torch.save(features, features_path)
        #     # else:
        #     #     features = torch.load(features_path)
        #
        #     for i in range(len(batched_inputs)):
        #         min_width = str(images.tensor[i].shape[1])
        #         # if batched_inputs[i]["image_id"] + "_" + min_width not in self.features:
        #         if batched_inputs[i]["image_id"] not in self.features:
        #             features = self.backbone(images.tensor[i].unsqueeze(0))
        #             # self.features[batched_inputs[i]["image_id"] + "_" + min_width] = features
        #             self.features[batched_inputs[i]["image_id"]] = features
        #
        #     if len(batched_inputs) == 1:
        #         min_width = str(images.tensor[0].shape[1])
        #         features = self.features[batched_inputs[0]["image_id"] + "_" + min_width]
        #         # features = self.features[batched_inputs[0]["image_id"]]
        #         # print(sorted(self.features.keys()))
        #     else:
        #         features = {k: torch.stack([self.features[batched_inputs[0]["image_id"]][k],
        #                                     self.features[batched_inputs[1]["image_id"]][k]]).squeeze() for k in
        #                     self.features[batched_inputs[0]["image_id"]].keys()}
        #     # features = self.features[batched_inputs[0]["image_id"]]
        # else:
        #     features = self.backbone(images.tensor)
        min_width = str(images.tensor[0].shape[1])
        name = batched_inputs[0]["image_id"] + "_" + min_width

        if self.freeze2:
            if name not in self.saved_images2:
                features = self.backbone(images.tensor)
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances
                )
                self.saved_images2.add(name)
            else:
                proposals = None
                features = None
                gt_instances = None

            _, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances, name
            )

            return detector_losses

        else:
            features = self.backbone(images.tensor)

            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances
                )

            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

    def inference(
            self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
