""" Functions/classes to perform inference and label tissue edges as apical vs basal """

import glob
import time
import random
import numpy as np
import cv2
from skimage.morphology import convex_hull_image
import tifffile
from matplotlib import pyplot as plt
from src.utils.logging import StandardLogger
from src.models.dl4mia_tissue_unet.predict import Predicter
from src.models.tissue_edge_inference.edge_utils.general import inv_dictionary
from src.models.tissue_edge_inference.edge_utils.error_utils import NoEdgesFoundError
from src.models.tissue_edge_inference.edge_utils.data_utils import mask_statistics
from src.models.tissue_edge_inference.edge_utils.img_utils import (
    overlay_image,
    rm_components_by_size,
)


class TissueEdgeClassifier:
    """
    Labels tissue edge as apical or basal

    Steps:
        1. Use segmentation model to generate mask of image
        2. Preprocess the mask to fill in holes and smooth edges
        3. Use gradient or dilation/subtration to get edges of mask
        4. Connected components to isolate distinct edges in image
        5. Convex hull for each connected component edge
        6. Copmute overlap of hull with mask for each edge convex hull
        7. classify edges hull-mask overlap > 50% = basal else apical
        8. Relabel edges with apical=1 basal=2
        9. Extract coordinates of the edges
    """

    def __init__(self, segmenter: Predicter, train_mask_dir: str):
        self._logger = StandardLogger(__name__)
        self.segmenter = segmenter
        self.edge_dict = {"apical": 1, "basal": 2}
        self._seg_stats = mask_statistics(train_dir=train_mask_dir)
        self._BG_5TH_PTILE = self._seg_stats["bg_5th"]
        self._FG_5TH_PTILE = self._seg_stats["fg_5th"]
        # self._BG_5TH_PTILE = 0.038
        # self._FG_5TH_PTILE = 0.043

    def classify_img(self, img: np.ndarray) -> dict:
        """
        Classify an image to label apical /basal edges and return dict
        of {'apical':apical_edge_list, 'basal':basal_edge_list}.

        Args:
            img (np.ndarray): Image of tissue to classify

        returns dictionary of apical and basal edge pixel coordinates
        """
        try:
            # Use segmenter to segment tissue into binary mask
            mask, activation = self.segmenter.predict(img)
            mask = mask.astype(np.uint8)

            # Classify edges of the mask as apical/basal
            edge_dict, edge_cc, sem_labels, mask = self.classify_mask(mask)
        except NoEdgesFoundError as e:
            self._logger.warning(e)
        except:
            self._logger.exception("Error while classifying image")
            raise
        else:

            return edge_dict, edge_cc, sem_labels, mask

    def classify_mask(self, mask: np.ndarray) -> dict:
        """
        Classify an mask to label apical /basal edges and return dict
        of {'apical':apical_edge_list, 'basal':basal_edge_list}.

        Args:
            img (np.ndarray): Tissue mask to classify

        returns dictionary of apical and basal edge pixel coordinates
        """
        try:
            # preprocess mask to fill in holes
            processed_mask = self._preprocess_mask(mask)

            # Label edges of binary mask with 1 for apical and 2 for basal
            (
                edge_cc,
                sem_labels,
                sem_edge_img,
                sem_largest_edge_img,
            ) = self._label_mask_edge(processed_mask)

            # Extract pixel coordinates of apical and basal edges to dictionary
            edge_dict = self._get_edge_dict(sem_edge_img)
        except NoEdgesFoundError as e:
            self._logger.warning(e)
        except:
            self._logger.exception("Error while classifying mask")
        else:

            return edge_dict, edge_cc, sem_labels, mask

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask to fill in holes, select the largest contiguous region,
        and smooth the mask edges

        Args:
            mask (np.ndarray): Tissue mask to preprocess

        returns np.ndarray of preprocessed mask
        """
        # Remove spurious fg/bg regions based on statistics of training data
        preprocessed = rm_components_by_size(
            mask=mask, size_frac=self._BG_5TH_PTILE, smaller=True, region="bg"
        )
        preprocessed = rm_components_by_size(
            mask=preprocessed, size_frac=self._FG_5TH_PTILE, smaller=True, region="fg"
        )
        preprocessed = preprocessed.astype(np.uint8)

        # smooth edges
        preprocessed = cv2.medianBlur(preprocessed, ksize=7)

        return preprocessed

    def _associate_edge_to_mask(
        self, mask: np.ndarray, num_edge_labels: int, edge_cc: np.ndarray
    ):
        """
        Associates and edge label with a detected foreground region in `mask`.
        Returns dictionary associating labels and a connected component labeled
        `mask` image.

        Args:
            mask: detected foreground image
            num_edge_labels: number of labeled edges from connected components
            edge_cc: connected component edge labels

        Returns:
            edge_to_mask: dictionary associating edge label to mask label
            mask_cc: np.ndarray of connected components in `mask`
        """
        # Get mask and edge connected components
        num_mask_labels, mask_cc = cv2.connectedComponents(mask)

        # init dictionary to associate edge with mask
        edge_to_mask = {}

        # Loop over edges. If edge intersects with mask region, then associate
        # edge label with mask label
        for edge_label in range(1, num_edge_labels):
            roi_edge = edge_cc == edge_label
            for mask_label in range(1, num_mask_labels):
                roi_mask = mask_cc == mask_label
                if np.any(np.bitwise_and(roi_edge, roi_mask)):
                    edge_to_mask[edge_label] = mask_label
                    break

        return edge_to_mask, mask_cc

    def _label_mask_edge(self, mask: np.ndarray) -> np.ndarray:
        """
        Labels edges of mask as apical or basal.

        Apical edges are assumed to be concave (curves in towards tissue) and
        basal edges are assumed to be convex (curves away from tissue).
        Evaluated by computing area overlap of convex hull for each edge with
        the tissue mask.

        For two edges:
            larger hull and mask overlap -> basal (other apical)

        else:
            hull and mask overlap > 50% -> basal (else apical)

        Args:
            mask: binary mask of segmented tissue

        returns numpy array with labels
        """
        # Get binary mask of mask edges
        edges = self._get_mask_edges(mask=mask)

        # get labeled connected components in image (bg=0)
        num_labels, edge_cc = cv2.connectedComponents(edges)
        if num_labels == 1:
            raise NoEdgesFoundError("Could not detect any edges in mask")

        # Associate edge with a mask region
        edge_to_mask, mask_cc = self._associate_edge_to_mask(
            mask=mask, num_edge_labels=num_labels, edge_cc=edge_cc
        )

        # Iterate through the first 2 labels (ignore 0 label since its background)
        # create dict of {'label1': overlap_ratio1, 'label2': overlap_ratio2}
        label_overlap = {}
        for label in np.unique(edge_cc):
            if label == 0:
                continue
            edge_mask = edge_cc == label
            hull = convex_hull_image(edge_mask)
            overlap_ratio = self._hull_overlap_ratio(
                edge=edge_mask, hull=hull, mask=mask
            )
            label_overlap[label] = overlap_ratio

        # Relabel the edges according to their overlap ratio and the edge_dict
        # attribute.
        semantic_labels = self._label_by_overlap_ratio(
            edge_cc=edge_cc, overlap_dict=label_overlap, edge_to_region=edge_to_mask
        )
        semantic_edges = self._semantic_edge_image(
            edge_cc=edge_cc, sem_labels=semantic_labels
        )

        # Post process the edges to get the largest apical and largest basal edge
        largest_semantic_labels = self._postprocess_edges(
            edge_cc=edge_cc, sem_labels=semantic_labels
        )
        largest_semantic_edges = semantic_edges = self._semantic_edge_image(
            edge_cc=edge_cc, sem_labels=largest_semantic_labels
        )

        return edge_cc, semantic_labels, semantic_edges, largest_semantic_edges

    def _get_mask_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Erode mask w/ 3x3 kernel and subtract it from the original mask.
        Results in the edge being the pixels exactly on the edge of the mask.

        Args:
            mask: The mask to dilate.

        returns the mask edges
        """
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
        eroded = cv2.erode(src=mask, kernel=kernel)
        edges = mask - eroded
        return edges

    def _hull_overlap_ratio(
        self, edge: np.ndarray, hull: np.ndarray, mask: np.ndarray
    ) -> float:
        """
        Computes area overlap between (edgeless) convex hull and (edgeless) mask.

        Args:
            edge: binary mask of edges
            hull: binary mask of convex hull
            mask: binary mask of tissue

        returns the ratio between the hull and mask overlap area to hull area
        """
        # compute the edgeless hull and its area
        ne_hull = (hull == 1) & (edge == 0)
        ne_hull_area = np.sum(ne_hull)

        # Compute overlap area between hull and mask and ratio of overlap to hull
        ne_overlap_area = np.sum((mask == 1) & (ne_hull == 1))
        area_ratio = ne_overlap_area / ne_hull_area

        return area_ratio

    def _label_by_overlap_ratio(
        self, edge_cc: np.ndarray, overlap_dict: dict, edge_to_region: dict
    ) -> dict:
        """
        Relabel connected component edges based on the convex hull and mask overlap.

        if 2 edges on a single tissue region:
            basal = larger overlap and apical = smaller overlap
        else:
            50% overlap = basal else apical

        Args:
            edge_cc: np.ndarray of connected component edges (can have at most 2 edges)
            overlap_dict: dict of overlap ratios for each edge label
            edge_to_dict: dict associating edge label with a fg region (mask) label

        Returns:
            dict assoicating connected components label with tissue type label ('apical' or 'basal')
        """
        # Ensure each edge has a overlap ratio
        edge_labels = np.sort(np.unique(edge_cc))[1:]  # ignore the 0 index since bg
        dict_labels = np.sort(np.array(list(overlap_dict.keys())))
        reg_labels = np.sort(np.array(list(edge_to_region.keys())))
        if not np.all((edge_labels == dict_labels) & (dict_labels == reg_labels)):
            raise ValueError("All edges must have an overlap ratio and region")

        # Invert dictionary to get dictionary of {mask_label:[edge_label1, edge_label2,...],...}
        region_to_edge = inv_dictionary(edge_to_region)

        # Dictionary to store the edge type for each edge
        sem_labels = {}

        # Iterate through each edge and assign a semantic label
        for label, overlap in overlap_dict.items():
            edge_region = edge_to_region[label]
            regions_edges = region_to_edge[edge_region]
            # 1 edge on tissue: 50% overlap or more = basal, else apical
            if len(regions_edges) == 1:
                if overlap > 0.5:
                    sem_labels[label] = "basal"
                else:
                    sem_labels[label] = "apical"
            # 2 edge on tissue: larger overlap = basal, smaller overlap = apical
            elif len(regions_edges) == 2:
                other_edge_label = np.setxor1d(
                    np.array(regions_edges), np.array(label)
                )[0]
                other_overlap = overlap_dict[other_edge_label]
                if overlap > other_overlap:
                    sem_labels[label] = "basal"
                else:
                    sem_labels[label] = "apical"
            # more than 1 edge on tissue: 50% overlap or more = basal, else apical
            else:
                if overlap > 0.5:
                    sem_labels[label] = "basal"
                else:
                    sem_labels[label] = "apical"
                self._logger.warning(
                    f"Unusual edge detection having {len(regions_edges)} edges. Usually a piece of tissue has 1 or 2 edges."
                )

        return sem_labels

    def _semantic_edge_image(self, edge_cc: np.ndarray, sem_labels: dict):
        """
        Create a new image of where each edge in the connected component gets
        assinged an new value based on its classification as 'apical' or 'basal'
        in the sem_labels dictionary.

        Args:
            edge_cc: connected component edge image
            sem_labels: Association between labels in `edge_cc` with 'apical' or 'basal'

        Returns:
            np.ndarray where each edge is semantically labeled
        """
        # Ensure each edge has a overlap ratio
        edge_labels = np.sort(np.unique(edge_cc))[1:]  # ignore the 0 index since bg
        dict_labels = np.sort(np.array(list(sem_labels.keys())))
        for label in dict_labels:
            if label not in edge_labels.tolist():
                raise ValueError(
                    f"Invalid semantic label ({label}) for labels in edges ({edge_labels})"
                )

        # Relabel each edge with appropriate value corresponding to basal or apical
        semantic_edges = np.zeros_like(edge_cc)
        for label, edge_type in sem_labels.items():
            edge_mask = edge_cc == label
            semantic_edges[edge_mask] = self.edge_dict[edge_type]

        return semantic_edges

    def _get_edge_dict(self, edge_img: np.ndarray) -> dict:
        """
        Return dict of {'apical':apical_edge_list, 'basal':basal_edge_list}
        from an image labeld as 1=apical and 2=basal.

        Args:
            edge_img (np.ndarray): Edge classified image with apical=1 and basal=2

        returns dictionary of apical and basal edge pixel coordinates
        """
        apical_edges = edge_img == self.edge_dict["apical"]
        apical_inds = np.where(apical_edges != 0)
        apical_coords = tuple(zip(*apical_inds))
        basal_edges = edge_img == self.edge_dict["basal"]
        basal_inds = np.where(basal_edges != 0)
        basal_coords = tuple(zip(*basal_inds))
        return {"apical": apical_coords, "basal": basal_coords}

    def _postprocess_edges(self, edge_cc: np.ndarray, sem_labels: dict):
        """
        Postprocess the edges.

        Args:
            edge_cc: connected component edge image
            sem_labels: Association between labels in `edge_cc` with 'apical' or 'basal'

        Returns:
            np.ndarray:
        """
        # Ensure each edge has a overlap ratio
        edge_labels = np.sort(np.unique(edge_cc))[1:]  # ignore the 0 index since bg
        dict_labels = np.sort(np.array(list(sem_labels.keys())))
        if not np.all(edge_labels == dict_labels):
            raise ValueError("All edges must have an have a sem_labels")

        # Get the largest edge of each type
        largest_apical_size = 0
        largest_apical_label = None
        largest_basal_size = 0
        largest_basal_label = None
        for label, edge_type in sem_labels.items():
            edge_size = np.sum(edge_cc[edge_cc == label])
            if edge_type == "apical":
                if edge_size > largest_apical_size:
                    largest_apical_size = edge_size
                    largest_apical_label = label
            if edge_type == "basal":
                if edge_size > largest_basal_size:
                    largest_basal_size = edge_size
                    largest_basal_label = label

        # Make dictionary associating largeset edge w/ its edge type
        largest_edges = {}
        for label in [largest_apical_label, largest_basal_label]:
            if label is not None:
                largest_edges[label] = sem_labels[label]

        return largest_edges


def make_plot(imgs: list):
    from math import ceil, sqrt

    n = len(imgs)
    ASP = 16 / 9
    rows = ceil(sqrt(n / ASP))
    cols = int(ASP * rows)
    fig, ax = plt.subplots(rows, cols)
    iter = 0
    for i in range(rows):
        for j in range(cols):
            if iter < len(imgs):
                ax[i, j].imshow(imgs[iter])

            ax[i, j].axis("off")
            iter += 1
    plt.show()


if __name__ == "__main__":
    unet_ckpt = (
        "src/models/dl4mia_tissue_unet/results/20220824_181000_Colab_gpu/best.pth"
    )
    unet_model = Predicter.from_ckpt(unet_ckpt)
    TRAIN_MASK_DIR = "data/processed/uncropped/train/masks"
    TC = TissueEdgeClassifier(segmenter=unet_model, train_mask_dir=TRAIN_MASK_DIR)
    CKPT_PATH = "models/"
    MASK_DIR = "data/processed/uncropped/val/masks"
    IMAGE_DIR = "data/processed/uncropped/val/images"
    INAME = "E14a_000008_crop00.tif"
    INAME = "e15 20x p_000011_crop00.tif"
    INAME = "e15 c_000011_crop00.tif"
    INAME = "e15 g_000012_crop00.tif"
    mask_paths = glob.glob(f"{MASK_DIR}/*.tif")
    img_paths = glob.glob(f"{IMAGE_DIR}/*.tif")
    paths = zip(mask_paths, img_paths)
    mask_path = random.choice(mask_paths)
    # mask_path = f"{MASK_DIR}/{INAME}"
    imgs = []
    # for mask_path in mask_paths:
    for mask_path, img_path in list(paths):
        mask_img = tifffile.imread(mask_path)
        real_img = tifffile.imread(img_path)
        t0 = time.time()
        # edge_dict, edge_cc, sem_labels = TC.classify_mask(mask_img)
        try:
            edge_dict, edge_cc, sem_labels, new_mask = TC.classify_img(real_img)
            t1 = time.time()
            print(f"Total time:\t{t1 - t0}")
            overlay = overlay_image(
                image=real_img,
                edge_label_dict=sem_labels,
                mask=new_mask,
                edge_labels=edge_cc,
            )
            imgs.append(overlay)
        except Exception as e:
            print(f"exception passed: {e}")
    make_plot(
        imgs,
    )
