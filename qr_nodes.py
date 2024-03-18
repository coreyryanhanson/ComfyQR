import numpy as np
import qrcode
from qrcode.image.styles.moduledrawers import (GappedSquareModuleDrawer,
                                               CircleModuleDrawer,
                                               RoundedModuleDrawer,
                                               VerticalBarsDrawer,
                                               HorizontalBarsDrawer)
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.colormasks import SolidFillColorMask
from qrcode.compat.pil import Image
import torch
import torch.nn.functional as F


class QRBase:
    def __init__(self):
        self.text = ""
        self.fill = None
        self.back = None

    FUNCTION = "generate_qr"
    CATEGORY = "ComfyQR"

    def _get_error_correction_constant(self, error_correction_string):
        if error_correction_string == "Low":
            return qrcode.constants.ERROR_CORRECT_L
        if error_correction_string == "Medium":
            return qrcode.constants.ERROR_CORRECT_M
        if error_correction_string == "Quartile":
            return qrcode.constants.ERROR_CORRECT_Q
        return qrcode.constants.ERROR_CORRECT_H

    def _img_to_tensor(self, img):
        out_image = np.array(img, dtype=np.uint8).astype(np.float32) / 255
        return torch.from_numpy(out_image).unsqueeze(0)

    def _make_qr(self, qr, fill_hexcolor, back_hexcolor, module_drawer):
        self.fill = self._parse_hexcolor_string(fill_hexcolor, "fill_hexcolor")
        self.back = self._parse_hexcolor_string(back_hexcolor, "back_hexcolor")
        qr.make(fit=True)
        if module_drawer == "Square":
            # Keeps using Square QR generation the old way for faster speeds.
            return qr.make_image(fill_color=self.fill, back_color=self.back)
        color_mask = SolidFillColorMask(back_color=self.back,
                                        front_color=self.fill)
        module_drawing_method = self._select_module_drawer(module_drawer)
        return qr.make_image(image_factory=StyledPilImage,
                             color_mask=color_mask,
                             module_drawer=module_drawing_method)

    def _parse_hexcolor_string(self, s, parameter):
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 3:
            rgb = (c + c for c in s)
        elif len(s) == 6:
            rgb = (s[i] + s[i+1] for i in range(0, 6, 2))
        else:
            raise ValueError(f"{parameter} must be 3 or 6 characters long")
        try:
            return tuple(int(channel, 16) for channel in rgb)
        except ValueError:
            raise ValueError(f"{parameter} contains invalid hexadecimal "
                             f"characters")

    def _validate_qr_size(self, size, max_size):
        if size > max_size:
            raise RuntimeError(f"QR dimensions of {size} exceed max size of "
                               f"{max_size}.")

    def _select_module_drawer(self, module_drawer_string):
        """Square is not included in the results, for a speed optimization
        applying color masks. Current version of python-qr code suffers a
        slowdown when using custom colors combined with custom module drawers.
        By bypassing square QRs, non standard colors will load faster."""
        if module_drawer_string == "Gapped square":
            return GappedSquareModuleDrawer()
        if module_drawer_string == "Circle":
            return CircleModuleDrawer()
        if module_drawer_string == "Rounded":
            return RoundedModuleDrawer()
        if module_drawer_string == "Vertical bars":
            return VerticalBarsDrawer()
        if module_drawer_string == "Horizontal bars":
            return HorizontalBarsDrawer()
        raise ValueError(f"Module drawing method of {module_drawer_string} "
                         f"not supported")

    def update_text(self, protocol, text):
        """This function takes input from a text box and a chosen internet
        protocol and stores a full address within an instance variable.
        Backslashes will invalidate text box input and this acts as a
        workaround to be able to use them when required in QR strings.

        Args:
            protocol: A categorical variable of one of the available internet
                protocols.
            text: The input from the text box.
        """
        if protocol == "Https":
            prefix = "https://"
        elif protocol == "Http":
            prefix = "http://"
        elif protocol == "None":
            prefix = ""
        self.text = prefix + text


class QRByImageSize(QRBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (["Http", "Https", "None"], {"default": "Https"}),
                "text": ("STRING", {"multiline": True}),
                "image_size": ("INT", {"default": 512,
                                       "min": 64,
                                       "max": 4096,
                                       "step": 64}),
                "fill_hexcolor": ("STRING", {"multiline": False,
                                             "default": "#000000"}),
                "back_hexcolor": ("STRING", {"multiline": False,
                                             "default": "#FFFFFF"}),
                "error_correction": (["Low", "Medium", "Quartile", "High"],
                                     {"default": "High"}),
                "border": ("INT", {"default": 1,
                                   "min": 0,
                                   "max": 100,
                                   "step": 1}),
                "resampling": (["Bicubic",
                                "Bilinear",
                                "Box",
                                "Hamming",
                                "Lanczos",
                                "Nearest"
                                ], {"default": "Nearest"}),
                "module_drawer": (["Square",
                                   "Gapped square",
                                   "Circle",
                                   "Rounded",
                                   "Vertical bars",
                                   "Horizontal bars"
                                   ], {"default": "Square"})
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("QR_CODE", "QR_VERSION")

    def _select_resampling_method(self, resampling_string):
        if resampling_string == "Nearest":
            return Image.NEAREST
        if resampling_string == "Bicubic":
            return Image.BICUBIC
        if resampling_string == "Bilinear":
            return Image.BILINEAR
        if resampling_string == "Lanczos":
            return Image.LANCZOS
        if resampling_string == "Box":
            return Image.BOX
        if resampling_string == "Hamming":
            return Image.HAMMING
        raise ValueError(f"Resampling method of {resampling_string} not "
                         f"supported")

    def generate_qr(
            self,
            protocol,
            text,
            image_size,
            fill_hexcolor,
            back_hexcolor,
            error_correction,
            border,
            resampling,
            module_drawer
            ):
        resampling_method = self._select_resampling_method(resampling)
        error_level = self._get_error_correction_constant(error_correction)
        self.update_text(protocol, text)
        qr = qrcode.QRCode(
                error_correction=error_level,
                box_size=16,
                border=border)
        qr.add_data(self.text)
        img = self._make_qr(qr, fill_hexcolor, back_hexcolor, module_drawer)
        img = img.resize((image_size, image_size), resample=resampling_method)
        return (self._img_to_tensor(img), qr.version)


class QRByModuleSize(QRBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (["Http", "Https", "None"], {"default": "Https"}),
                "text": ("STRING", {"multiline": True}),
                "module_size": ("INT", {"default": 16,
                                        "min": 1,
                                        "max": 64,
                                        "step": 1}),
                "max_image_size": ("INT", {"default": 512,
                                           "min": 64,
                                           "max": 4096,
                                           "step": 64}),
                "fill_hexcolor": ("STRING", {"multiline": False,
                                             "default": "#000000"}),
                "back_hexcolor": ("STRING", {"multiline": False,
                                             "default": "#FFFFFF"}),
                "error_correction": (["Low", "Medium", "Quartile", "High"],
                                     {"default": "High"}),
                "border": ("INT", {"default": 1,
                                   "min": 0,
                                   "max": 100,
                                   "step": 1}),
                "module_drawer": (["Square",
                                   "Gapped square",
                                   "Circle",
                                   "Rounded",
                                   "Vertical bars",
                                   "Horizontal bars"
                                   ], {"default": "Square"})
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("QR_CODE", "QR_VERSION", "IMAGE_SIZE")

    def generate_qr(
            self,
            protocol,
            text,
            module_size,
            max_image_size,
            fill_hexcolor,
            back_hexcolor,
            error_correction,
            border,
            module_drawer
            ):
        self.update_text(protocol, text)
        error_level = self._get_error_correction_constant(error_correction)
        qr = qrcode.QRCode(
                error_correction=error_level,
                box_size=module_size,
                border=border)
        qr.add_data(self.text)
        img = self._make_qr(qr, fill_hexcolor, back_hexcolor, module_drawer)
        self._validate_qr_size(img.pixel_size, max_image_size)
        return (self._img_to_tensor(img), qr.version, img.pixel_size)


class QRByModuleSizeSplitFunctionPatterns(QRBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (["Http", "Https", "None"], {"default": "Https"}),
                "text": ("STRING", {"multiline": True}),
                "module_size": ("INT", {"default": 16,
                                        "min": 1,
                                        "max": 64,
                                        "step": 1}),
                "max_image_size": ("INT", {"default": 512,
                                           "min": 64,
                                           "max": 4096,
                                           "step": 64}),
                "fill_hexcolor": ("STRING", {"multiline": False,
                                             "default": "#000000"}),
                "back_hexcolor": ("STRING", {"multiline": False,
                                             "default": "#FFFFFF"}),
                "error_correction": (["Low", "Medium", "Quartile", "High"],
                                     {"default": "High"}),
                "border": ("INT", {"default": 1,
                                   "min": 0,
                                   "max": 100,
                                   "step": 1}),
                "module_drawer": (["Square",
                                   "Gapped square",
                                   "Circle",
                                   "Rounded",
                                   "Vertical bars",
                                   "Horizontal bars"
                                   ], {"default": "Square"})
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("QR_CODE",
                    "MODULE_LAYER",
                    "FINDER_LAYER",
                    "FINDER_MASK",
                    "QR_VERSION",
                    "IMAGE_SIZE")

    def _generate_finder_pattern_ranges(self, module_size, border_size):
        outer = module_size * border_size
        inner = 7 * module_size + outer
        # Alternate behavior is required to prevent bugs from 0 border_size.
        far_outer = -outer if border_size else None
        return [
            (outer, inner, outer, inner),
            (outer, inner, -inner, far_outer),
            (-inner, far_outer, outer, inner)
        ]

    def _generate_finder_pattern_mask(self,
                                      pixel_size,
                                      module_size,
                                      border_size):
        mask = np.zeros((pixel_size, pixel_size), dtype=bool)
        for (x_min,
             x_max,
             y_min,
             y_max) in self._generate_finder_pattern_ranges(module_size,
                                                            border_size):
            mask[y_min:y_max, x_min:x_max] = True
        return mask

    def _apply_fill_to_mask(self, img, mask):
        array = np.array(img).copy()
        indices = np.nonzero(mask)
        array[indices[0], indices[1], :] = self.back
        return Image.fromarray(array)

    def _mask_to_tensor(self, mask):
        out_image = mask.astype(np.float32)
        return torch.from_numpy(out_image).unsqueeze(0)

    def generate_qr(
            self,
            protocol,
            text,
            module_size,
            max_image_size,
            fill_hexcolor,
            back_hexcolor,
            error_correction,
            border,
            module_drawer
            ):
        self.update_text(protocol, text)
        error_level = self._get_error_correction_constant(error_correction)
        qr = qrcode.QRCode(
                error_correction=error_level,
                box_size=module_size,
                border=border)
        qr.add_data(self.text)
        img = self._make_qr(qr, fill_hexcolor, back_hexcolor, module_drawer)
        pixel_size = img.pixel_size
        self._validate_qr_size(pixel_size, max_image_size)
        mask = self._generate_finder_pattern_mask(pixel_size,
                                                  module_size,
                                                  border)
        module_image = self._apply_fill_to_mask(img, mask)
        function_image = self._apply_fill_to_mask(img, ~mask)
        return (
            self._img_to_tensor(img),
            self._img_to_tensor(module_image),
            self._img_to_tensor(function_image),
            self._mask_to_tensor(mask),
            qr.version,
            pixel_size,
            )


class QRErrorMasker:
    def __init__(self):
        self.module_size = None
        self.canvas_shape = None
        self.qr_bounds = None

    FUNCTION = "find_qr_errors"
    CATEGORY = "ComfyQR"
    RETURN_TYPES = ("MASK", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("QR_ERROR_MASK", "PERCENT_ERROR", "CORRELATION", "RMSE")
    OUTPUT_IS_LIST = (False, True, True, True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_qr": ("IMAGE",),
                "modified_qr": ("IMAGE",),
                "module_size": ("INT", {"default": 16,
                                        "min": 1,
                                        "max": 64,
                                        "step": 1}),
                "grayscale_method": (["mean", "luminance"],
                                     {"default": "luminance"}),
                "aggregate_method": (["mean",], {"default": "mean"}),
                "evaluate": (["full_qr", "module_pattern", "finder_pattern"],
                             {"default": "module_pattern"}),
                "error_difficulty": ("FLOAT", {"default": 0,
                                               "min": 0,
                                               "max": 1,
                                               "step": .01}),
                "inverted_pattern": ("BOOLEAN", {"default": False}),
                "gamma": ("FLOAT", {"default": 2.2,
                                    "min": .1,
                                    "max": 2.8,
                                    "step": .1}),
            },
        }

    def _get_qr_bounds(self, tensor, invert):
        module_color = 1.0 if invert else 0.0
        module_pixels = (tensor == module_color)
        indices = torch.nonzero(module_pixels, as_tuple=True)
        # The viewer patterns will guarentee a module pixel in the upper left
        # The bottom right does not have that guarentee so max is used.
        return (indices[0][0],
                indices[0].max() + 1,
                indices[1][0], indices[1].max() + 1)

    def _extract_pattern_from_bounds(self, tensor):
        return tensor[self.qr_bounds[0]: self.qr_bounds[1],
                      self.qr_bounds[2]: self.qr_bounds[3]]

    def _trim_to_qr_area(self, source_qr, modified_qr, inverted_pattern):
        self.qr_bounds = self._get_qr_bounds(source_qr, inverted_pattern)
        self._check_bounds_and_module_size()
        source_qr = self._extract_pattern_from_bounds(source_qr)
        modified_qr = self._extract_pattern_from_bounds(modified_qr)
        return source_qr, modified_qr

    def _reshape_tensor_to_modules(self, tensor):
        if len(tensor.shape) != 2:
            raise RuntimeError("Module reshaping requires a 2 dimensional "
                               "array.")
        length = tensor.shape[0] // self.module_size
        reshaped_tensor = tensor.view(length,
                                      self.module_size,
                                      length,
                                      self.module_size)
        rehaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()
        return rehaped_tensor.view(length, length, self.module_size ** 2)

    def _check_bounds_and_module_size(self):
        height = self.qr_bounds[1] - self.qr_bounds[0]
        width = self.qr_bounds[3] - self.qr_bounds[2]
        color_warning = "Make sure that qr_fill and back colors have exact "
        "#FFFFFFF and #000000 values (and that module color values do not "
        "occur outside the QR) and invert is set correctly."
        if width != height:
            raise RuntimeError(f"Source QR dimensions are {width} x {height}. "
                               f"They must be a perfect square. "
                               f"{color_warning}")
        if width % self.module_size:
            raise RuntimeError(f"QR width of {width} does not fit module_size "
                               f"of {self.module_size}. It must be perfectly "
                               f"divisible. {color_warning}")

    def _check_equal_shape(self, source_qr, modified_qr):
        if source_qr.shape != modified_qr.shape:
            raise ValueError("Source and modified QR must have the same batch "
                             "size and dimensions.")

    def _squeeze_by_mean(self, tensor):
        return torch.mean(tensor, dim=-1)

    def _gamma_expansion(self, tensor, gamma):
        if gamma == 1:
            return tensor
        if gamma == 2.2:
            return torch.where(tensor <= 0.04045,
                               tensor / 12.92,
                               (tensor + 0.055) / 1.055) ** 2.4
        return tensor ** gamma

    def _gamma_compression(self, tensor, gamma):
        if gamma == 1:
            return tensor
        if gamma == 2.2:
            return torch.where(tensor <= .0031308,
                               tensor * 12.92,
                               1.055 * tensor ** (1/2.4) - 0.055)
        return tensor ** (1/gamma)

    def _grayscale_by_luminance(self, tensor, gamma):
        weights = torch.tensor([0.2125, 0.7154, 0.0721], dtype=torch.float32)
        tensor = self._gamma_expansion(tensor, gamma)
        tensor = tensor @ weights
        if gamma != 1:
            tensor = tensor ** gamma
        return self._gamma_compression(tensor, gamma)

    def _squeeze_to_modules(self, tensor, method):
        tensor = self._reshape_tensor_to_modules(tensor)
        if method == "mean":
            return self._squeeze_by_mean(tensor)
        raise RuntimeError("Module aggregation currently only supports the "
                           "mean.")

    def _reduce_to_modules(
            self,
            source_qr,
            modified_qr,
            module_size,
            grayscale_method,
            aggregate_method,
            inverted_pattern,
            gamma
            ):
        self.module_size = module_size
        self.canvas_shape = (source_qr.shape[0], source_qr.shape[1])
        # Processed first for simplified indexing of QR bounds.
        source_qr = self._squeeze_by_mean(source_qr)
        source_qr, modified_qr = self._trim_to_qr_area(source_qr,
                                                       modified_qr,
                                                       inverted_pattern
                                                       )
        if grayscale_method == "mean":
            modified_qr = self._squeeze_by_mean(modified_qr)
        elif grayscale_method == "luminance":
            modified_qr = self._grayscale_by_luminance(modified_qr, gamma)
        else:
            raise ValueError("Currently only mean is supported for rgb to "
                             "grayscale conversion.")
        source_qr = torch.round(self._squeeze_to_modules(source_qr, "mean"))
        modified_qr = self._squeeze_to_modules(modified_qr, aggregate_method)
        return source_qr, modified_qr

    def _create_finder_pattern_mask(self, width, inverted):
        mask = torch.zeros((width, width), dtype=torch.bool)
        # When borders are trimmed and QR code has module size of 1, results
        #  are consistent.
        finder_coords = [[0, 7, 0, 7], [0, 7, -7, None], [-7, None, 0, 7]]
        for x_min, x_max, y_min, y_max in finder_coords:
            mask[y_min:y_max, x_min:x_max] = True
        return ~mask if inverted else mask

    def _create_qr_mask(self, tensor, evaluate):
        if evaluate == "module_pattern":
            return self._create_finder_pattern_mask(tensor, True)
        if evaluate == "finder_pattern":
            return self._create_finder_pattern_mask(tensor, False)
        return None

    def _bin_tensor_to_threshold(self, tensor, contrast_difficulty):
        tensor = tensor.clone()
        threshold = contrast_difficulty / 2
        # Since we are only interested in value matches and there is a clear
        # stable dividing line of .5, bringing in the other array is
        # unneccessary and the binning process can be simplified.
        bin_condition = (tensor + threshold <= .5) & (tensor != .5)
        tensor[bin_condition] = 0.0
        bin_condition = (tensor - threshold >= .5) & (tensor != .5)
        tensor[bin_condition] = 1.0
        return tensor

    def _replace_qr_to_canvas(self, tensor):
        length = tensor.shape[0] * self.module_size
        bounds = self.qr_bounds
        tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0),
                               size=(length, length),
                               mode='nearest')
        canvas = torch.zeros(self.canvas_shape, dtype=torch.float32)
        canvas[bounds[0]:bounds[1], bounds[2]:bounds[3]] = tensor.squeeze()
        return canvas

    def _compare_modules(
            self,
            source_qr,
            modified_qr,
            mask,
            error_difficulty
            ):
        modified_qr = self._bin_tensor_to_threshold(modified_qr,
                                                    error_difficulty)
        error = source_qr != modified_qr
        percent_error = error[mask].sum().item() / error[mask].numel()
        if mask is not None:
            error[~mask] = False
        return (self._replace_qr_to_canvas((error).to(torch.float32)),
                percent_error)

    def _qr_correlation(self, source_qr, modified_qr, mask):
        source_qr = source_qr[mask].numpy().reshape((-1))
        modified_qr = modified_qr[mask].numpy().reshape((-1))
        return np.corrcoef(source_qr, modified_qr)[0, 1]

    def _qr_rmse(self, source_qr, modified_qr, mask):
        diff = source_qr[mask].numpy() - modified_qr[mask].numpy()
        return np.sqrt((diff ** 2).mean())

    def find_qr_errors(
            self,
            source_qr,
            modified_qr,
            module_size,
            grayscale_method,
            aggregate_method,
            evaluate,
            error_difficulty,
            inverted_pattern,
            gamma,
            ):
        self._check_equal_shape(source_qr, modified_qr)
        error_masks, error_percents, correlations, rmses = [], [], [], []
        for i in range(source_qr.shape[0]):
            qr_s, qr_m = source_qr[i], modified_qr[i]
            qr_s, qr_m = self._reduce_to_modules(qr_s,
                                                 qr_m,
                                                 module_size,
                                                 grayscale_method,
                                                 aggregate_method,
                                                 inverted_pattern,
                                                 gamma
                                                 )
            mask = self._create_qr_mask(qr_s.shape[0], evaluate)
            error_mask, percent_error = self._compare_modules(qr_s,
                                                              qr_m,
                                                              mask,
                                                              error_difficulty
                                                              )
            correlation = self._qr_correlation(qr_s, qr_m, mask)
            rmse = self._qr_rmse(qr_s, qr_m, mask)
            error_masks.append(error_mask)
            error_percents.append(percent_error)
            correlations.append(correlation)
            rmses.append(rmse)
        error_masks = torch.stack(error_masks, dim=0)
        return (error_masks, error_percents, correlations, rmses)


NODE_CLASS_MAPPINGS = {
                       "comfy-qr-by-module-size": QRByModuleSize,
                       "comfy-qr-by-image-size": QRByImageSize,
                       "comfy-qr-by-module-split":
                       QRByModuleSizeSplitFunctionPatterns,
                       "comfy-qr-mask_errors": QRErrorMasker,
                       }


NODE_DISPLAY_NAME_MAPPINGS = {
                              "comfy-qr-by-module-size": "QR Code",
                              "comfy-qr-by-image-size": "QR Code (Conformed "
                              "to Image Size)",
                              "comfy-qr-by-module-split": "QR Code (Split)",
                              "comfy-qr-mask_errors": "Mask QR Errors",
                              }
