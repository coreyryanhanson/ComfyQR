import numpy as np
import qrcode
from qrcode.compat.pil import Image
import torch


class QRBase:
    def __init__(self):
        self.text = ""
        self.fill = None
        self.back = None

    FUNCTION = "generate_qr"
    CATEGORY = "Comfy-QR"

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

    def _make_qr(self, qr, fill_hexcolor, back_hexcolor):
        self.fill = self._parse_hexcolor_string(fill_hexcolor, "fill_hexcolor")
        self.back = self._parse_hexcolor_string(back_hexcolor, "back_hexcolor")
        qr.make(fit=True)
        return qr.make_image(fill_color=self.fill, back_color=self.back)

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
            raise ValueError(f"{parameter} contains invalid hexadecimal characters")

    def _validate_qr_size(self, size, max_size):
        if size > max_size:
            raise RuntimeError(f"QR dimensions of {size} exceed max size of {max_size}.")

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
                "image_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "fill_hexcolor": ("STRING", {"multiline": False, "default": "#000000"}),
                "back_hexcolor": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "error_correction": (["Low", "Medium", "Quartile", "High"], {"default": "High"}),
                "border": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "resampling": (["Bicubic", "Bilinear", "Box", "Hamming", "Lanczos", "Nearest"], {"default": "Nearest"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "QR_VERSION")

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
        raise ValueError(f"Resampling method of {resampling_string} not supported")

    def generate_qr(
            self,
            protocol,
            text,
            image_size,
            fill_hexcolor,
            back_hexcolor,
            error_correction,
            border,
            resampling
            ):
        resampling_method = self._select_resampling_method(resampling)
        error_level = self._get_error_correction_constant(error_correction)
        self.update_text(protocol, text)
        qr = qrcode.QRCode(
                error_correction=error_level,
                box_size=1,
                border=border)
        qr.add_data(self.text)
        img = self._make_qr(qr, fill_hexcolor, back_hexcolor)
        self._validate_qr_size(img.pixel_size, image_size)
        img = img.resize((image_size, image_size), resample=resampling_method)
        return (self._img_to_tensor(img), qr.version)


class QRByModuleSize(QRBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (["Http", "Https", "None"], {"default": "Https"}),
                "text": ("STRING", {"multiline": True}),
                "module_size": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "max_image_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "fill_hexcolor": ("STRING", {"multiline": False, "default": "#000000"}),
                "back_hexcolor": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "error_correction": (["Low", "Medium", "Quartile", "High"], {"default": "High"}),
                "border": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "QR_VERSION", "IMAGE_SIZE")

    def generate_qr(
            self,
            protocol,
            text,
            module_size,
            max_image_size,
            fill_hexcolor,
            back_hexcolor,
            error_correction,
            border
            ):
        self.update_text(protocol, text)
        error_level = self._get_error_correction_constant(error_correction)
        qr = qrcode.QRCode(
                error_correction=error_level,
                box_size=module_size,
                border=border)
        qr.add_data(self.text)
        img = self._make_qr(qr, fill_hexcolor, back_hexcolor)
        self._validate_qr_size(img.pixel_size, max_image_size)
        return (self._img_to_tensor(img), qr.version, img.pixel_size)


class QRByModuleSizeSplitFunctionPatterns(QRBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (["Http", "Https", "None"], {"default": "Https"}),
                "text": ("STRING", {"multiline": True}),
                "module_size": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "max_image_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "fill_hexcolor": ("STRING", {"multiline": False, "default": "#000000"}),
                "back_hexcolor": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "error_correction": (["Low", "Medium", "Quartile", "High"], {"default": "High"}),
                "border": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("QR_FLATTENED", "MODULE_LAYER", "FUNCTION_LAYER", "FUNCTION_MASK", "QR_VERSION", "IMAGE_SIZE")

    def _generate_finder_pattern_ranges(self, module_size, border_size):
        outer = module_size * border_size
        inner = 7 * module_size + outer
        return [
            (outer, inner, outer, inner),
            (outer, inner, -inner, -outer),
            (-inner, -outer, outer, inner)
        ]

    def _generate_finder_pattern_mask(self, pixel_size, module_size, border_size):
        mask = np.zeros((pixel_size, pixel_size), dtype=bool)
        for x_min, x_max, y_min, y_max in self._generate_finder_pattern_ranges(module_size, border_size):
            mask[y_min:y_max, x_min:x_max] = True
        return mask

    def _apply_fill_to_mask(self, img, mask):
        array = np.array(img).copy()
        indices = np.nonzero(mask)
        array[indices[0], indices[1], :] = self.back
        return Image.fromarray(array)

    def _mask_to_tensor(self, mask):
        out_image = mask.astype(np.float32)
        return torch.from_numpy(out_image)

    def generate_qr(
            self,
            protocol,
            text,
            module_size,
            max_image_size,
            fill_hexcolor,
            back_hexcolor,
            error_correction,
            border
            ):
        self.update_text(protocol, text)
        error_level = self._get_error_correction_constant(error_correction)
        qr = qrcode.QRCode(
                error_correction=error_level,
                box_size=module_size,
                border=border)
        qr.add_data(self.text)
        img = self._make_qr(qr, fill_hexcolor, back_hexcolor)
        pixel_size = img.pixel_size
        self._validate_qr_size(pixel_size, max_image_size)
        mask = self._generate_finder_pattern_mask(pixel_size, module_size, border)
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


NODE_CLASS_MAPPINGS = {
                       "comfy-qr-by-module-size": QRByModuleSize,
                       "comfy-qr-by-image-size": QRByImageSize,
                       "comfy-qr-by-module-split": QRByModuleSizeSplitFunctionPatterns,
                       }


NODE_DISPLAY_NAME_MAPPINGS = {
                              "comfy-qr-by-module-size": "QR Code",
                              "comfy-qr-by-image-size": "QR Code (Conformed to Image Size)",
                              "comfy-qr-by-module-split": "QR Code (Split)",
                              }
