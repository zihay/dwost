"""Image I/O utilities for reading and writing various image formats.

This module provides functions for reading and writing images in various formats,
including PNG, JPEG, EXR, and HDR. It handles color space conversions between
linear and sRGB color spaces.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import imageio
import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as im
from skimage.transform import resize

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# sRGB <-> Linear conversion constants (IEC 61966-2-1)
_SRGB_LINEAR_THRESHOLD = 0.0031308
_SRGB_INVERSE_THRESHOLD = 0.04045


def color_map(data: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """Apply a colormap to scalar data.

    Args:
        data: Input scalar data array.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.

    Returns:
        RGBA array with colormap applied.
    """
    my_cm = matplotlib.colormaps.get_cmap("viridis")
    normed_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    return my_cm(normed_data)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to sRGB color space.

    Args:
        linear: Linear RGB values in range [0, 1].

    Returns:
        sRGB values in range [0, 1].
    """
    srgb = np.zeros_like(linear)
    mask = linear <= _SRGB_LINEAR_THRESHOLD
    srgb[mask] = linear[mask] * 12.92
    srgb[~mask] = 1.055 * (linear[~mask] ** (1.0 / 2.4)) - 0.055
    return srgb


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values to linear RGB color space.

    Args:
        srgb: sRGB values in range [0, 1].

    Returns:
        Linear RGB values in range [0, 1].
    """
    linear = np.zeros_like(srgb)
    mask = srgb <= _SRGB_INVERSE_THRESHOLD
    linear[mask] = srgb[mask] / 12.92
    linear[~mask] = ((srgb[~mask] + 0.055) / 1.055) ** 2.4
    return linear


def to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert image from linear to sRGB color space.

    Args:
        image: Image in linear color space.

    Returns:
        Image in sRGB color space, clipped to [0, 1].
    """
    return np.clip(linear_to_srgb(to_numpy(image)), 0, 1)


def to_linear(image: np.ndarray) -> np.ndarray:
    """Convert image from sRGB to linear color space.

    Args:
        image: Image in sRGB color space.

    Returns:
        Image in linear color space.
    """
    return srgb_to_linear(to_numpy(image))


def to_numpy(data: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert data to numpy array.

    Handles both numpy arrays and PyTorch tensors.

    Args:
        data: Input data (numpy array or torch tensor).

    Returns:
        Numpy array.
    """
    if HAS_TORCH and torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.array(data)


def read_image(
    image_path: Union[str, Path], is_srgb: Optional[bool] = None
) -> np.ndarray:
    """Read an image from disk.

    Automatically handles color space conversion based on file extension.
    HDR formats (EXR, HDR, RGBE) are assumed to be linear; LDR formats
    are assumed to be sRGB.

    Args:
        image_path: Path to the image file.
        is_srgb: Override automatic color space detection. If True, convert
            from sRGB to linear. If False, keep as-is. If None, auto-detect.

    Returns:
        Image as a numpy array in linear color space with shape (H, W, C).
    """
    image_path = Path(image_path)
    image = iio.imread(image_path)
    image = np.atleast_3d(image)

    # Normalize integer types to float
    if image.dtype in (np.uint8, np.int16):
        image = image.astype(np.float32) / 255.0
    elif image.dtype in (np.uint16, np.int32):
        image = image.astype(np.float32) / 65535.0

    # Auto-detect color space from extension
    if is_srgb is None:
        is_srgb = image_path.suffix.lower() not in {".exr", ".hdr", ".rgbe"}

    if is_srgb:
        image = to_linear(image)

    return image


def read_png(png_path: Union[str, Path], is_srgb: bool = True) -> np.ndarray:
    """Read a PNG image file.

    Args:
        png_path: Path to the PNG file.
        is_srgb: Whether to convert from sRGB to linear color space.

    Returns:
        Image as numpy array with shape (H, W, 3) in linear color space.
    """
    image = iio.imread(png_path, extension=".png")
    if image.dtype in (np.uint8, np.int16):
        image = image.astype(np.float32) / 255.0
    elif image.dtype in (np.uint16, np.int32):
        image = image.astype(np.float32) / 65535.0

    # Handle 4D arrays (batch dimension)
    if len(image.shape) == 4:
        image = image[0]

    # Extract RGB channels only
    if len(image.shape) == 3:
        image = image[:, :, :3]

    return to_linear(image) if is_srgb else image


def read_exr(exr_path: Union[str, Path]) -> np.ndarray:
    """Read an EXR (OpenEXR) image file.

    Args:
        exr_path: Path to the EXR file.

    Returns:
        Image as numpy array with shape (H, W, C).
    """
    image = iio.imread(exr_path, extension=".exr")
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    return image


def write_image(
    image_path: Union[str, Path],
    image: np.ndarray,
    is_srgb: Optional[bool] = None,
) -> None:
    """Write an image to disk.

    Automatically handles color space conversion based on file extension.
    HDR formats are written as float32; LDR formats are written as uint8.

    Args:
        image_path: Output path for the image.
        image: Image array in linear color space with shape (H, W) or (H, W, C).
        is_srgb: Override automatic color space conversion. If True, convert
            to sRGB before writing. If False, write linear values. If None,
            auto-detect from extension.
    """
    image_path = Path(image_path)
    image_ext = image_path.suffix.lower()

    # Plugin and flag configuration
    iio_plugins = {
        ".exr": "EXR-FI",
        ".hdr": "HDR-FI",
        ".png": "PNG-FI",
    }
    iio_flags = {
        ".exr": imageio.plugins.freeimage.IO_FLAGS.EXR_NONE,
    }
    hdr_formats = {".exr", ".hdr", ".rgbe"}

    image = to_numpy(image)
    image = np.atleast_3d(image)

    # Ensure at least 3 channels for compatibility
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    # Auto-detect color space conversion
    if is_srgb is None:
        is_srgb = image_ext not in hdr_formats

    if is_srgb:
        image = to_srgb(image)

    # Convert to appropriate dtype
    if image_ext in hdr_formats:
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)

    flags = iio_flags.get(image_ext, 0)
    iio.imwrite(image_path, image, flags=flags, plugin=iio_plugins.get(image_ext))


def write_png(png_path: Union[str, Path], image: np.ndarray) -> None:
    """Write an image as PNG file.

    Args:
        png_path: Output path for the PNG file.
        image: Image array in linear color space.
    """
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    iio.imwrite(png_path, image, extension=".png")


def write_jpg(jpg_path: Union[str, Path], image: np.ndarray) -> None:
    """Write an image as JPEG file.

    Args:
        jpg_path: Output path for the JPEG file.
        image: Image array in linear color space.
    """
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    rgb_im = im.fromarray(image).convert("RGB")
    rgb_im.save(jpg_path, format="JPEG", quality=95)


def write_exr(exr_path: Union[str, Path], image: np.ndarray) -> None:
    """Write an image as EXR (OpenEXR) file.

    Creates parent directories if they don't exist.

    Args:
        exr_path: Output path for the EXR file.
        image: Image array in linear color space.

    Raises:
        AssertionError: If the file extension is not .exr.
    """
    exr_path = Path(exr_path)
    exr_path.parent.mkdir(parents=True, exist_ok=True)
    assert exr_path.suffix.lower() == ".exr", "File must have .exr extension"
    write_image(exr_path, image, is_srgb=False)


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image to the specified dimensions.

    Args:
        image: Input image array.
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        Resized image array.
    """
    return resize(image, (height, width))


def print_quartiles(image: np.ndarray) -> None:
    """Print the quartile values of an image.

    Args:
        image: Input image array.
    """
    percentiles = [0, 25, 50, 75, 100]
    values = [np.percentile(image, p) for p in percentiles]
    print(f"Quartiles: {values}")


def subplot(images: list, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Display multiple images in a horizontal subplot.

    Args:
        images: List of images to display.
        vmin: Minimum value for color mapping.
        vmax: Maximum value for color mapping.
    """
    n = len(images)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], vmin=vmin, vmax=vmax, cmap="viridis")
        plt.axis("off")


class FileStream:
    """Binary file stream reader with numpy dtype support."""

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize the file stream.

        Args:
            path: Path to the file to read.
        """
        self.path = Path(path)
        self.file = open(self.path, "rb")

    def __enter__(self) -> "FileStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.file.close()

    def read(self, count: int, dtype: np.dtype = np.byte) -> np.ndarray:
        """Read binary data from the file.

        Args:
            count: Number of elements to read.
            dtype: Numpy dtype of the elements.

        Returns:
            Numpy array of the read data.
        """
        data = self.file.read(count * np.dtype(dtype).itemsize)
        return np.frombuffer(data, dtype=dtype)


@dataclass
class ColorMap:
    """Configurable colormap for scientific visualization.

    Attributes:
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        cmap: Colormap name. Use "cubicL" for perceptually uniform colormap.
        remap: Whether to apply log-scale remapping for large dynamic ranges.
    """

    vmin: float = -2.0
    vmax: float = 2.0
    cmap: str = "cubicL"
    remap: bool = False

    def __post_init__(self) -> None:
        """Load the cubicL colormap from file."""
        path = Path(__file__).parent / "cubicL.txt"
        self._cubicL = LinearSegmentedColormap.from_list(
            "cubicL", np.loadtxt(path), N=256
        )

    def __call__(self, value: np.ndarray) -> np.ndarray:
        """Apply the colormap to scalar values.

        Args:
            value: Input scalar values.

        Returns:
            RGBA values with colormap applied.
        """
        if self.remap:
            value = np.sign(value) * np.log1p(np.abs(value))

        cmap = self._cubicL if self.cmap == "cubicL" else self.cmap
        norm = matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        return mapper.to_rgba(value)
