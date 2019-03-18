import torch
import numpy as np
from PIL import Image
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


with open('cuda_utils.cu', 'r') as f:
    SOURCE = content_file.read()


def smooth_local_affine(output_cpu, input_cpu, epsilon, patch, h, w, f_r, f_e):

    program = Program(SOURCE, 'best_local_affine_kernel.cu')
    ptx = program.compile(['-I/usr/local/cuda/include'])
    m = function.Module()
    m.load(bytes(ptx.encode()))

    _reconstruction_best_kernel = m.get_function('reconstruction_best_kernel')
    _bilateral_smooth_kernel = m.get_function('bilateral_smooth_kernel')
    _best_local_affine_kernel = m.get_function('best_local_affine_kernel')
    stream = namedtuple('Stream', ['ptr'])
    s = stream(ptr=torch.cuda.current_stream().cuda_stream)

    filter_radius = f_r
    sigma1 = filter_radius / 3
    sigma2 = f_e
    radius = (patch - 1) / 2

    filtered_best_output = torch.zeros(np.shape(input_cpu)).cuda()
    affine_model = torch.zeros((h * w, 12)).cuda()
    filtered_affine_model = torch.zeros((h * w, 12)).cuda()

    input_ = torch.from_numpy(input_cpu).cuda()
    output_ = torch.from_numpy(output_cpu).cuda()

    _best_local_affine_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[
            output_.data_ptr(), input_.data_ptr(), affine_model.data_ptr(),
            np.int32(h), np.int32(w), np.float32(epsilon), np.int32(radius)
        ], stream=s
     )

    _bilateral_smooth_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[
            affine_model.data_ptr(), filtered_affine_model.data_ptr(), input_.data_ptr(),
            np.int32(h), np.int32(w), np.int32(f_r), np.float32(sigma1), np.float32(sigma2)
        ], stream=s
    )

    _reconstruction_best_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[
            input_.data_ptr(), filtered_affine_model.data_ptr(),
            filtered_best_output.data_ptr(), np.int32(h), np.int32(w)
        ], stream=s
    )

    numpy_filtered_best_output = filtered_best_output.cpu().numpy()
    return numpy_filtered_best_output


def secondary_smoothing(Y, content, f_radius=15, f_edge=1e-1):
    """
    Arguments:
        Y, content: numpy uint8 arrays with shape [h, w, 3].
        f_radius:
        f_edge:
    Returns:
        a numpy uint8 array with shape [h, w, 3].
    """

    Y = Y.astype('float32')
    h, w, c = Y.shape
    Y = Y.transpose((2, 0, 1))

    content = cv2.resize(content, (w, h))
    content = content.astype('float32')
    content = content.transpose((2, 0, 1))

    input_ = np.ascontiguousarray(content) / 255.0
    output_ = np.ascontiguousarray(Y) / 255.0

    best_ = smooth_local_affine(output_, input_, 1e-7, 3, h, w, f_radius, f_edge)
    best_ = best_.transpose(1, 2, 0)
    result = np.clip(best_ * 255.0, 0, 255.0).astype('uint8')
    return result
