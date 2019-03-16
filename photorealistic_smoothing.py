import numpy as np
import scipy.sparse


def photorealistic_smoothing(X, Y, lambda_weight=1e-4):
    """
    Arguments:
        X: a numpy uint8 array with shape [h, w, c].
        Y: a numpy uint8 array with shape [h', w', c].
        lambda_weight: a float number.
    Returns:
        a numpy uint8 array with shape [h - 4, w - 4, c].
    """
    h, w, c = Y.shape
    assert c == 3

    Y = Y[2:(h - 2), 2:(w - 2)]
    Y = np.pad(Y, pad_width=((2, 2), (2, 2), (0, 0)), mode='edge')
    Y = np.reshape(Y, [h * w, c])
    Y = Y.astype('float64')/255.0  # shape [h * w, c]

    X = cv2.resize(X, (w - 4, h - 4))
    X = np.pad(X, pad_width=((2, 2), (2, 2), (0, 0)), mode='edge')
    X = X.astype('float64')/255.0  # shape [h, w, c]

    N = h * w  # the number of pixels
    W = compute_laplacian(X)
    W = W.tocsc()  # shape [N, N]
    d = W.sum(0).A.squeeze(0)  # shape [N]
    d = np.pow(d, -0.5)
    D = scipy.sparse.csc_matrix((d, (np.arange(0, N), np.arange(0, N))), shape=(N, N))
    S = D.dot(W).dot(D)

    alpha = 1/(1 + lambda_weight)
    A = scipy.sparse.identity(N, dtype='float64') - alpha * S
    solver = scipy.sparse.linalg.factorized(A.tocsc())

    R = np.zeros([N, c], dtype='float64')
    R[:, 0] = solver(Y[:, 0])
    R[:, 1] = solver(Y[:, 1])
    R[:, 2] = solver(Y[:, 2])

    # so, it is true that
    # R[:, i] = (I - alpha * S)^{-1} * Y[:, i]

    R *= (1 - alpha)
    R = R.reshape(h, w, c)
    R = R[2:(h - 2), 2:(w - 2)]

    R = 255.0 * np.clip(V, 0.0, 1.0)
    R = R.astype('uint8')
    return R


def compute_laplacian(image, epsilon=1e-7, size=3):
    """
    This function computes the matting Laplacian from
    paper "A Closed Form Solution to Natural Image Matting".

    Arguments:
        image: a numpy double array with shape [h, w, c].
        epsilon: a float number.
        size: an integer, size of a patch.
    Returns:
        a sparse double matrix with shape [N, N].
        Where N = h * w (the number of pixels).
    """
    area = size * size
    h, w, _ = image.shape

    windows, indices = get_patches(image, size)
    # they have shapes [p, q, area, c] and [p, q, area].

    # `indices` has values in range [0, h * w - 1],
    # if image = image.reshape(h * w, c) then
    # windows[i, j, k, :] = image[indices[i, j, k], :]

    mean = np.mean(windows, axis=2, keepdims=True)  # shape [p, q, 1, c]
    windows -= mean

    var = np.einsum('...ji,...jk->...ik', windows, windows)/area
    # it has shape [p, q, c, c]

    x = np.linalg.inv(var + (epsilon/area) * np.eye(c))  # shape [p, q, c, c]
    x = np.einsum('...ij,...jk->...ik', windows, x)  # shape [p, q, area, c]
    x = np.einsum('...ij,...kj->...ik', x, windows)  # shape [p, q, area, area]
    x = (1/area)*(1 + x)

    row = np.tile(np.expand_dims(indices, 3), (1, 1, 1, area))  # shape [p, q, area, area]
    col = np.tile(np.expand_dims(indices, 2), (1, 1, area, 1))  # shape [p, q, area, area]

    row = row.ravel()
    col = col.ravel()
    values = x.ravel()
    # they all have shape [p * q * area * area]

    # duplicate entries will be summed together
    return scipy.sparse.coo_matrix((values, (row, col)), shape=(h * w, h * w))


def get_patches(image, size):
    """
    Extracts patches with stride 1.

    Arguments:
        image: a numpy array with shape [h, w, c].
        size: an integer.
    Returns:
        patches: a numpy array with shape [p, q, size * size, c].
            Where p = h - size + 1 and q = w - size + 1.
            p * q is the number of patches.
        indices: a numpy long array with shape [p, q, size * size].
    """
    h, w, c = image.shape

    x = np.arange(h * w).reshape(h, w)
    shape = (h - size + 1, w - size + 1, size, size)
    strides = x.strides + x.strides

    indices = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    p, q, _, _ = indices.shape
    indices = indices.reshape(p, q, size * size)

    image = image.reshape(h * w, c)
    patches = image[indices]
    return patches, indices
