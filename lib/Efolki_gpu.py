import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def conv2SepMatlabbis(I, fen):

    rad = int((fen.size-1)/2)
    ligne = np.zeros((rad, I.shape[1]))
    I = np.append(ligne, I, axis=0)
    I = np.append(I, ligne, axis=0)

    colonne = np.zeros((I.shape[0], rad))
    I = np.append(colonne, I, axis=1)
    I = np.append(I, colonne, axis=1)

    res = conv2bis(conv2bis(I, fen.T), fen)
    return res

def EFolkiIter(I0, I1, iteration=5, radius=[8, 4], rank=4, uinit=None,vinit=None):
    talon=1.e-8
    if rank > 0:
        I0 = rank_filter_sup(I0, rank)
        I1 = rank_filter_sup(I1, rank)

    if uinit is None:
        u = np.zeros(I0.shape)
    else:
        u = uinit
    if vinit is None:
        v = np.zeros(I1.shape)
    else:
        v = vinit

    Iy, Ix = np.gradient(I0)

    cols, rows = I0.shape[1], I0.shape[0]
    x, y = np.meshgrid(range(cols), range(rows))

    for rad in radius:

        burt1D = np.array(np.ones([1, 2*rad+1]))/(2*rad + 1)
        W = lambda x: conv2SepMatlabbis(x, burt1D)

        Ixx = W(Ix*Ix) + talon
        Iyy = W(Iy*Iy) + talon
        Ixy = W(Ix*Iy)
        D = Ixx*Iyy - Ixy**2

        for i in range(iteration):
            i1w = interp2(I1, x+u, y+v)

            it = I0 - i1w + u*Ix + v*Iy
            Ixt = W(Ix * it)
            Iyt = W(Iy * it)
            u = (Iyy * Ixt - Ixy * Iyt)/D
            v = (Ixx * Iyt - Ixy * Ixt)/D
            unvalid = np.isnan(u) | np.isinf(u) | np.isnan(v) | np.isinf(v)
            u[unvalid] = 0
            v[unvalid] = 0
    return u, v

kernel_code = """
__global__ void gradientKernel(...) {
    // Implement gradient calculation
}
"""

mod = SourceModule(kernel_code)
gradient_kernel = mod.get_function("gradientKernel")

I0 = ...
u = np.zeros(I0.shape, dtype=np.float32)

I0_gpu = cuda.mem_alloc(I0.nbytes)
cuda.memcpy_htod(I0_gpu, I0)

block_size = (16, 16, 1)
grid_size = (int(np.ceil(I0.shape[0] / block_size[0])), int(np.ceil(I0.shape[1] / block_size[1])))


gradient_kernel(I0_gpu, ... , block=block_size, grid=grid_size)

cuda.memcpy_dtoh(u, u_gpu)

I0_gpu.free()
