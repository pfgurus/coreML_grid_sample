import torch
import torch.nn.functional as F


def grid_sample(input, grid):
    """
    Exportable grid sample.

    See:
    https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    https://stackoverflow.com/questions/52888146/what-is-the-equivalent-of-torch-nn-functional-grid-sample-in-tensorflow-numpy
    https://en.wikipedia.org/wiki/Bilinear_interpolation
    """

    b, c, hi, wi = input.shape

    hw = grid.new_tensor((hi, wi)[-1::-1])
    grid = ((grid + 1) * hw - 1) / 2

    grid0 = torch.floor(grid).type(torch.int32)
    grid1 = grid0 + 1

    grid0 = grid0.clamp(0, wi + 1)
    grid1 = grid1.clamp(0, wi + 1)

    x = grid[..., 0]
    y = grid[..., 1]

    x0 = grid0[..., 0]
    x1 = grid1[..., 0]
    y0 = grid0[..., 1]
    y1 = grid1[..., 1]

    w00 = (x - x0) * (y - y0)
    w10 = (x1 - x) * (y - y0)
    w01 = (x - x0) * (y1 - y)
    w11 = (x1 - x) * (y1 - y)

    bi = torch.arange(b).unsqueeze(-1).unsqueeze(-1).expand(b, hi, wi).type(torch.int32)

    input = F.pad(input, (1, 1, 1, 1))
    input = input.permute((0, 2, 3, 1))

    i00 = input[bi, y0, x0]
    i01 = input[bi, y1, x0]
    i10 = input[bi, y0, x1]
    i11 = input[bi, y1, x1]

    result = i00 * w11.unsqueeze_(-1) + i01 * w10.unsqueeze_(-1) + i10 * w01.unsqueeze_(-1) + i11 * w00.unsqueeze_(-1)
    result = result.permute((0, 3, 1, 2))
    return result

