import pyopencl as cl
import numpy
import sys
import datetime

class NoiseGenerator(object):
    def __init__(self, block_dim=None):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        if block_dim: self.block_dim = block_dim
        else: self.block_dim = 256, 256, 256

    def load_program(self, filename):
        with open(filename, 'r') as f:
            fstr = "".join(f.readlines())
        self.program = cl.Program(self.ctx, fstr).build()

    def noise3d(self, xoff=0, yoff=0, zoff=0):
        self.load_program('simplex.cl')

        chunk_size = self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
        global_ws = (chunk_size,)
        local_ws = None

        mf = cl.mem_flags
        res = numpy.empty(shape=global_ws, dtype=numpy.float32)
        res_d = cl.Buffer(self.ctx, mf.WRITE_ONLY, numpy.float32(1).nbytes*chunk_size)

        event = self.program.sdnoise3(
            self.queue, global_ws, local_ws,
            numpy.float32(xoff), numpy.float32(yoff), numpy.float32(zoff),
            numpy.uint32(self.block_dim[0]), numpy.uint32(self.block_dim[1]), numpy.uint32(self.block_dim[2]),
            res_d
        )

        cl.enqueue_read_buffer(self.queue, res_d, res).wait()
        return res

def print_chunk(chunk, xdim, ydim, zdim):
    for z in range(zdim):
        sys.stdout.write('\n\n')

        for x in range(xdim):
            sys.stdout.write('\n')

            for y in range(ydim):
                val = chunk[x + y*ydim + z*(zdim**2)]
                if val > 0: sys.stdout.write(' # ')
                else: sys.stdout.write(' . ')

if __name__ == '__main__':
    gen = NoiseGenerator()
    noise = gen.noise3d()

    print_chunk(noise, 32, 32, 32)

