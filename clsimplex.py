import pyopencl as cl
import numpy
import sys
import datetime

class NoiseGenerator(object):
    def __init__(self, seed=None):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.seed = seed

        numpy.random.seed(seed)
        self._compute_seed()

    def _compute_seed(self):
        self.perm = numpy.array([i for i in range(256)], dtype=numpy.uint8)

        numpy.random.shuffle(self.perm)
        self.perm = numpy.concatenate([self.perm, self.perm])

    def _load_program(self, filename):
        with open(filename, 'r') as f:
            fstr = "".join(f.readlines())
        self.program = cl.Program(self.ctx, fstr).build()

    def noise3d(self, offset=(0,0,0), block_dim=(256,256,256)):
        self._load_program('simplex.cl')

        chunk_size = block_dim[0] * block_dim[1] * block_dim[2]
        global_ws = (chunk_size,)
        local_ws = None

        mf = cl.mem_flags
        res = numpy.empty(shape=global_ws, dtype=numpy.float32)
        res_d = cl.Buffer(self.ctx, mf.WRITE_ONLY, numpy.float32(1).nbytes*chunk_size)
        perm_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.perm)

        event = self.program.sdnoise3(
            self.queue, global_ws, local_ws,
            numpy.float32(offset[0]), numpy.float32(offset[1]), numpy.float32(offset[2]),
            numpy.uint32(block_dim[0]), numpy.uint32(block_dim[1]), numpy.uint32(block_dim[2]),
            res_d, perm_d
        )

        cl.enqueue_read_buffer(self.queue, res_d, res).wait()
        return res

def print_chunk(chunk, xdim, ydim, zdim):
    for z in range(zdim):
        #sys.stdout.write('\n\n')

        for x in range(xdim):
            sys.stdout.write('\n')

            for y in range(ydim):
                val = chunk[x + y*ydim + z*(zdim**2)]
                if val > 0: sys.stdout.write(' . ')
                else: sys.stdout.write(' # ')

if __name__ == '__main__':
    gen = NoiseGenerator()

    start = datetime.datetime.now()
    noise = gen.noise3d()
    elapsed = datetime.datetime.now() - start

    print_chunk(noise, 32, 32, 32)
    print("\nElapsed time: {} seconds".format(elapsed.total_seconds()))

