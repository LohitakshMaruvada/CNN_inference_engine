import numpy as np

class LineMemory:
    def __init__(self, row_size, n_outputs):
        """
        row_size  : length of one row of the input feature map
        n_outputs : number of parallel outputs to feed PEs
        """
        self.memory = [0] * row_size
        self.row_size = row_size
        self.n_outputs = n_outputs
        self.write_ptr = 0
        self.read_ptr = 0

    def reset(self):
        self.memory = [0] * self.row_size
        self.write_ptr = 0
        self.read_ptr = 0

    def write(self, data):
        """Sequential write like WAG"""
        if self.write_ptr < self.row_size:
            self.memory[self.write_ptr] = data
            self.write_ptr += 1

    def read_stride(self, stride_size=1):
        """
        Provide n_outputs data for one stride.
        Old data reused, new inserted.
        Returns None if not enough data.
        """
        if self.read_ptr + self.n_outputs > self.row_size:
            return None  # No more data in buffer

        out = self.memory[self.read_ptr:self.read_ptr+self.n_outputs]
        self.read_ptr += stride_size
        return out

# Example demo
if __name__ == "__main__":
    lm = LineMemory(row_size=8, n_outputs=4)

    # Load one row of feature map
    for val in [1, 2, 3, 4, 5, 6, 7, 8]:
        lm.write(val)

    # Read out with stride=2
    while True:
        out = lm.read_stride(stride_size=2)
        if out is None:
            break
        print("Stride output:", out)

