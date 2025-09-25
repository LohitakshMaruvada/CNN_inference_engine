import numpy as np

class ProcessingElement:
    def __init__(self, weight_mem_size=16, k_bit=16):
        # Local filter weight memory
        self.weight_mem = [0] * weight_mem_size
        self.accumulator = 0
        self.k_bit = k_bit

    def load_weights(self, weights):
        """Load filter weights into local memory"""
        for i, w in enumerate(weights):
            self.weight_mem[i] = w

    def reset(self):
        self.accumulator = 0

    def mac(self, inputs, weights):
        """Multiply and accumulate"""
        for x, w in zip(inputs, weights):
            self.accumulator += x * w
        return self.accumulator

    def max_pool(self, inputs):
        """Max pooling"""
        return max(inputs)

    def avg_pool(self, inputs):
        """Average pooling using shift instead of division (power of 2 only)"""
        n = len(inputs)
        avg = sum(inputs) >> int(np.log2(n)) if n > 0 else 0
        return avg

    def relu(self, x):
        """ReLU using SZD (sign and zero detector)"""
        return max(0, x)

    def relu6(self, x):
        """ReLU6 (clip output ≤ 6)"""
        return min(max(0, x), 6)

    def compute(self, mode, inputs, weights=None):
        """
        mode: 'conv', 'fc', 'maxpool', 'avgpool', 'relu', 'relu6'
        """
        if mode in ['conv', 'fc']:
            if weights is None:
                raise ValueError("Weights required for Conv/FC")
            return self.mac(inputs, weights)

        elif mode == 'maxpool':
            return self.max_pool(inputs)

        elif mode == 'avgpool':
            return self.avg_pool(inputs)

        elif mode == 'relu':
            return self.relu(inputs[0])

        elif mode == 'relu6':
            return self.relu6(inputs[0])

        else:
            raise ValueError("Unknown mode")

# Example usage
if __name__ == "__main__":
    pe = ProcessingElement(weight_mem_size=4)
    pe.load_weights([1, -1, 2, 0])

    # Conv/FC
    out1 = pe.compute('conv', [2, 3, 1, 0], pe.weight_mem)
    print("Conv/FC Output:", out1)

    # MaxPool
    out2 = pe.compute('maxpool', [1, 5, 3, 2])
    print("MaxPool Output:", out2)

    # AvgPool
    out3 = pe.compute('avgpool', [4, 8, 12, 16])
    print("AvgPool Output:", out3)

    # ReLU
    out4 = pe.compute('relu', [-5])
    print("ReLU Output:", out4)

    # ReLU6
    out5 = pe.compute('relu6', [10])
    print("ReLU6 Output:", out5)

