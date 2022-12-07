import numpy as np
from os import urandom

class speck():

    def __init__(self, nr=22, alpha=7, beta=2, block_size=32, key_size=64):
        self.alpha = alpha
        self.beta = beta
        self.block_size = block_size
        self.word_size = int(block_size/2)
        self.key_size = key_size
        self.mask = 2 ** self.word_size - 1
        self.nr = nr
        self.ks = None

    # Left Rotation
    def rol(self, x, t):
        return ((x << t) & self.mask) | (x >> (self.word_size - t))

    # Right Rotation
    def ror(self, x, t):
        return (x >> t) | ((x << (self.word_size - t)) & self.mask)
    
    # Return master key with size (4, batch_size)
    # urandom() returns one bytes, which is 8 bits of true random number
    # As we need uint16 here, we need two bytes for one data
    def generate_master_key(self, num = None, m = 4):
        return np.frombuffer(urandom(2 * m * num), dtype = np.uint16).reshape(m, -1)
    
    # Expand keys for nr rounds
    # This is a 4-word key schedules
    # Thus input k: (4, batch_size)
    # ks = self.expand_key(keys, nr) 
    def expand_key(self, k, nr):
        ks = [0 for i in range(nr)]
        k = list(reversed(k))
        ks[0] = k[0]
        l = k[1:]
        for i in range(nr-1):
            l[i%3], ks[i+1] = self.enc_one_round((l[i%3], ks[i]), i)
        return ks

    # Enc for one round
    def enc_one_round(self, p, k):
        x, y = p[0], p[1]
        x = ((self.ror(x, self.alpha) + y) & self.mask) ^ k
        y = self.rol(y, self.beta) ^ x
        return x, y

    # Multi-rounds enc using enc_one_round
    # Input p would be array of plaintext_l and plaintext r, each with 16 bits
    # For batch, p: ((batch_size), (batch_size))
    # nr represents encryption rounds
    def encrypt(self, p, ks):
        x, y = p[0], p[1]
        for k in ks:
            x, y = self.enc_one_round((x, y), k)
        return x, y

    # Dec for one round
    def dec_one_round(self, c, k):
        x, y = c[0], c[1]
        y = self.ror(x ^ y, self.beta)
        x = self.rol(((x ^ k) - y) & self.mask, self.alpha)
        return x, y

    # Multi-rounds dec using dec_one_round
    def decrypt(self, c, ks):
        x, y = c[0], c[1]
        for k in reversed(ks):
            x, y = self.dec_one_round((x, y), k)
        return x, y

    # test whether the code is right
    def test_vector(self):
        key = (0x1918,0x1110,0x0908,0x0100)
        pt = (0x6574, 0x694c)
        ks = self.expand_key(key, 22)
        ct = self.encrypt(pt, ks)
        pp = self.decrypt(ct, ks)

        if (ct == (0xa868, 0x42f2) and pt == pp):
            print("Testvector verified.")
            return(True)
        else:
            print("Testvector not verified.")
            return(False)

    ### The next part is to generate data pairs ###

    # Turn X into binary representation
    def convert_to_binary(self, arr):
        X = np.zeros((4 * self.word_size, len(arr[0])), dtype=np.uint8)
        for i in range(X.shape[0]):
            index = i // self.word_size
            offset = self.word_size - (i % self.word_size) - 1
            X[i] = (arr[index] >> offset) & 1
        return X.transpose()

    def generate_train_data(self, n, nr, diff=(0x0040, 0)):
        # generate labels
        # half 0, half 1
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
        # generate plaintext
        p0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
        p1l = p0l^diff[0]
        p1r = p0r^diff[1]
        p1l[Y==0] = np.frombuffer(urandom(2*np.sum(Y==0)), dtype=np.uint16)
        p1r[Y==0] = np.frombuffer(urandom(2*np.sum(Y==0)), dtype=np.uint16)
        # generate keys
        master_keys = self.generate_master_key(n)
        ks = self.expand_key(master_keys, nr)
        # generate ciphertext
        c0l, c0r = self.encrypt((p0l, p0r), ks)
        c1l, c1r = self.encrypt((p1l, p1r), ks)
        # Generate data with binary representation
        X = self.convert_to_binary([c0l, c0r, c1l, c1r])
        return X, Y


if __name__=='__main__':
    speck = speck()
    speck.test_vector()