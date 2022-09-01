# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


""" Basic building blocks for implementing floating point computataion"""

__author__ = "Zhiyu Liang"

# from numba import jit

import numpy as np
cimport numpy as np
import math
from math import fabs
from FedHC import glb




class basic_building_blocks(object):

    def __init__(self, num_parties=3):
        self.num_parties = num_parties

    
    
    
    def Output(self, x):
        """ All parties open the shares

        """

        sum_temp = np.sum(x, axis=-1, keepdims=True)
        if sum_temp.ndim == 1:
            sum_temp = sum_temp.reshape(1,-1)
        if glb.has_global and glb.get_value('communication') is not None:
            glb.set_value('communication', glb.get_value('communication') + sum_temp.shape[0] * self.num_parties)

        return sum_temp 

    def RandInt(self, k, n=1):
        """ Generate n random interger for each party without interaction
        
        """
        
        return np.random.randint(0, 2 ** (k % 64), (n, self.num_parties), dtype='int64').astype('object')
    
    def RandBit(self, n):
        """ generate n random shared bits each row is a shared bits, and each column is shares of a party 
            use determined shares for test only
        
        """
        #result = np.zeros((n,3), dtype='int')
        #for row in np.random.choice(list(range(n)),np.random.randint(n), replace=False):
        #    column = np.random.randint(0,3,1)
        #    result[row, column] = 1
        #return result

        result = np.random.randint(-10,10, (n, self.num_parties))
        result[:,-1] = (np.random.randint(0, 2, n) - np.sum(result[:,:-1], axis=1))
        return result

    
    def RandM(self, k, m, n=1):
        """ Generate n shared random integers rp and rpp, together with the shared bits of rp
            # rpp from [0, 2^(k-m) - 1]
            # rp generated from m bits 
        """

        rpp = self.RandInt(k+2-m, n)
        b = self.RandBit(m*n).reshape(n, m, -1)
        rp = np.array(list(map(lambda x: np.sum( (np.logspace(0,m-1,num=m,base=2,dtype='int64').reshape(-1,1) * x), axis=0 ), b)), dtype='object')
        return rpp, rp, b
    
    # @jit(fastmath=True)
    def Mul(self, x, y):
        """ Secret shared multiplication of two sets of shared x and y

        """
        #print('x', x, '\n\n\n', 'y', y)

        
        fake_triples = np.array([[0] * (self.num_parties - 1) + [1], [1] * self.num_parties, [1] * self.num_parties], dtype='int64')
        alpha = self.Output(x - fake_triples[0])
        beta = self.Output(y - fake_triples[1])
        #print(alpha, '\n\n', beta )

        prod = fake_triples[2] + alpha * fake_triples[1] + beta * fake_triples[0]
        prod[:,0] += np.squeeze(alpha * beta)
        return prod

    def Prod(self, x):
        """ 
        Return secret shared product x[0]*x[1]*..*x[k-1] of each of the n sets 
        x.shape = (num_sets, num_integers, num_parties)

        """

        n = x.shape[0]
        k = x.shape[1]
        if k == 1:
            return x
        else:
            l = k - k % 2
            left = [i+j*k for j in range(n) for i in range(l//2)]
            right = [i+j*k for j in range(n) for i in range(l//2, l)]
            result = self.Mul(x.reshape(n*k,-1)[left], x.reshape(n*k,-1)[right]).reshape(n,l//2,-1)
            return self.Prod(result) if l == k else self.Prod(np.concatenate((result, x[:,-1,:].reshape(n,1,-1)), 1))
    
    def MulPub(self, x, y):
        """ The parties generate a pseudo-random sharing of zero(PRZS) 
            then each party compute a randomized product of shares [z] = [x][y] + [0], and exchange the shares
            to reconstruct z = xy
        
        """
        # Here PRZS is ignored, compute [x][y] directly 
        return self.Output(self.Mul(x,y))
    
    def XOR(self, a, b):
        return a + b - 2 * self.Mul(a, b)

    def OR(self, a, b):
        return a + b - self.Mul(a, b)

    def Inv(self, a):
        """ Return 1/a

        """

        if a.ndim == 1:
            a = a.reshape(1, -1)
        n = a.shape[0]
        r = np.random.randint(1,17,(n, self.num_parties))
        c = self.MulPub(r, a)
        return r/c



    
    
    # @jit(fastmath=True)
    def PreMul(self, x):
        """ # Calculate prefix multiplication where [p_i] = Mul([x_0], [x_1]. ..., [x_i])
            # x.shape = (num_sets, num_integers, num_parties)
        
        """
        n = x.shape[0]
        k = x.shape[1]
        if k == 1:
            return x
        
        r = np.random.randint(1, 17, (n*k, self.num_parties))
        s = np.random.randint(1, 17, (n*k, self.num_parties))
        #u = np.sum(r,axis=1) * np.sum(s,axis=1)
        u = self.MulPub(r,s)
        # ignore 1st element of each of the k sets of r, and k-st element of each of s
        r_idx = [i for i in range(n*k) if i % k != 0]
        su_idx = [i for i in range(n*k) if i % k != k-1]
        v = self.Mul(r[r_idx],s[su_idx])
        #w = np.zeros((n,3))
        #w[0] = r[0]
        #w[1:] = v*((1/u[:-1].reshape(-1,1))%17)
        vu = v*((1/u[su_idx]))
        w0_idx = [i*k for i in range(n)]
        w = np.concatenate((r[w0_idx].reshape(n,1,-1), vu.reshape(n,k-1,-1)), axis=1)
        #w = np.concatenate((r[0].reshape(1,-1), v*((1/u[i for in range(k) and i % k != k-1].reshape(-1,1))%17)), axis=0)
        z = (s * ((1/u.reshape(-1,1)))).reshape(n,k,-1)
        #m = np.array([np.sum(w[0]*a1, axis=0),np.sum(w[1]*a2, axis=0),np.sum(w[2]*a3, axis=0)])
        #m = np.sum(w, axis=1) * np.array([np.sum(a1), np.sum(a2), np.sum(a3)])
        m = self.MulPub(w.reshape(n*k,-1),x.reshape(n*k,-1)).reshape(n,k,-1)
        #p = np.zeros((n,3))
        #p[0] = a1
        #p[1] = z[1]*np.cumprod(m,axis=0)[1]
        #p[2] = z[2]*np.cumprod(m,axis=0)[2]
        m_cumprod = np.cumprod(m, axis=1)
        p = np.concatenate((x[:,0,:].reshape(n,1,-1), z[:,1:,:] * m_cumprod[:,1:,:]), axis=1)
        p_integer = np.frompyfunc(int, 1, 1)(p)
        # each party send its decimal to party_0
        p_integer[:,:,0] += np.frompyfunc(int, 1, 1)(np.frompyfunc(round, 1, 1)(np.sum(p - p_integer, axis=-1)))
        return p_integer
    
    
    # @jit(fastmath=True)
    def PreOr(self, a):
        """ # Calculate prefix Or of n sets of k shared bits
            # a.shape = (num_bits_set, num_bits, num_parties)
        
        """

        n = a.shape[0]
        k = a.shape[1]
        if k == 1:
            return a
        ap = np.concatenate( (a[:,:,0].reshape(n,k,1)+1, a[:,:,1:]), axis=2) 
        b = self.PreMul(ap)
        p = (np.concatenate( (a[:,0,:].reshape(n,1,-1),  -1 * self.Mod2(b[:,1:,:].reshape(n*(k-1),-1), k).reshape(n,k-1,-1) ), axis=1))
        p[:,1:,0] += 1

        return p

    # @jit(fastmath=True)
    def KOrCS(self, a):
        """ k-ary Or of k shared bits
            Cannot find algorithm within corresponding reference, so implemented similar to PreOr instead
            a.shape = (num_bits_set, num_bits, num_parties)
        
        """

        n = a.shape[0]
        k = a.shape[1]
        if k == 1:
            return a
        #b = PreMul(a+1)
        #return 1 - Mod2(b[-1], k)
        ap = np.concatenate( (a[:,:,0].reshape(n,k,1)+1, a[:,:,1:]), axis=2)
        b = self.Prod(ap)
        #return 1 - Mod2(b, k)
        result = -1 * self.Mod2(b.reshape(n,-1),k).reshape(n,1,-1)
        result[:,:,0] += 1
        return result
    
    def Bits(self, x, m):
        """ convert integer x to binary representation of m+1 least bits
        
        """
        #print(x)
        x_bits = bin(x)[2:] if x >= 0 else bin(x)[3:]
        x_bits = np.array([b for b in x_bits[::-1]], dtype='int')
        x_length = x_bits.shape[0]
        if x_length >= m:
            return x_bits[:m].reshape(-1,1)
        else:
            return np.concatenate( (x_bits, np.zeros((m - x_length), dtype='int')), axis=0 ).reshape(-1,1)
    
    
    # @jit(fastmath=True)
    def KOr(self, a):
        """ # k-ary Or operation with low complexity
            # transform k bits to logk bits and compute k-ary Or of the logk bits
            # a.shape = (num_sets, num_bits, num_parties)
        
        """

        n = a.shape[0]
        k = a.shape[1]
        # m = ceil(log_2(k)) + 1 rather than ceil(log_2(k)) because when k = 2^m , m bits can only represent
        # x \in [0,k-1] while sum(a) \in [0,k]. When sum(a) is k, the result will be incorrect
        m = np.int64(np.ceil(np.log2(k))) + 1
        rpp, rp, b = self.RandM(k, m, n)
        c = self.Output((2 ** m) * rpp + rp + np.sum(a, axis=1))
        #c = Output((2 ** m) * rpp + rp)[0,0] + Output(np.sum(a, axis=0))[0,0] % k
        #c_bits = np.frompyfunc(Bits, 2, 1)(c, m)
        #print(c)
        c_bits = np.frompyfunc(self.Bits, 2, 1)(c, m)
        c_bits = np.array([arr[0] for arr in c_bits])
        #print(c_bits)
        #print(b)
        #d = c_bits + b[0] - 2 * c_bits * b[0]
        d = np.concatenate((c_bits, np.zeros((n, m, self.num_parties - 1), dtype='int')), axis=2) + b - 2 * c_bits * b
        #print(d)
        return self.KOrCS(d)
    
    # @jit(fastmath=True)
    def Mod2(self, x, k):
        """ Extract the least significant bit of x \in Z_k
        
        """

        n = 1 if x.ndim == 1 else x.shape[0]
        rpp, rp, b = self.RandM(k,1,n)
        rp_0 = b[:,0,:]
        c = 2**(k) + self.Output( x + 2*rpp + rp_0)
        c_0 = c % 2
        return np.concatenate((c_0, np.zeros((n, self.num_parties - 1), dtype='int')), axis=1)+ rp_0 - 2 * c_0 * rp_0 

    
    # @jit(fastmath=True)
    def BitLT(self, a, b):
        """ # Compare two sets of integers encoded in bits
            # a is public while c is shared
            # s = (a < b) ? 1 : 0
            # Protocol 4.5: BitLTC1 of "Improved Primitives for Secure Multiparty Integer Computation"

            Parameters
            -------------------------------
            a.shape = (num_integers, k, 1)
            b.shape = (num_integers, k, parties)
            where k is num_bits
            
            """
        n = a.shape[0]
        k = a.shape[1]
        d = np.concatenate( (a+1, np.zeros((n, k, self.num_parties - 1), dtype='int')), axis=2 ) + b - 2 * a * b
        p = self.PreMul(d[:,-1::-1,:])
        p = p[:,-1::-1, :]
        s = np.concatenate( ( p[:,:-1,:] - p[:,1:,:], p[:,-1,:].reshape(n,1,-1)), axis=1 )
        s[:,-1,0] -= 1
        s = np.sum( (1 - a) * s, axis=1)
        return self.Mod2(s, k)

    
    # @jit(fastmath=True)
    def Mod2m(self, x, k, m):
        """ Compute x mod 2^m
            x.shape = (num_integers, num_parties)
            k is a constant
            m is a constanct

        """
        n = x.shape[0]
        rpp, rp, b = self.RandM(k, m, n)
        #c = self.Output( x + (2 ** m) * rpp + rp)
        c = 2**(k-1) + self.Output( x + (2 ** m) * rpp + rp)
        #print(c)
        cp = c % (2 ** m)
        c_bits = np.frompyfunc(self.Bits, 2, 1)(c, m)
        c_bits = np.array([arr[0] for arr in c_bits])
        u = self.BitLT(c_bits, b)
        ap = np.concatenate( (cp, np.zeros((n, self.num_parties - 1), dtype='int')), axis=1 ) - rp + 2 ** m * u
        return ap

    
    # @jit(fastmath=True)
    def Mod2m_shared_m(self, x, k, m):
        """ Compute x mod 2^m where m is shared
            x.shape = (num_integers, num_parties)
            k is a constant
            m.shape = (num_integers, num_parties)

        """
        m_unary_bits = self.B2U(m, k)
        m_pow2 = self.Pow2(m, k)
        # m_pow2_inv = self.Inv(m_pow2)

        n = x.shape[0]
        rpp, rp, b = self.RandM(k, k, n)

        s = 2 ** np.arange(k, dtype='object').reshape(-1,1)

        #print(m_unary_bits.shape, b.shape)
        rp = np.sum(s * self.Mul(m_unary_bits.reshape(-1, self.num_parties), b.reshape(-1, self.num_parties)).reshape(b.shape), axis=1)

        m_not = - m_unary_bits
        m_not[:,:,0] += 1
    
        rpp = 2 ** k * rpp + np.sum(s * self.Mul(m_not.reshape(-1, self.num_parties), b.reshape(-1, self.num_parties)).reshape(b.shape), axis=1)

        c = self.Output(x + rpp + rp)

        cp = (c % s.T).reshape(n, k, -1)
        cpp = np.sum(cp[:,1:,:] * (m_unary_bits[:,:-1,:] - m_unary_bits[:,1:,:]), axis=1)
        d = self.LT(cpp, rp, k)

        result = cpp - rp + self.Mul(m_pow2, d)
        result_int = np.frompyfunc(int,1,1)(result)
        result_int[:,0] += np.frompyfunc(int,1,1)(np.sum(result - result_int, axis=1))

        return result_int, m_pow2

    
    # @jit(fastmath=True)
    def Trunc(self, x, k, m):
        """ Coupute x / 2^m
        
        """

        ap = self.Mod2m(x, k, m)
        d = (x - ap) * 2 ** (-m)
        dint = np.frompyfunc(int, 1, 1)(d)
        dint[:,0] += np.frompyfunc(int, 1, 1)(np.sum(d - dint, axis=1))
        return dint

    
    # @jit(fastmath=True)
    def EQZ(self, x, k):
        """ Determine if x equals to zero or not

        """
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n = x.shape[0]
        rpp, rp, b = self.RandM(k, k, n)
        # Using x + 2^k * rpp + rp instead of 2^(k-1) + (x + 2^k * rpp + rp) because 2^(k-1) will introduce 1 into
        # the k-th bit thus c_bits differs from rp while x is zero. It's not clear why 2^(k-1) is needed in Protocol
        # 3.7 of "Improved Primitives for Secure Multiparty Integer Computation"
        c = self.Output( x + 2 ** k * rpp + rp)
        c_bits = np.frompyfunc(self.Bits, 2, 1)(c, k)
        c_bits = np.array([arr[0] for arr in c_bits])
        #d = c_bits + b - 2 * c_bits * b
        d = np.concatenate((c_bits, np.zeros((n, k, self.num_parties - 1), dtype='int')), axis=2) + b - 2 * c_bits * b
        #result = np.concatenate([ - KOr(di.reshape(1,k,3)) for di in d], axis=0)
        result = -1 * self.KOr(d).reshape(n,-1)
        result[:,0] += 1
        return result

    
    def LTZ(self, x, k):
        """ Determine if x is less than zero or not

        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # n = x.shape[0]
        return -1 * self.Trunc(x, k, k-1)


    def EQ(self, x, y, k):
        """ Determine if x == y or not

        """
        return self.EQZ(x - y, k)

    def LT(self, x, y, k):
        """ Determine if x < y or not

        """

        return self.LTZ(x - y, k)

    
    
    # @jit(fastmath=True)
    def BitAdd(self, a, b):
        """ Output the shared bits of a + b, a is clear and b is shared   
            a.shape = (num_integers, 1)
            b.shape = (num_integers, k, num_parties)
            k = num_bits

        """
        n = b.shape[0]
        k = b.shape[1]
        exp = 2 ** np.arange(k, dtype='object').reshape(k,1)
        #a_integer = np.sum(a * exp, axis=1)
        b_integer = np.sum(np.sum(b, axis=2, keepdims=True) * exp, axis=1)
        c = a + b_integer
        #print(a, b_integer, c)
        c_bits = np.frompyfunc(self.Bits, 2, 1)(c, k+1) 
        c_bits = np.array([arr[0] for arr in c_bits])
        result = np.random.randint(-10,10, (n, k+1, self.num_parties))
        result[:,:,-1] = c_bits[:,:,0] - np.sum(result[:,:,:-1], axis=2)
        return result


    
    # @jit(fastmath=True)
    def BitDec(self, x, k, m):
        """ bit decomposition of m least significant bits of x represented in complementary format

        """

        n = x.shape[0]
        rpp, rp, b = self.RandM(k, m, n)
        c = 2 ** k + self.Output(x - rp + (2 ** m) * rpp)
        #c_bits = np.frompyfunc(Bits, 2, 1)(c, m) 
        #c_bits = np.array([arr[0] for arr in c_bits])
        #print(c)
        #print(Output(rp))
    
        return self.BitAdd(c % 2**m, b)[:,:-1]

    
    # @jit(fastmath=True)
    def Pow2(self, x, k):
        """ Output shares of 2^x where x \in [0, k)

        """

        m = np.int64(np.ceil(np.log2(k)))
        x_bits = self.BitDec(x, m, m)
        d = (2 ** (2 ** np.arange(m, dtype='object'))).reshape(-1,1) * x_bits - x_bits
        d[:,:,0] += 1
        return self.Prod(d).reshape(x.shape)


    
    # @jit(fastmath=True)
    def B2U(self, x, k):
        """ Conversion 0 <= x < k from binary to unary bitwise representation where x least significant bits are 1
            and all other k - x bits are 0

        """

        n = x.shape[0]
        x_pow2 = self.Pow2(x, k)
        rpp, rp, b = self.RandM(k, k, n)
        c = self.Output(x_pow2 + 2 ** k * rpp + rp)
        c_bits = np.frompyfunc(self.Bits, 2, 1)(c, k) 
        c_bits = np.array([arr[0] for arr in c_bits])
        d = np.concatenate((c_bits, np.zeros((n, k, self.num_parties - 1), dtype='int')), axis=2) + b - 2 * c_bits * b
        y = - 1 * self.PreOr(d)
        y[:,:,0] += 1
        return y
    
    
    # @jit(fastmath=True)
    def Trunc_shared_m(self, x, k, m):
        """ Perform trunction of [x] by an unknown number of bits [m], same as x // m, where k is the bitlength of x
            x.shape = (num_integers, num_parties)
        """
        m_unary_bits = self.B2U(m, k)
        m_pow2_inv = self.Inv(self.Pow2(m, k))

        n = x.shape[0]
        rpp, rp, b = self.RandM(k, k, n)

        s = 2 ** np.arange(k, dtype='int64').reshape(-1,1)

        rp = np.sum(s * self.Mul(np.tile(m_unary_bits.reshape(-1, self.num_parties), (n,1)), b.reshape(-1, self.num_parties)).reshape(b.shape), axis=1)

        m_not = - m_unary_bits
        m_not[:,:,0] += 1
    
        rpp = 2 ** k * rpp + np.sum(s * self.Mul(np.tile(m_not.reshape(-1, self.num_parties), (n,1)), b.reshape(-1, self.num_parties)).reshape(b.shape), axis=1)

        c = self.Output(x + rpp + rp)

        cp = (c % s.T).reshape(n, k, -1)
        cpp = np.sum(cp[:,1:,:] * (m_unary_bits[:,:-1,:] - m_unary_bits[:,1:,:]), axis=1)
        d = self.LT(cpp, rp, k)

        result = self.Mul(x - cpp + rp, m_pow2_inv) - d
        result_int = np.frompyfunc(int,1,1)(result)
        result_int[:,0] += np.frompyfunc(int,1,1)(np.sum(result - result_int, axis=1))

        return result_int


    
    # @jit(fastmath=True)
    def TruncPr(self, x, k, m):
        """ Return d = floor(x / 2 ^ m) + u where u is a random bit and Pr(u = 1) = (x mod 2 ^m) / 2 ^ m
            The protocol rounds  x / 2 ^ m to the nearest integer with probability 1 - alpha, alpha is the distance 
            between x / 2 ^ m and the nearest integer

            0 < m < k
        """

        #y = x.copy()
        #y[:,0] += 2 ** (k - 1)
        n = x.shape[0]

        rpp, rp, b = self.RandM(k, m, n)

        c = 2 ** (k - 1) + self.Output(x + 2 ** m * rpp + rp)
        cp = c % (2 ** m)
        xp = -1 * rp
        #print(cp.dtype)
        xp[:,0] += cp[:,0]
        d = (x - xp) / (2 ** m)
        dint = np.frompyfunc(int, 1, 1)(d)
        dint[:,0] += np.frompyfunc(int, 1, 1)(np.sum(d - dint, axis=-1))
        return dint

    
    # @jit(fastmath=True)
    def SDiv(self, a, b, k):
        theta = np.int32(np.ceil(np.log2(k)))
        x = b.astype(object)
        y = a.astype(object)
        exp_k = np.zeros_like(x)
        exp_k[:,0] += 2 ** (k + 1)
        for i in range(theta - 1):
            #print(i, 1,'\n','y: ', y, '\n', 'x: ', x)
            y = self.Mul(y, exp_k - x)
            #print(i, 2,'\n','y: ', y, '\n', 'x: ', x)
            y = self.TruncPr(y, 2 * k + 1, k)
            #print(i, 3,'\n','y: ', y, '\n', 'x: ', x)
            x = self.Mul(x, exp_k - x)
            #print(i, 4,'\n','y: ', y, '\n', 'x: ', x)
            x = self.TruncPr(x, 2 * k + 1, k)
            #print(i, 5,'\n','y: ', y, '\n', 'x: ', x)

        y = self.Mul(y, exp_k - x)
        y = self.TruncPr(y, 2 * k + 1, k)
        #print(y)   
        return y


    

