# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


""" Secure Floating Point Number Computation"""

# Author: Zhiyu Liang

# Reference:  @MISC{Aliasgari_securecomputation,
    # author = {Mehrdad Aliasgari and Marina Blanton and Yihua Zhang and Aaron Steele},
    # title = {Secure Computation on Floating Point Numbers},
    # year = {}
# }

__author__ = "Zhiyu Liang"

# from numba import jit

import numpy as np
import math
from math import fabs
from ._basic_building_blocks_original import basic_building_blocks as bbb

from FedHC import glb

class fl(object):
    """ Convertion between real number and shared float number
        Implemented with numpy array
    
    """
    
    def __init__(self, k=9, l=23, num_parties=3):
        self.k = k
        self.l = l
        self.num_parties = num_parties
        self.bbb = bbb(num_parties)

    
    # @jit(cache=True, fastmath=True)
    def real_to_fl(self, x, index=0):
        """ Indexed party converts real number to shared float number and sends the shares to others
        
        --------------------------------
        Input: array x.shape = (num_of_real_numbers, ) or (num_of_real_numbers, 1)
        
        Output: (v, p, z, s), where    
        v.shape = (num_of_real_numbers, num_parties)      each number encoded in self.l bits
        p.shape = (num_of_real_numbers, num_parties)      one integer
        z.shape = (num_of_real_numbers, num_parties)      1 for zero else 0
        s.shape = (num_of_real_numbers, num_parties)      0 for positive, 1 for negative
        
        """
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n = x.shape[0]
        s = np.zeros((n,1), dtype='int')
        z = np.zeros((n,1), dtype='int')
        p = np.zeros((n,1), dtype='int')
        v = np.zeros((n, self.l), dtype='int')
        
        s[x < 0] = 1
        z[np.frompyfunc(fabs, 1, 1)(x) < 1e-10] = 1
        #print(z)
        if not (z == 1).all():
            v_temp, p_temp = np.frompyfunc(self._real2bin, 1, 2)(np.abs(x[z == 0]))
            #print(v[z[:,0] == 0], np.array(v_temp), list(v_temp))
            #v_temp = np.array([arr for arr in v_temp])
            #print(v_temp)
            v[z[:,0] == 0] = np.array(list(v_temp))
            p[z == 0] = np.array(p_temp)
        
        v_shares = np.random.randint(-20,20, (n, self.num_parties)).astype(object)
        p_shares = np.random.randint(-20,20, (n, self.num_parties))
        z_shares = np.random.randint(-20,20, (n, self.num_parties))
        s_shares = np.random.randint(-20,20, (n, self.num_parties))
        
        exp = 2 ** np.arange(self.l - 1, -1, -1, dtype='object')
        v_shares[:,index] += np.sum(v * exp, axis=1) - np.sum(v_shares, axis=-1)
        p_shares[:,index] += p[:,0] - np.sum(p_shares, axis=-1)
        z_shares[:,index] += z[:,0] - np.sum(z_shares, axis=-1)
        s_shares[:,index] += s[:,0] - np.sum(s_shares, axis=-1)

        if glb.has_global and glb.get_value('communication') is not None:
            glb.set_value('communication', glb.get_value('communication') + 4 * n * (self.num_parties - 1))
        
        
        return (v_shares, p_shares, z_shares, s_shares)
        
    # @jit(cache=True, fastmath=True)
    def fl_to_real(self, x):
        """ Convert a set of float numbers to real numbers
        
        
        """
        v, p, z, s = x
        n = v.shape[0]
        #value = np.zeros((n,1))
        
        v = self.bbb.Output(v).reshape(n, -1)
        p = self.bbb.Output(p).reshape(n, -1)
        z = self.bbb.Output(z).reshape(n, -1)
        s = self.bbb.Output(s).reshape(n, -1)
        
        value = ((1 - 2 * s) * (1 - z) * v * (2.0 ** p)).astype('float64')
        
        #value = ((1 - 2 * s) * (1 - z) * v * (2.0 ** p))
        #for i in range(n):
        #    if z[i, 0] == 0:
        #        valuep[i, 0] == 0
        #    v_bits = [ int(bit) for bit in bin(v[i,0])[2:] ]
        #    #print(i, v_bits)
        #    if len(v_bits) < self.l:
        #        v_bits = [0 for j in range(self.l - len(v_bits))] + v_bits
        #    value[i, 0] = self._to_real(v_bits, p[i,0], z[i,0], s[i,0])
        
        return value
        
    def _real2bin(self, input_real):
        input_int = int(input_real)
        input_dec = input_real - input_int
            
        if input_int > 0:
            v = [int(bit) for bit in bin(input_int)[2:]]
            p = (len(v) - self.l)
            if p >= 0:
                v = v[:self.l]
            else:
                result, shift = self._real2bin_dec(input_dec, -p)
                v = v + result
            
        else:
            v, shift = self._real2bin_dec(input_real, self.l, normalized=True)
            p = -self.l - shift
        
        return v, p
                
    
    
    def _real2bin_dec(self, dec, bit_length, *, normalized=False):
            if dec < 1:
                result = []
                i = 0
                shift = 0
                while i < bit_length:
                    dec *= 2
                    bit = int(dec)
                    dec -= bit
                    if normalized:
                        if bit == 0:
                            shift += 1
                        else:
                            result.append(bit)
                            normalized = False
                            i += 1
                    else:
                        result.append(bit)
                        i += 1
                
                return result, shift   
            else:
                print('Decimal error, input is %f' % dec)
                
    
    
#    def _to_real(self, v, p, z, s):
#        """ Convert one float number encoded in (v,p,z,s) into real number
#        
#        """
#        print(p)
#        if p >= 0:
#            value = self._bin2real(v + [ 0 for _ in range(p) ])
#        elif p <= -self.l:
#            value = self._bin2real([ 0 for _ in range(-self.l - p) ] + v, dtype='dec')
#        else:
#            value = self._bin2real(v[:p]) + self._bin2real(v[p:], dtype='dec')
#        return (1 - 2 * s) * (1 - z) * value
#    
#   
#    def _bin2real(self, bin_input, *, dtype='int'):
#        result = 0
#        if dtype == 'int':
#            print(bin_input)
#            for i in range(len(bin_input)-1):
#                result = (result + bin_input[i]) * 2
#            result += bin_input[-1]
#            return result
#        
#        elif dtype == 'dec':
#            for bit in reversed(bin_input):
#                result = (result + bit) / 2
#            return result
#        
#        else:
#            print("Incorrect binary input") 

    # @jit(fastmath=True, cache=True)
    def FLMul(self, x, y):
        """ Multiplication of x and y


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            y = (vy, py, zy, sy)

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative

        """

        l = self.l    
        v = self.bbb.Mul(x[0], y[0])
        #print(1,v)
        v = self.bbb.Trunc(v, 2 * l, l - 1)
        #print(2, v)
        diff = v.copy()
        diff[:,0] -= 2 ** l
        b = self.bbb.LTZ(diff, l + 1)
        #print(3, b, v)
        one_minus_b = -1 * b
        one_minus_b[:,0] += 1
        v = self.bbb.Trunc(2 * self.bbb.Mul(b, v) + self.bbb.Mul(one_minus_b, v), l+1, 1)
        #print(v)
        z = self.bbb.OR(x[2], y[2])
        s = self.bbb.XOR(x[3], y[3])
        plb = x[1] + y[1] - b
        #print(x[1], y[1], b, plb)
        plb[:,0] += l
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        #print(plb, one_minus_z)
        p = self.bbb.Mul(plb, one_minus_z)
     
        return (v, p, z, s)

    # @jit(fastmath=True, cache=True)
    def FLDiv(self, x, y):
        """ Division of x and y


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            y = (vy, py, zy, sy)

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
            Error.shape = (num_of_real_numbers, num_parties)  a bit indicate whether divisor is zero or not, 0 for False, 1 for True
        """
        l = self.l
        v = self.bbb.SDiv(x[0], y[0] + y[2], l)
        #print(v)
        diff = v.copy()
        diff[:,0] -= 2 ** l
        b = self.bbb.LTZ(diff, l + 1)
        one_minus_b = -1 * b.copy()
        one_minus_b[:,0] += 1
        v = self.bbb.Trunc(2 * self.bbb.Mul(b, v) + self.bbb.Mul(one_minus_b, v), l+1, 1)
        z = x[2]
        s = self.bbb.XOR(x[3], y[3])
        Error = y[2]

        plb = x[1] - y[1] - b
        #print(x[1], y[1], b, plb)
        plb[:,0] += 1 - l
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        #print(plb, one_minus_z)
        p = self.bbb.Mul(one_minus_z, plb)

        return ((v,p,z,s), Error)

    # @jit(fastmath=True, cache=True)
    def FLAdd(self, x, y):
        """ Addition of x and y 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            y = (vy, py, zy, sy)

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 
        """
    
        k = self.k
        l = self.l

        a = self.bbb.LT(x[1], y[1], k)
        b = self.bbb.EQ(x[1], y[1], k)
        c = self.bbb.LT(x[0], y[0], l)
        one_minus_a = -1 * a.copy()
        one_minus_a[:,0] += 1
        one_minus_b = -1 * b.copy()
        one_minus_b[:,0] += 1
        one_minus_c = -1 * c.copy()
        one_minus_c[:,0] += 1

        pmax = self.bbb.Mul(a, y[1]) + self.bbb.Mul(one_minus_a, x[1])
        pmin = self.bbb.Mul(one_minus_a, y[1]) + self.bbb.Mul(a, x[1])
        vmax = self.bbb.Mul(one_minus_b, self.bbb.Mul(a, y[0]) + self.bbb.Mul(one_minus_a, x[0])) + self.bbb.Mul(b, self.bbb.Mul(c, y[0]) + self.bbb.Mul(one_minus_c, x[0]))
        vmin = self.bbb.Mul(one_minus_b, self.bbb.Mul(a, x[0]) + self.bbb.Mul(one_minus_a, y[0])) + self.bbb.Mul(b, self.bbb.Mul(c, x[0]) + self.bbb.Mul(one_minus_c, y[0]))

        s3 = self.bbb.XOR(x[3], y[3])
        l_minus_p = -1 * (pmax - pmin)
        l_minus_p[:,0] += l
        d = self.bbb.LTZ(l_minus_p, k)
        one_minus_d = -1 * d.copy()
        one_minus_d[:,0] += 1
        two_delta = self.bbb.Pow2(self.bbb.Mul(one_minus_d, pmax - pmin), l+1)
        v3 = 2 * (vmax - s3)
        v3[:,0] += 1
        one_minus_two_s3 = - 2 * s3.copy()
        one_minus_two_s3[:,0] += 1
        v4 = self.bbb.Mul(vmax, two_delta) + self.bbb.Mul(one_minus_two_s3, vmin)
        v = np.frompyfunc(int, 1, 1)(self.bbb.Mul(self.bbb.Mul(d, v3) + self.bbb.Mul(one_minus_d, v4), self.bbb.Inv(two_delta)) * (2 ** l))
        #print(v)
        v = self.bbb.Trunc(v, 2 * l + 1, l - 1)
        u = self.bbb.BitDec(v, l+2, l+2)
        h = self.bbb.PreOr(u[:,-1::-1,:])
        p0 = -1 * np.sum(h, axis=1)
        p0[:,0] += l + 2
        exp = 2 ** np.arange(l+2, dtype='object').reshape(-1,1)
        one_minus_h = -1 * h.copy()
        one_minus_h[:,:,0] += 1
        pow_2_p0 = np.sum(exp * one_minus_h, axis=1)
        pow_2_p0[:,0] += 1

        v = self.bbb.Trunc(self.bbb.Mul(pow_2_p0, v), l+2, 2)
        p = pmax - p0 + one_minus_d
        one_minus_z1 = -1 * x[2].copy()
        one_minus_z1[:,0] += 1
        one_minus_z2 = -1 * y[2].copy()
        one_minus_z2[:,0] += 1
        v = self.bbb.Mul(self.bbb.Mul(one_minus_z1, one_minus_z2), v) + self.bbb.Mul(x[2], y[0]) + self.bbb.Mul(y[2], x[0])
        z = self.bbb.EQZ(v, l)
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        p = self.bbb.Mul(self.bbb.Mul(self.bbb.Mul(one_minus_z1, one_minus_z2), p) + self.bbb.Mul(x[2], y[1]) + self.bbb.Mul(y[2], x[1]), one_minus_z)
        s = self.bbb.Mul(one_minus_b, self.bbb.Mul(a, y[3]) + self.bbb.Mul(one_minus_a, x[3])) + self.bbb.Mul(b, self.bbb.Mul(c, y[3]) + self.bbb.Mul(one_minus_c, x[3]))
        s = self.bbb.Mul(self.bbb.Mul(one_minus_z1, one_minus_z2), s) + self.bbb.Mul(self.bbb.Mul(one_minus_z1, y[2]), x[3]) + self.bbb.Mul(self.bbb.Mul(x[2], one_minus_z2), y[3])

        return (v, p, z, s)


    def _FLAgg(self, x, mode):
        """ Aggregation of each of the elements of x


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            mode: 1 for summaration. 2 for production 

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 
        """
        vx, px, zx, sx = x
        num_element = vx.shape[0]

        if num_element == 1:
            return x
        
        mid = num_element // 2
        if mode == 1:
            return self.FLAdd(self._FLAgg((vx[:mid], px[:mid], zx[:mid], sx[:mid]), mode), self._FLAgg((vx[mid:], px[mid:], zx[mid:], sx[mid:]), mode))    
        if mode == 2:
            return self.FLMul(self._FLAgg((vx[:mid], px[:mid], zx[:mid], sx[:mid]), mode), self._FLAgg((vx[mid:], px[mid:], zx[mid:], sx[mid:]), mode))
        else:
            raise ValueError('Unexpected input value of mode = %d' % mode) 


    def FLSum(self, x):
        """ Summaration of each the elements of x 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            mode: 1 for summary. 2 for product 

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 
        """
        return self._FLAgg(x, 1)
    

    def FLProd(self, x):
        """ Production of each of the elements of x 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            mode: 1 for summary. 2 for product 

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 
        """
        return self._FLAgg(x, 2)
    
    # @jit(fastmath=True, cache=True)
    def FLSub(self, x, y):
        """ Substraction of x and y 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            y = (vy, py, zy, sy)

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 
        """
        sp = -1 * y[3].copy()
        sp[:,0] += 1
        yp = (y[0], y[1], y[2], sp)
        #print(y, yp)
        return self.FLAdd(x, yp) 
    
    # @jit(fastmath=True, cache=True)
    def FLLT(self, x, y):
        """ Determin whether x is less than y or not

        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            y = (vy, py, zy, sy)

        Returns
        -----------------------------------------------------------
            b.shape = (num_of_real_numbers, num_parties)      a bit, 1 for True, 0 for False
        """

        l = self.l
        k = self.k

        a = self.bbb.LT(x[1], y[1], k)
        c = self.bbb.EQ(x[1], y[1], k)
        one_minus_two_s1 = -2 * x[3].copy()
        one_minus_two_s1[:, 0] += 1
        one_minus_two_s2 = -2 * y[3].copy()
        one_minus_two_s2[:, 0] += 1
        one_minus_a = -1 * a.copy()
        one_minus_a[:, 0] += 1
        one_minus_c = -1 * c.copy()
        one_minus_c[:, 0] += 1
        d = self.bbb.LT(self.bbb.Mul(one_minus_two_s1, x[0]), self.bbb.Mul(one_minus_two_s2, y[0]), l+1)
        b_pos = self.bbb.Mul(c, d) + self.bbb.Mul(one_minus_c, a)
        b_neg = self.bbb.Mul(c, d) + self.bbb.Mul(one_minus_c, one_minus_a)

        one_minus_s1 = -1 * x[3].copy()
        one_minus_s1[:,0] += 1
        one_minus_s2 = -1 * y[3].copy()
        one_minus_s2[:,0] += 1
        one_minus_z1 = -1 * x[2].copy()
        one_minus_z1[:,0] += 1
        one_minus_z2 = -1 * y[2].copy()
        one_minus_z2[:,0] += 1

        b = self.bbb.Mul(self.bbb.Mul(x[2], one_minus_z2), one_minus_s2) + \
            self.bbb.Mul(self.bbb.Mul(one_minus_z1, y[2]), x[3]) + \
            self.bbb.Mul(self.bbb.Mul(one_minus_z1, one_minus_z2), \
                         self.bbb.Mul(x[3], one_minus_s2) + self.bbb.Mul(self.bbb.Mul(one_minus_s1, one_minus_s2), b_pos) + self.bbb.Mul(self.bbb.Mul(x[3], y[3]), b_neg))
        return b

    # @jit(fastmath=True, cache=True)
    def FLRound(self, x, mode):
        """ Round x 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            mode = 0 computes floor
            mode = 1 computes ceiling

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 
        """

        k = self.k
        l = self.l
        num_parties = self.num_parties

        a = self.bbb.LTZ(x[1], k)
        pl = x[1].copy()
        pl[:,0] -= 1 - l
        b = self.bbb.LTZ(pl, k)
        one_minus_b = -1 * b.copy()
        one_minus_b[:, 0] += 1
        # print(- Mul(Mul(a, one_minus_b), x[1]))
        v2, pow_2_p_inv = self.bbb.Mod2m_shared_m(x[0], l, -1 * self.bbb.Mul(self.bbb.Mul(a, one_minus_b), x[1]))
        c = self.bbb.EQZ(v2, l)
        one_minus_c = -1 * c.copy()
        one_minus_c[:, 0] += 1
        mode_shares = np.zeros((1, num_parties), dtype='int')
        mode_shares[:,0] = mode
        v = x[0] - v2 + self.bbb.Mul(self.bbb.Mul(one_minus_c, pow_2_p_inv), self.bbb.XOR(mode_shares, x[3]))
        v_minus_pow2_l = v.copy()
        v_minus_pow2_l[:,0] -= 2 ** l
        d = self.bbb.EQZ(v_minus_pow2_l, l+1)
        one_minus_d = -1 * d.copy()
        one_minus_d[:, 0] += 1
        v = d * (2 ** (l-1)) + self.bbb.Mul(one_minus_d, v)
        one_minus_a = -1 * a.copy()
        one_minus_a[:, 0] += 1

        # To convert v = +/- 1 to float format, refine v = +/- 1 to 2 ^ (l-1)
        v = self.bbb.Mul( a, self.bbb.Mul(one_minus_b, v) + 2 ** (l-1) * self.bbb.Mul(self.bbb.Mul(b, mode_shares - x[3]), mode_shares - x[3]) ) + self.bbb.Mul(one_minus_a, x[0])
        #v = Mul( a, Mul(one_minus_b, v) + Mul(Mul(b, mode_shares - x[3]), v) ) + Mul(one_minus_a, x[0])
        one_minus_bmode = -1 * b * mode
        one_minus_bmode[:,0] += 1
        s = self.bbb.Mul(one_minus_bmode, x[3])
        z = self.bbb.OR(self.bbb.EQZ(v, l), x[2])
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
    
        v = self.bbb.Mul(v, one_minus_z)

        # To convert v = +/- 1 to float format, refine p with additional b

        p = self.bbb.Mul(x[1] + b + self.bbb.Mul(self.bbb.Mul(d, a), one_minus_b), one_minus_z)

        return (v, p, z, s)


    # @jit(fastmath=True, cache=True)
    def Int2FL(self, a, gamma, l):
        """ Convert a signed gamma-bit integer 'a' into a floating point number

        Parameters
        -------------------------------
            a.shape = (number_of_integers, num_parties) 
            gamma     a public integer, the number of bits of the input integers 
            l         a public integer, the number of bits of the significands of the output floating point numbers 

        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
 

        """

        s = self.bbb.LTZ(a, gamma)
        z = self.bbb.EQZ(a, gamma)
        one_minus_2s = - 2 * s.copy()
        one_minus_2s[:,0] += 1
        a = self.bbb.Mul(one_minus_2s, a)
        a_bits = self.bbb.BitDec(a, gamma - 1, gamma - 1)
        b_bits = self.bbb.PreOr(a_bits[:, -1::-1, :])
        one_minus_b = -1 * b_bits.copy()
        one_minus_b[:,:, 0] += 1
        exp = 2 ** np.arange(gamma - 1, dtype='object').reshape(-1,1)
        v = a + self.bbb.Mul(a, np.sum(exp * one_minus_b, axis = 1))
        p = np.sum(b_bits, axis=1)
        p[:,0] -= gamma - 1
        v = self.bbb.Trunc(v, gamma - 1, gamma - l - 1) if gamma - 1 > l else 2 ** (l - gamma + 1) * v
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        p[:,0] += gamma - 1 - l
        p = self.bbb.Mul(p, one_minus_z)

        return (v, p, z, s)


    # @jit(fastmath=True, cache=True)
    def FLSqrt(self, x):
        """ Square root of x 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            
        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
            Error.shape = (num_of_real_numbers, num_parties)  a bit indicate the sign of x, 0 for positive, 1 for negative
        """

        l = self.l
        n = x[0].shape[0]

        b = self.bbb.BitDec(x[1], l, 1).reshape(n, -1)
        shared_zero = np.zeros_like(x[0], dtype='int')
        shared_l_0 = shared_zero.copy()
        shared_l_0[:,0] = l % 2
        c = self.bbb.XOR(b, shared_l_0)
        p = (x[1] - b) / 2 + b
        #p = (x[1] - b) / 2 + OR(b, shared_l_0)
        p[:,0] += math.floor(l / 2)
        alpha = self.real_to_fl(np.array([-0.8099868542], dtype='object'))
        beta = self.real_to_fl(np.array([1.787727479], dtype='object'))

        shared_l = shared_zero.copy()
        shared_l[:,0] += l
        x2 = self.FLMul((x[0], -1 * shared_l, shared_zero, shared_zero), alpha)
        x0 = self.FLAdd(x2, beta)
        xg = self.FLMul((x[0], -1 * shared_l, shared_zero, shared_zero), x0)
        xh = x0
        xh[1][:,0] -= 1
        for _ in range(math.ceil(math.log2(l/5.4)) - 1):
            x2 = self.FLMul(xg, xh)
            shared_three_pow2_l_minus_two = shared_zero.copy()
            shared_three_pow2_l_minus_two[:,0] += 3 * 2 ** (l - 2)
            shared_one_minus_l = -1 * shared_l.copy()
            shared_one_minus_l[:,0] += 1
            x2 = self.FLSub((shared_three_pow2_l_minus_two, shared_one_minus_l, shared_zero, shared_zero), x2)
            xg = self.FLMul(xg, x2)
            xh = self.FLMul(xh, x2)

        xh2 = self.FLMul(xh, xh)
        x2 = self.FLMul((x[0], -1 * shared_l, shared_zero, shared_zero), xh2)
        x2[1][:,0] += 1
        x2 = self.FLSub((shared_three_pow2_l_minus_two, shared_one_minus_l, shared_zero, shared_zero), x2)
        xh = self.FLMul(xh, x2)
        xh[1][:,0] += 1
        x2 = self.FLMul((x[0], -1 * shared_l, shared_zero, shared_zero), xh)
        fl_sqrt2 = self.real_to_fl(np.array([math.sqrt(2)]))
        one_minus_c = -1 * c.copy()
        one_minus_c[:,0] += 1
        x2 = self.FLMul(x2, (one_minus_c * 2 ** (l-1) + c * self.bbb.Output(fl_sqrt2[0]), \
                        - one_minus_c * (l-1) + c * self.bbb.Output(fl_sqrt2[1]), shared_zero, shared_zero))
        one_minus_z1 = -1 * x[2].copy()
        one_minus_z1[:,0] += 1
        #print(Output(x2[1]), Output(p))
        p = self.bbb.Mul(x2[1] + p, one_minus_z1)

        v = self.bbb.Mul(x2[0], one_minus_z1)
        Error = x[3]

        return ((v, p, x[2], x[3]), Error)


    # @jit(fastmath=True, cache=True)
    def FLLog2(self, x):
        """ Logarithm of x 


        Parameters
        -------------------------------
            x = (vx, px, zx, sx)
            
        Returns
        -----------------------------------------------------------
            v.shape = (num_of_real_numbers, num_parties)      self.l normalized(i.e., the most significant bit is always 1) bits significand, represented in unsigned integer \in [2 ^ (self.l - 1), 2 ^ self.l - 1]
            p.shape = (num_of_real_numbers, num_parties)      self.k-bits exponent, represented in signed integer \in [- 2 ^ (self.k - 1), 2 ^ (self.k -1)] 
            z.shape = (num_of_real_numbers, num_parties)      a bit indicate if x is zero, 1 for zero else 0
            s.shape = (num_of_real_numbers, num_parties)      a bit indicate the sign of x, 0 for positive, 1 for negative
            Error.shape = (num_of_real_numbers, num_parties)  a bit indicate whether x < 0 or not, 0 for False, 1 for True

        """

        l = self.l
        k = self.k
        n = x[0].shape[0]

        shared_zero = np.zeros_like(x[0], dtype='int')
        shared_pow2_l_minus_one = shared_zero.copy()
        shared_pow2_l_minus_one[:, 0] = 2 ** (l - 1)
        shared_l = shared_zero.copy()
        shared_l[:, 0] = l
        shared_l_minus_one = shared_l.copy()
        shared_l_minus_one[:, 0] -= 1
        x2 = self.FLSub((shared_pow2_l_minus_one, -1 * shared_l_minus_one, shared_zero, shared_zero), (x[0], -1 * shared_l, shared_zero, shared_zero))
        x3 = self.FLAdd((shared_pow2_l_minus_one, -1 * shared_l_minus_one, shared_zero, shared_zero), (x[0], -1 * shared_l, shared_zero, shared_zero))
        xy = self.FLDiv(x2, x3)[0]

        xy2 = self.FLMul(xy, xy)

        c0 = self.real_to_fl(np.array([2 * math.log2(math.e)] * n))
        v, p, z, s = self.FLMul(xy, c0)
        M = math.ceil(l / (2 * math.log2(3)) - 0.5)

        for i in range(1, M + 1, 1):

            xy = self.FLMul(xy, xy2)
            ci = self.real_to_fl(np.array([2 * math.log2(math.e) / (2 * i + 1)] * n, dtype='object' ))

            x2 = self.FLMul(xy, ci)

            v, p, z, s = self.FLAdd((v, p, z, s), x2)


        x2 = self.Int2FL(shared_l + x[1], l, l)
    
        v, p, z, s = self.FLSub(x2, (v, p, z, s))
        a = self.bbb.EQ(x[1], -1 * shared_l_minus_one, k)
        b = self.bbb.EQ(x[0], shared_pow2_l_minus_one, l)
        z = self.bbb.Mul(a, b)
        one_minus_z = z.copy()
        one_minus_z[:, 0] += 1
        v = self.bbb.Mul(v, one_minus_z)
        Error = self.bbb.OR(x[2], x[3])
        p = self.bbb.Mul(p, one_minus_z)


        return ((v, p, z, s), Error)



