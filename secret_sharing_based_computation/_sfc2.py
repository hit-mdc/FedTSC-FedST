""" Secure Floating Point Number Computation"""

# Author: Zhiyu Liang

# Reference:  @MISC{Aliasgari_securecomputation,
    # author = {Mehrdad Aliasgari and Marina Blanton and Yihua Zhang and Aaron Steele},
    # title = {Secure Computation on Floating Point Numbers},
    # year = {}
# }

__author__ = "Zhiyu Liang"

from numba import jit

import numpy as np
import math
from math import fabs
import random

# from secret_sharing_based_computation._basic_building_blocks import basic_building_blocks as bbb

class fl(object):
    """ Convertion between real number and shared float number
        Implemented with numpy array
    
    """
    
    def __init__(self, k=9, l=23, num_parties=3):
        self.k = k
        self.l = l
        self.num_parties = num_parties
        # self.bbb = bbb(num_parties)

    
    @jit(cache=True, fastmath=True)
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
        
        
        return (v_shares, p_shares, z_shares, s_shares)
        
    @jit(cache=True, fastmath=True)
    def fl_to_real(self, x):
        """ Convert a set of float numbers to real numbers
        
        
        """
        v, p, z, s = x
        n = v.shape[0]
        #value = np.zeros((n,1))
        
        v = Output(v).reshape(n, -1)
        p = Output(p).reshape(n, -1)
        z = Output(z).reshape(n, -1)
        s = Output(s).reshape(n, -1)
        
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
                
    
    
    def _to_real(self, v, p, z, s):
        """ Convert one float number encoded in (v,p,z,s) into real number
        
        """
        print(p)
        if p >= 0:
            value = self._bin2real(v + [ 0 for _ in range(p) ])
        elif p <= -self.l:
            value = self._bin2real([ 0 for _ in range(-self.l - p) ] + v, dtype='dec')
        else:
            value = self._bin2real(v[:p]) + self._bin2real(v[p:], dtype='dec')
        return (1 - 2 * s) * (1 - z) * value
    
   
    def _bin2real(self, bin_input, *, dtype='int'):
        result = 0
        if dtype == 'int':
            print(bin_input)
            for i in range(len(bin_input)-1):
                result = (result + bin_input[i]) * 2
            result += bin_input[-1]
            return result
        
        elif dtype == 'dec':
            for bit in reversed(bin_input):
                result = (result + bit) / 2
            return result
        
        else:
            print("Incorrect binary input") 

    @jit(fastmath=True, cache=True)
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
        v = Mul(x[0], y[0])
        #print(1,v)
        v = Trunc(v, 2 * l, l - 1)
        #print(2, v)
        diff = v.copy()
        diff[:,0] -= 2 ** l
        b = LTZ(diff, l + 1)
        #print(3, b, v)
        one_minus_b = -1 * b
        one_minus_b[:,0] += 1
        v = Trunc(2 * Mul(b, v) + Mul(one_minus_b, v), l+1, 1)
        #print(v)
        z = OR(x[2], y[2])
        s = XOR(x[3], y[3])
        plb = x[1] + y[1] - b
        #print(x[1], y[1], b, plb)
        plb[:,0] += l
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        #print(plb, one_minus_z)
        p = Mul(plb, one_minus_z)
     
        return (v, p, z, s)

    @jit(fastmath=True, cache=True)
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
        v = SDiv(x[0], y[0] + y[2], l)
        #print(v)
        diff = v.copy()
        diff[:,0] -= 2 ** l
        b = LTZ(diff, l + 1)
        one_minus_b = -1 * b.copy()
        one_minus_b[:,0] += 1
        v = Trunc(2 * Mul(b, v) + Mul(one_minus_b, v), l+1, 1)
        z = x[2]
        s = XOR(x[3], y[3])
        Error = y[2]

        plb = x[1] - y[1] - b
        #print(x[1], y[1], b, plb)
        plb[:,0] += 1 - l
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        #print(plb, one_minus_z)
        p = Mul(one_minus_z, plb)

        return ((v,p,z,s), Error)

    @jit(fastmath=True, cache=True)
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

        a = LT(x[1], y[1], k)
        b = EQ(x[1], y[1], k)
        c = LT(x[0], y[0], l)
        one_minus_a = -1 * a.copy()
        one_minus_a[:,0] += 1
        one_minus_b = -1 * b.copy()
        one_minus_b[:,0] += 1
        one_minus_c = -1 * c.copy()
        one_minus_c[:,0] += 1

        pmax = Mul(a, y[1]) + Mul(one_minus_a, x[1])
        pmin = Mul(one_minus_a, y[1]) + Mul(a, x[1])
        vmax = Mul(one_minus_b, Mul(a, y[0]) + Mul(one_minus_a, x[0])) + Mul(b, Mul(c, y[0]) + Mul(one_minus_c, x[0]))
        vmin = Mul(one_minus_b, Mul(a, x[0]) + Mul(one_minus_a, y[0])) + Mul(b, Mul(c, x[0]) + Mul(one_minus_c, y[0]))

        s3 = XOR(x[3], y[3])
        l_minus_p = -1 * (pmax - pmin)
        l_minus_p[:,0] += l
        d = LTZ(l_minus_p, k)
        one_minus_d = -1 * d.copy()
        one_minus_d[:,0] += 1
        two_delta = Pow2(Mul(one_minus_d, pmax - pmin), l+1)
        v3 = 2 * (vmax - s3)
        v3[:,0] += 1
        one_minus_two_s3 = - 2 * s3.copy()
        one_minus_two_s3[:,0] += 1
        v4 = Mul(vmax, two_delta) + Mul(one_minus_two_s3, vmin)
        v = np.frompyfunc(int, 1, 1)(Mul(Mul(d, v3) + Mul(one_minus_d, v4), Inv(two_delta)) * (2 ** l))
        #print(v)
        v = Trunc(v, 2 * l + 1, l - 1)
        u = BitDec(v, l+2, l+2)
        h = PreOr(u[:,-1::-1,:])
        p0 = -1 * np.sum(h, axis=1)
        p0[:,0] += l + 2
        exp = 2 ** np.arange(l+2, dtype='object').reshape(-1,1)
        one_minus_h = -1 * h.copy()
        one_minus_h[:,:,0] += 1
        pow_2_p0 = np.sum(exp * one_minus_h, axis=1)
        pow_2_p0[:,0] += 1

        v = Trunc(Mul(pow_2_p0, v), l+2, 2)
        p = pmax - p0 + one_minus_d
        one_minus_z1 = -1 * x[2].copy()
        one_minus_z1[:,0] += 1
        one_minus_z2 = -1 * y[2].copy()
        one_minus_z2[:,0] += 1
        v = Mul(Mul(one_minus_z1, one_minus_z2), v) + Mul(x[2], y[0]) + Mul(y[2], x[0])
        z = EQZ(v, l)
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        p = Mul(Mul(Mul(one_minus_z1, one_minus_z2), p) + Mul(x[2], y[1]) + Mul(y[2], x[1]), one_minus_z)
        s = Mul(one_minus_b, Mul(a, y[3]) + Mul(one_minus_a, x[3])) + Mul(b, Mul(c, y[3]) + Mul(one_minus_c, x[3]))
        s = Mul(Mul(one_minus_z1, one_minus_z2), s) + Mul(Mul(one_minus_z1, y[2]), x[3]) + Mul(Mul(x[2], one_minus_z2), y[3])

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
    
    @jit(fastmath=True, cache=True)
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
    
    @jit(fastmath=True, cache=True)
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

        a = LT(x[1], y[1], k)
        c = EQ(x[1], y[1], k)
        one_minus_two_s1 = -2 * x[3].copy()
        one_minus_two_s1[:, 0] += 1
        one_minus_two_s2 = -2 * y[3].copy()
        one_minus_two_s2[:, 0] += 1
        one_minus_a = -1 * a.copy()
        one_minus_a[:, 0] += 1
        one_minus_c = -1 * c.copy()
        one_minus_c[:, 0] += 1
        d = LT(Mul(one_minus_two_s1, x[0]), Mul(one_minus_two_s2, y[0]), l+1)
        b_pos = Mul(c, d) + Mul(one_minus_c, a)
        b_neg = Mul(c, d) + Mul(one_minus_c, one_minus_a)

        one_minus_s1 = -1 * x[3].copy()
        one_minus_s1[:,0] += 1
        one_minus_s2 = -1 * y[3].copy()
        one_minus_s2[:,0] += 1
        one_minus_z1 = -1 * x[2].copy()
        one_minus_z1[:,0] += 1
        one_minus_z2 = -1 * y[2].copy()
        one_minus_z2[:,0] += 1

        b = Mul(Mul(x[2], one_minus_z2), one_minus_s2) + \
            Mul(Mul(one_minus_z1, y[2]), x[3]) + \
            Mul(Mul(one_minus_z1, one_minus_z2), \
                         Mul(x[3], one_minus_s2) + Mul(Mul(one_minus_s1, one_minus_s2), b_pos) + Mul(Mul(x[3], y[3]), b_neg))
        return b

    @jit(fastmath=True, cache=True)
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

        a = LTZ(x[1], k)
        pl = x[1].copy()
        pl[:,0] -= 1 - l
        b = LTZ(pl, k)
        one_minus_b = -1 * b.copy()
        one_minus_b[:, 0] += 1
        # print(- Mul(Mul(a, one_minus_b), x[1]))
        v2, pow_2_p_inv = Mod2m_shared_m(x[0], l, -1 * Mul(Mul(a, one_minus_b), x[1]))
        c = EQZ(v2, l)
        one_minus_c = -1 * c.copy()
        one_minus_c[:, 0] += 1
        mode_shares = np.zeros((1, num_parties), dtype='int')
        mode_shares[:,0] = mode
        v = x[0] - v2 + Mul(Mul(one_minus_c, pow_2_p_inv), XOR(mode_shares, x[3]))
        v_minus_pow2_l = v.copy()
        v_minus_pow2_l[:,0] -= 2 ** l
        d = EQZ(v_minus_pow2_l, l+1)
        one_minus_d = -1 * d.copy()
        one_minus_d[:, 0] += 1
        v = d * (2 ** (l-1)) + Mul(one_minus_d, v)
        one_minus_a = -1 * a.copy()
        one_minus_a[:, 0] += 1

        # To convert v = +/- 1 to float format, refine v = +/- 1 to 2 ^ (l-1)
        v = Mul( a, Mul(one_minus_b, v) + 2 ** (l-1) * Mul(Mul(b, mode_shares - x[3]), mode_shares - x[3]) ) + Mul(one_minus_a, x[0])
        #v = Mul( a, Mul(one_minus_b, v) + Mul(Mul(b, mode_shares - x[3]), v) ) + Mul(one_minus_a, x[0])
        one_minus_bmode = -1 * b * mode
        one_minus_bmode[:,0] += 1
        s = Mul(one_minus_bmode, x[3])
        z = OR(EQZ(v, l), x[2])
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
    
        v = Mul(v, one_minus_z)

        # To convert v = +/- 1 to float format, refine p with additional b

        p = Mul(x[1] + b + Mul(Mul(d, a), one_minus_b), one_minus_z)

        return (v, p, z, s)


    @jit(fastmath=True, cache=True)
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

        s = LTZ(a, gamma)
        z = EQZ(a, gamma)
        one_minus_2s = - 2 * s.copy()
        one_minus_2s[:,0] += 1
        a = Mul(one_minus_2s, a)
        a_bits = BitDec(a, gamma - 1, gamma - 1)
        b_bits = PreOr(a_bits[:, -1::-1, :])
        one_minus_b = -1 * b_bits.copy()
        one_minus_b[:,:, 0] += 1
        exp = 2 ** np.arange(gamma - 1, dtype='object').reshape(-1,1)
        v = a + Mul(a, np.sum(exp * one_minus_b, axis = 1))
        p = np.sum(b_bits, axis=1)
        p[:,0] -= gamma - 1
        v = Trunc(v, gamma - 1, gamma - l - 1) if gamma - 1 > l else 2 ** (l - gamma + 1) * v
        one_minus_z = -1 * z.copy()
        one_minus_z[:,0] += 1
        p[:,0] += gamma - 1 - l
        p = Mul(p, one_minus_z)

        return (v, p, z, s)


    @jit(fastmath=True, cache=True)
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

        b = BitDec(x[1], l, 1).reshape(n, -1)
        shared_zero = np.zeros_like(x[0], dtype='int')
        shared_l_0 = shared_zero.copy()
        shared_l_0[:,0] = l % 2
        c = XOR(b, shared_l_0)
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
        x2 = self.FLMul(x2, (one_minus_c * 2 ** (l-1) + c * Output(fl_sqrt2[0]), \
                        - one_minus_c * (l-1) + c * Output(fl_sqrt2[1]), shared_zero, shared_zero))
        one_minus_z1 = -1 * x[2].copy()
        one_minus_z1[:,0] += 1
        #print(Output(x2[1]), Output(p))
        p = Mul(x2[1] + p, one_minus_z1)

        v = Mul(x2[0], one_minus_z1)
        Error = x[3]

        return ((v, p, x[2], x[3]), Error)


    @jit(fastmath=True, cache=True)
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
        a = EQ(x[1], -1 * shared_l_minus_one, k)
        b = EQ(x[0], shared_pow2_l_minus_one, l)
        z = Mul(a, b)
        one_minus_z = z.copy()
        one_minus_z[:, 0] += 1
        v = Mul(v, one_minus_z)
        Error = OR(x[2], x[3])
        p = Mul(p, one_minus_z)


        return ((v, p, z, s), Error)

@jit(nopython=True, fastmath=True)
def Output(x):
    """ All parties open the shares

    """

    sum_temp = np.sum(x, axis=-1, keepdims=True)
    if sum_temp.ndim == 1:
        sum_temp = sum_temp.reshape(1,-1)
    return sum_temp 

@jit(nopython=True, fastmath=True) 
def RandInt(k, n=1, num_parties=3):
    """ Generate n random interger for each party without interaction
        
    """

    result = np.zeros((n, num_parties)).astype('int64').astype('object')
    for i in range(n):
        for j in range(num_parties):
            result[i,j] = random.randint(0, 2 ** (k % 64))
    return result

    # return np.random.randint(0, 2 ** (k % 64), (n, num_parties), dtype='int64').astype('object')

@jit(nopython=True, fastmath=True)  
def RandBit(n, num_parties):
    """ generate n random shared bits each row is a shared bits, and each column is shares of a party 
        use determined shares for test only
        
    """
        #result = np.zeros((n,3), dtype='int')
        #for row in np.random.choice(list(range(n)),np.random.randint(n), replace=False):
        #    column = np.random.randint(0,3,1)
        #    result[row, column] = 1
        #return result
    result = np.zeros((n, num_parties)).astype('int64')
    for i in range(n):
        for j in range(num_parties - 1):
            result[i,j] = np.random.randint(-10, 10)
        result[i, -1] = np.random.randint(0, 2) - np.sum(result[i,:-1])
    
    # result = np.random.randint(-10,10, (n, num_parties))
    # result[:,-1] = (np.random.randint(0, 2, n) - np.sum(result[:,:-1], axis=1))
    
    return result

@jit(nopython=True, fastmath=True) 
def RandM(self, k, m, n=1, num_parties=3):
    """ Generate n shared random integers rp and rpp, together with the shared bits of rp
            # rpp from [0, 2^(k-m) - 1]
            # rp generated from m bits 
    """

    rpp = RandInt(k+2-m, n, num_parties)
    b = RandBit(m*n, num_parties).reshape(n, m, -1)
    # rp = np.array(list(map(lambda x: np.sum( (np.logspace(0,m-1,num=m,base=2,dtype='int64').reshape(-1,1) * x), axis=0 ), b)), dtype='object')
    
    rp_base = np.logspace(0,m-1,num=m,base=2,dtype='object').reshape(-1,1)
    rp = np.empty_like(rpp, dtype='object')
    for i in range(num_parties):
        for j in range(n):
            rp[j][i] = np.sum(rp_base * b[j][:][i])

    return rpp, rp, b
    
@jit(nopython=True, fastmath=True)
def Mul(self, x, y):
    """ Secret shared multiplication of two sets of shared x and y

    """
        #print('x', x, '\n\n\n', 'y', y)

    num_parties = x.shape[1]
        
    fake_triples = np.array([[0] * (num_parties - 1) + [1], [1] * num_parties, [1] * num_parties], dtype='int64')
    alpha = Output(x - fake_triples[0])
    beta = Output(y - fake_triples[1])
        #print(alpha, '\n\n', beta )

    prod = fake_triples[2] + alpha * fake_triples[1] + beta * fake_triples[0]
    prod[:,0] += np.squeeze(alpha * beta)
    return prod

def Prod(x):
    """ 
    Return secret shared product x[0]*x[1]*..*x[k-1] of each of the n sets 
    x.shape = (num_sets, num_integers, num_parties)

    """

    n = x.shape[0]
    k = x.shape[1]
    # num_parties = x.shape[2]

    if k == 1:
        return x
    else:
        l = k - k % 2
        left = [i+j*k for j in range(n) for i in range(l//2)]
        right = [i+j*k for j in range(n) for i in range(l//2, l)]
        result = Mul(x.reshape(n*k,-1)[left], x.reshape(n*k,-1)[right]).reshape(n,l//2,-1)
        return Prod(result) if l == k else Prod(np.concatenate((result, x[:,-1,:].reshape(n,1,-1)), axis=1))
    
def MulPub(x, y):
    """ The parties generate a pseudo-random sharing of zero(PRZS) 
        then each party compute a randomized product of shares [z] = [x][y] + [0], and exchange the shares
        to reconstruct z = xy
        
    """
    # Here PRZS is ignored, compute [x][y] directly 
    return Output(Mul(x,y))

@jit(nopython=True, fastmath=True)    
def XOR(a, b):
    return a + b - 2 * Mul(a, b)

@jit(nopython=True, fastmath=True)
def OR(a, b):
    return a + b - Mul(a, b)


def Inv(a):
    """ Return 1/a

    """

    # if a.ndim == 1:
        
    a = a.reshape(1, -1)
    n = a.shape[0]
    num_parties = a.shape[1]

    r = np.random.randint(1,17,(n, num_parties))
    c = MulPub(r, a)
    return r/c

def PreMul(x):
    """ # Calculate prefix multiplication where [p_i] = Mul([x_0], [x_1]. ..., [x_i])
        # x.shape = (num_sets, num_integers, num_parties)
    
    """
    n = x.shape[0]
    k = x.shape[1]
    if k == 1:
        return x
    return _PreMul(x)

    
    
@jit(nopython=True, fastmath=True)
def _PreMul(x):
    
    num_parties = x.shape[1]

    r = np.random.randint(1, 17, (n*k, num_parties))
    s = np.random.randint(1, 17, (n*k, num_parties))
    #u = np.sum(r,axis=1) * np.sum(s,axis=1)
    u = MulPub(r,s)
    # ignore 1st element of each of the k sets of r, and k-st element of each of s
    r_idx = [i for i in range(n*k) if i % k != 0]
    su_idx = [i for i in range(n*k) if i % k != k-1]
    v = Mul(r[r_idx],s[su_idx])
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
    m = MulPub(w.reshape(n*k,-1),x.reshape(n*k,-1)).reshape(n,k,-1)
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


@jit(nopython=True, fastmath=True)
def PreOr(a):
    """ # Calculate prefix Or of n sets of k shared bits
        # a.shape = (num_bits_set, num_bits, num_parties)
    
    """
    n = a.shape[0]
    k = a.shape[1]
    num_parties = a.shape[2]

    if k == 1:
        return a
    ap = np.concatenate( (a[:,:,0].reshape(n,k,1)+1, a[:,:,1:]), axis=2) 
    b = PreMul(ap)
    p = (np.concatenate( (a[:,0,:].reshape(n,1,-1),  -1 * Mod2(b[:,1:,:].reshape(n*(k-1),-1), k).reshape(n,k-1,-1) ), axis=1))
    p[:,1:,0] += 1
    return p

@jit(nopython=True, fastmath=True)    
def KOrCS(a):
    """ k-ary Or of k shared bits
        Cannot find algorithm within corresponding reference, so implemented similar to PreOr instead
        a.shape = (num_bits_set, num_bits, num_parties)
    
    """
    n = a.shape[0]
    k = a.shape[1]
    num_parties = a.shape[2]

    if k == 1:
        return a
    #b = PreMul(a+1)
    #return 1 - Mod2(b[-1], k)
    ap = np.concatenate( (a[:,:,0].reshape(n,k,1)+1, a[:,:,1:]), axis=2)
    b = Prod(ap)
    #return 1 - Mod2(b, k)
    result = -1 * Mod2(b.reshape(n,-1),k).reshape(n,1,-1)
    result[:,:,0] += 1
    return result
    
# def Bits(x, m):
#     """ convert integer x to binary representation of m+1 least bits
    
#     """
#     #print(x)
#     x_bits = bin(x)[2:] if x >= 0 else bin(x)[3:]
#     x_bits = np.array([b for b in x_bits[::-1]], dtype='int')
#     x_length = x_bits.shape[0]
#     if x_length >= m:
#         return x_bits[:m].reshape(-1,1)
#     else:
#         return np.concatenate( (x_bits, np.zeros((m - x_length), dtype='int')), axis=0 ).reshape(-1,1)




@jit(nopython=True, fastmath=True)
def Bits(x, m):
    """ convert integer x to binary representation of m+1 least bits
    
    """
    #print(x)
#     x_bits = bin(x)[2:] if x >= 0 else bin(x)[3:]
    if x < 0:
        x *= -1
    x_bits = []
    while(x != 0):
        bit_temp = x % 2
        x_bits.append(bit_temp)
        x = (x - bit_temp) // 2
    
    x_bits = np.asarray(x_bits)
#     x_bits = np.array([b for b in x_bits[::-1]], dtype='int')
    x_length = x_bits.shape[0]
    if x_length >= m:
        return x_bits[:m].reshape(-1,1)
    else:
        return np.concatenate( (x_bits, np.zeros((m - x_length), dtype='int')), axis=0 ).reshape(-1,1)
    
@jit(nopython=True, fastmath=True)
def KOr(a):
    """ # k-ary Or operation with low complexity
        # transform k bits to logk bits and compute k-ary Or of the logk bits
        # a.shape = (num_sets, num_bits, num_parties)
    
    """
    n = a.shape[0]
    k = a.shape[1]
    num_parties = a.shape[2]
    # m = ceil(log_2(k)) + 1 rather than ceil(log_2(k)) because when k = 2^m , m bits can only represent
    # x \in [0,k-1] while sum(a) \in [0,k]. When sum(a) is k, the result will be incorrect
    m = np.int64(np.ceil(np.log2(k))) + 1
    rpp, rp, b = RandM(k, m, n, num_parties)
    c = Output((2 ** m) * rpp + rp + np.sum(a, axis=1))
    #c = Output((2 ** m) * rpp + rp)[0,0] + Output(np.sum(a, axis=0))[0,0] % k
    #c_bits = np.frompyfunc(Bits, 2, 1)(c, m)
    #print(c)
    c_bits = np.frompyfunc(Bits, 2, 1)(c, m)
    c_bits = np.array([arr[0] for arr in c_bits])
    #print(c_bits)
    #print(b)
    #d = c_bits + b[0] - 2 * c_bits * b[0]
    d = np.concatenate((c_bits, np.zeros((n, m, num_parties - 1), dtype='int')), axis=2) + b - 2 * c_bits * b
    #print(d)
    return KOrCS(d)

@jit(nopython=True, fastmath=True)  
def Mod2(x, k):
    """ Extract the least significant bit of x \in Z_k
    
    """
    
    n = x.shape[0]
    num_parties = x.shape[1]
    
    rpp, rp, b = RandM(k,1,n,num_parties)
    rp_0 = b[:,0,:]
    c = 2**(k) + Output( x + 2*rpp + rp_0)
    c_0 = c % 2
    return np.concatenate((c_0, np.zeros((n, num_parties - 1), dtype='int')), axis=1)+ rp_0 - 2 * c_0 * rp_0 

    
@jit(nopython=True, fastmath=True)
def BitLT(a, b):
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
    num_parties = b.shape[2]

    d = np.concatenate( (a+1, np.zeros((n, k, num_parties - 1), dtype='int')), axis=2 ) + b - 2 * a * b
    p = PreMul(d[:,-1::-1,:])
    p = p[:,-1::-1, :]
    s = np.concatenate( ( p[:,:-1,:] - p[:,1:,:], p[:,-1,:].reshape(n,1,-1)), axis=1 )
    s[:,-1,0] -= 1
    s = np.sum( (1 - a) * s, axis=1)
    return Mod2(s, k)

@jit(nopython=True, fastmath=True)
def Mod2m(x, k, m):
    """ Compute x mod 2^m
        x.shape = (num_integers, num_parties)
        k is a constant
        m is a constanct
    """
    n = x.shape[0]
    num_parties = x.shape[1]
    rpp, rp, b = RandM(k, m, n, num_parties)
    #c = self.Output( x + (2 ** m) * rpp + rp)
    c = 2**(k-1) + Output( x + (2 ** m) * rpp + rp)
    #print(c)
    cp = c % (2 ** m)
    c_bits = np.frompyfunc(Bits, 2, 1)(c, m)
    c_bits = np.array([arr[0] for arr in c_bits])
    u = BitLT(c_bits, b)
    ap = np.concatenate( (cp, np.zeros((n, num_parties - 1), dtype='int')), axis=1 ) - rp + 2 ** m * u
    return ap
    
@jit(nopython=True, fastmath=True)
def Mod2m_shared_m(x, k, m):
    """ Compute x mod 2^m where m is shared
        x.shape = (num_integers, num_parties)
        k is a constant
        m.shape = (num_integers, num_parties)
    """
    m_unary_bits = B2U(m, k)
    m_pow2 = Pow2(m, k)
    # m_pow2_inv = self.Inv(m_pow2)
    n = x.shape[0]
    num_parties = x.shape[1]
    rpp, rp, b = RandM(k, k, n, num_parties)
    s = 2 ** np.arange(k, dtype='object').reshape(-1,1)
    #print(m_unary_bits.shape, b.shape)
    rp = np.sum(s * Mul(m_unary_bits.reshape(-1, num_parties), b.reshape(-1, num_parties)).reshape(b.shape), axis=1)
    m_not = - m_unary_bits
    m_not[:,:,0] += 1

    rpp = 2 ** k * rpp + np.sum(s * Mul(m_not.reshape(-1, num_parties), b.reshape(-1, num_parties)).reshape(b.shape), axis=1)
    c = Output(x + rpp + rp)
    cp = (c % s.T).reshape(n, k, -1)
    cpp = np.sum(cp[:,1:,:] * (m_unary_bits[:,:-1,:] - m_unary_bits[:,1:,:]), axis=1)
    d = LT(cpp, rp, k)
    result = cpp - rp + Mul(m_pow2, d)
    result_int = np.frompyfunc(int,1,1)(result)
    result_int[:,0] += np.frompyfunc(int,1,1)(np.sum(result - result_int, axis=1))
    return result_int, m_pow2

@jit(nopython=True, fastmath=True)
def Trunc(x, k, m):
    """ Coupute x / 2^m
    
    """
    ap = Mod2m(x, k, m)
    d = (x - ap) * 2 ** (-m)
    dint = np.frompyfunc(int, 1, 1)(d)
    dint[:,0] += np.frompyfunc(int, 1, 1)(np.sum(d - dint, axis=1))
    return dint

    
@jit(nopython=True, fastmath=True)
def EQZ(x, k):
    """ Determine if x equals to zero or not
    """
    
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n = x.shape[0]
    num_parties = x.shape[1]

    rpp, rp, b = RandM(k, k, n, num_parties)
    # Using x + 2^k * rpp + rp instead of 2^(k-1) + (x + 2^k * rpp + rp) because 2^(k-1) will introduce 1 into
    # the k-th bit thus c_bits differs from rp while x is zero. It's not clear why 2^(k-1) is needed in Protocol
    # 3.7 of "Improved Primitives for Secure Multiparty Integer Computation"
    c = Output( x + 2 ** k * rpp + rp)
    c_bits = np.frompyfunc(Bits, 2, 1)(c, k)
    c_bits = np.array([arr[0] for arr in c_bits])
    #d = c_bits + b - 2 * c_bits * b
    d = np.concatenate((c_bits, np.zeros((n, k, num_parties - 1), dtype='int')), axis=2) + b - 2 * c_bits * b
    #result = np.concatenate([ - KOr(di.reshape(1,k,3)) for di in d], axis=0)
    result = -1 * KOr(d).reshape(n,-1)
    result[:,0] += 1
    return result
    
def LTZ(x, k):
    """ Determine if x is less than zero or not
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # n = x.shape[0]
    return -1 * Trunc(x, k, k-1)

def EQ(x, y, k):
    """ Determine if x == y or not
    """
    return EQZ(x - y, k)

def LT(x, y, k):
    """ Determine if x < y or not
    """
    return LTZ(x - y, k)

    
    
@jit(nopython=True, fastmath=True)
def BitAdd(a, b):
    """ Output the shared bits of a + b, a is clear and b is shared   
        a.shape = (num_integers, 1)
        b.shape = (num_integers, k, num_parties)
        k = num_bits
    """
    n = b.shape[0]
    k = b.shape[1]
    num_parties = b.shape[2]

    exp = 2 ** np.arange(k, dtype='object').reshape(k,1)
    #a_integer = np.sum(a * exp, axis=1)
    b_integer = np.sum(np.sum(b, axis=2, keepdims=True) * exp, axis=1)
    c = a + b_integer
    #print(a, b_integer, c)
    c_bits = np.frompyfunc(Bits, 2, 1)(c, k+1) 
    c_bits = np.array([arr[0] for arr in c_bits])
    result = np.random.randint(-10,10, (n, k+1, num_parties))
    result[:,:,-1] = c_bits[:,:,0] - np.sum(result[:,:,:-1], axis=2)
    return result

    
@jit(nopython=True, fastmath=True)
def BitDec(x, k, m):
    """ bit decomposition of m least significant bits of x represented in complementary format
    """
    n = x.shape[0]
    num_parties = x.shape[1]

    rpp, rp, b = RandM(k, m, n, num_parties)
    c = 2 ** k + Output(x - rp + (2 ** m) * rpp)
    #c_bits = np.frompyfunc(Bits, 2, 1)(c, m) 
    #c_bits = np.array([arr[0] for arr in c_bits])
    #print(c)
    #print(Output(rp))

    return BitAdd(c % 2**m, b)[:,:-1]

    
@jit(nopython=True, fastmath=True)
def Pow2(x, k):
    """ Output shares of 2^x where x \in [0, k)
    """
    m = np.int64(np.ceil(np.log2(k)))
    x_bits = BitDec(x, m, m)
    d = (2 ** (2 ** np.arange(m, dtype='object'))).reshape(-1,1) * x_bits - x_bits
    d[:,:,0] += 1
    return Prod(d).reshape(x.shape)

@jit(nopython=True, fastmath=True)
def B2U(x, k):
    """ Conversion 0 <= x < k from binary to unary bitwise representation where x least significant bits are 1
        and all other k - x bits are 0
    """
    n = x.shape[0]
    num_parties = x.shape[1]

    x_pow2 = Pow2(x, k)
    rpp, rp, b = RandM(k, k, n, num_parties)
    c = Output(x_pow2 + 2 ** k * rpp + rp)
    c_bits = np.frompyfunc(Bits, 2, 1)(c, k) 
    c_bits = np.array([arr[0] for arr in c_bits])
    d = np.concatenate((c_bits, np.zeros((n, k, num_parties - 1), dtype='int')), axis=2) + b - 2 * c_bits * b
    y = - 1 * PreOr(d)
    y[:,:,0] += 1
    return y
    
    
@jit(nopython=True, fastmath=True)
def Trunc_shared_m(x, k, m, num_parties):
    """ Perform trunction of [x] by an unknown number of bits [m], same as x // m, where k is the bitlength of x
        x.shape = (num_integers, num_parties)
    """
    n = x.shape[0]
    num_parties = x.shape[1]
    
    m_unary_bits = B2U(m, k)
    m_pow2_inv = Inv(Pow2(m, k))
    
    rpp, rp, b = RandM(k, k, n, num_parties)
    s = 2 ** np.arange(k, dtype='int64').reshape(-1,1)
    rp = np.sum(s * Mul(np.tile(m_unary_bits.reshape(-1, num_parties), (n,1)), b.reshape(-1, num_parties)).reshape(b.shape), axis=1)
    m_not = - m_unary_bits
    m_not[:,:,0] += 1

    rpp = 2 ** k * rpp + np.sum(s * Mul(np.tile(m_not.reshape(-1, num_parties), (n,1)), b.reshape(-1, num_parties)).reshape(b.shape), axis=1)
    c = Output(x + rpp + rp)
    cp = (c % s.T).reshape(n, k, -1)
    cpp = np.sum(cp[:,1:,:] * (m_unary_bits[:,:-1,:] - m_unary_bits[:,1:,:]), axis=1)
    d = LT(cpp, rp, k)
    result = Mul(x - cpp + rp, m_pow2_inv) - d
    result_int = np.frompyfunc(int,1,1)(result)
    result_int[:,0] += np.frompyfunc(int,1,1)(np.sum(result - result_int, axis=1))
    return result_int


    
@jit(nopython=True, fastmath=True)
def TruncPr(self, x, k, m):
    """ Return d = floor(x / 2 ^ m) + u where u is a random bit and Pr(u = 1) = (x mod 2 ^m) / 2 ^ m
        The protocol rounds  x / 2 ^ m to the nearest integer with probability 1 - alpha, alpha is the distance 
        between x / 2 ^ m and the nearest integer
        0 < m < k
    """
    #y = x.copy()
    #y[:,0] += 2 ** (k - 1)
    n = x.shape[0]
    num_parties = x.shape[1]
    rpp, rp, b = RandM(k, m, n, num_parties)
    c = 2 ** (k - 1) + Output(x + 2 ** m * rpp + rp)
    cp = c % (2 ** m)
    xp = -1 * rp
    #print(cp.dtype)
    xp[:,0] += cp[:,0]
    d = (x - xp) / (2 ** m)
    dint = np.frompyfunc(int, 1, 1)(d)
    dint[:,0] += np.frompyfunc(int, 1, 1)(np.sum(d - dint, axis=-1))
    return dint

    
@jit(nopython=True, fastmath=True)
def SDiv(a, b, k):
    theta = np.int32(np.ceil(np.log2(k)))
    x = b.astype(object)
    y = a.astype(object)
    exp_k = np.zeros_like(x)
    exp_k[:,0] += 2 ** (k + 1)
    for i in range(theta - 1):
        #print(i, 1,'\n','y: ', y, '\n', 'x: ', x)
        y = Mul(y, exp_k - x)
        #print(i, 2,'\n','y: ', y, '\n', 'x: ', x)
        y = TruncPr(y, 2 * k + 1, k)
        #print(i, 3,'\n','y: ', y, '\n', 'x: ', x)
        x = Mul(x, exp_k - x)
        #print(i, 4,'\n','y: ', y, '\n', 'x: ', x)
        x = TruncPr(x, 2 * k + 1, k)
        #print(i, 5,'\n','y: ', y, '\n', 'x: ', x)
    y = Mul(y, exp_k - x)
    y = TruncPr(y, 2 * k + 1, k)
    #print(y)   
    return y
