import numpy as np
import scipy.linalg
import functools, sys, os, math, itertools, operator, six
from sympy.utilities.iterables import multiset_permutations, ordered_partitions, kbins
from sympy.ntheory import mobius, divisors
from sympy import Rational
import sympy
import unittest
import pyparsing

#This module provides free Lie algebra calculations in the manner of
#the first part of Reutenauer, on an alphabet of positive integers and
#using float (or sympy expressions) as the "field" K.
#Hopefully the code and definitions of most functions look similar.
#It is not written to be efficient, but to be hackable.
#Works on Python 2 or 3, but not caring about inefficiencies caused by
#things like "range" and "items" on Python 2

#The test() function at the bottom of this file illustrates the kind of calculations you can do.

#The main objects include Elt (which represents an element of tensor space)
#and EltElt (which represents Elts tensored together).
#This code is written defensively around them: functions do not modify
#such inputs, but they do not assume that their inputs are immutable.
#For example, return values do not share references to parts of their inputs.
#Words and coefficients, however, are considered immutable.
#The Elt object represents both an element of tensor space and its dual
#(through the function dotprod), which
#provides flexibility but is arguably not algebraically nice.

#There's a simple string representation of an Elt
#(which you go *to* via pretty() and *from* via parse() )
#in which coefficients are surrounded by square brackets.
#For example "[1]" is the unit element and "[3]12-[3]21" is a multiple of signed area.

#If you want to use your own coefficient as K, make it something hashable,
#something with operators +, -, * and ==,
#and make sure * accepts float and int as well as itself
#and change the next five functions.
#The "distance" function won't work, but you can sort that yourself.
def assert_coefficient(c):
    #this should accept float and int as well as a custom coefficient
    #note that isinstance(c,float) includes numpy floating point types as well as actual float
    assert isinstance(c, (float, int, sympy.Basic)), (c, type(c))
    #assert type(c) in (float, int), c
def unit_coefficient():
    return 1
def zero_coefficient():
    return 0
def isunit_coefficient(c):#should accept float and int as well as custom coefficient
    return 1==c
def iszero_coefficient(c):#should accept float and int as well as custom coefficient
    return 0==c

_defaultMaxLevel=None
def _getMaxLevel(level):
    if level is None:
        return _defaultMaxLevel
    return level

_defaultUseRational=False
def _reciprocateInteger(i,useRational=None):
    if useRational is None:
        useRational=_defaultUseRational
    if useRational:
        return Rational(1,i)
    return 1.0/i

class Word:
    """The alphabet is int. This class represents an immutable word on the alphabet"""
    def __init__(self, letters):
        self.letters=tuple(int(i) for i in letters)
    def __hash__(self):
        return hash(self.letters)
    def __eq__(self,other):
        return self.letters==other.letters
    def __repr__(self):
        return "".join(str(i) for i in self.letters)
    def crudeNumber(self):
        """for small d, the integer whose decimal representation is str(self)"""
        return functools.reduce(lambda x, y: 10*x+y, self.letters)
emptyWord=Word([])
def concatenate(a,b):
    assert isinstance(a,Word) and isinstance(b,Word), (a,b)
    return Word(a.letters+b.letters)

#element of tensor space
class Elt:
    """An element of the tensor algebra on the alphabet.
       data is a list (one for each level) of dictionaries word->coefficient.
       The * operator is only used for multiplication by a scalar
    """
    def __init__(self,data):
        assert type(data)==list, data
        self.data=data
    def __hash__(self):
        return hash( tuple( frozenset(level_dict.items()) for level_dict in self.data) )
    def __eq__(self,other):
        #return self.data==other.data
        assert isinstance(other,Elt), other
        for a,b in six.moves.zip_longest(self.data, other.data,fillvalue=dict()):
            for k,v in a.items():
                if v!=b.get(k,0):
                    return False
            for k,v in b.items():
                if k not in a and not iszero_coefficient(v):
                    return False
        return True
    def __repr__(self):
        return "E"+str(self.data)
    def __rmul__(self, scale):
        assert_coefficient(scale)
        return self*scale
    def __mul__(self, scale):
        assert_coefficient(scale)
        if iszero_coefficient(scale):
            return zeroElt
        out=[{k:scale*v
              for k,v in x.items()}
             for x in self.data]
        return Elt(out)
    def __add__(self, b):
        assert isinstance(b,Elt), b
        if len(self.data)<len(b.data):
            return b+self
        out=[i.copy() for i in self.data]
        for level,d in enumerate(b.data):
            for w,v in d.items():
                _increment_value_in_dict_to_coeff(out[level],w,v)
        return Elt(out)
    def __sub__(self,b):
        assert isinstance(b,Elt), b
        return self+(-1*b)
    def __neg__(self):
        return -1*self
    def truncatedToLevel(self, level):
        return Elt([i.copy() for i in self.data[:level+1]])
    def restrictedToLevel(self, level):
        if len(self.data)<level+1:
            return Elt([dict()])
        o=[dict() for i in range(1+level)]
        o[-1]=self.data[level].copy()
        return Elt(o)
    def maxLetter(self):
        """get the maximum letter used"""
        l=1
        for i in self.data:
            for j in i:
                for k in j.letters:
                    if l<k:
                        l=k
        return l
    def pretty(self, dp=15, tol=None):
        """a pretty string representation"""
        #This could be made __repr__ if we trust it
        if tol is None:
            tol=10**(-dp)
        if dp is None:
            formatString = "[{}]"
        else:
            formatString = "[{:."+str(dp)+"g}]"
        def item(i,j):
            if math.fabs(j)<tol:
                return ""
            sign = ("+" if j>=0 else "-")
            omitCoeff = len(i.letters)!=0 and (j==1 or j==-1)
            #numpy floats inherit from float
            formatString2Use = formatString if isinstance(j,(float,sympy.Float)) else "[{}]"
            coeff = ("" if omitCoeff else formatString2Use.format(j if j>=0 else -j))
            if len(i.letters)!=0 and coeff=="[1]":
                coeff=""
            lets = "".join(str(j) for j in i.letters)
            return sign+coeff+lets       
        o= "".join(item(i,a[i]) for a in self.data for i in sorted(a,key=lambda x:x.letters))
        if len(o)>0 and o[0]=="+":
            return o[1:]
        return o
    def prettySympy(self):
        o=""
        for lev in self.data:
            for k in sorted(lev,key=lambda x:x.letters):
                vv=sympy.expand(lev[k])
                if vv!=0:
                    o=o+("+[{}]{}".format(vv,k))
        if len(o)>0 and o[0]=="+":
            return o[1:]
        return o
        

#TODO: almost anywhere this function is used in a loop
#is an optimisation opportunity
def word2Elt(word):
    if type(word) in (str,tuple):
        word=Word(word)
    assert isinstance(word,Word),word
    a=[dict() for i in range(1+len(word.letters))]
    a[-1]={word:unit_coefficient()}
    return Elt(a)
def letter2Elt(letter):
    return Elt([dict(),{Word((letter,)):unit_coefficient()}])
    
unitElt = Elt([{emptyWord:unit_coefficient()}])
zeroElt = Elt([{emptyWord:zero_coefficient()}])

def removeTinies(a):
    """a version of an Elt with tiny elements removed"""
    assert isinstance(a,Elt), a
    d=[{k:v for k,v in i.items() if math.fabs(v)>1e-15} for i in a.data]
    return Elt(d)
    
def wordIter(d,m, topOnly=False, asNumbers=False):
    """return all words with up to or exactly m of the  d letters"""
    from itertools import chain, product
    alphabet=range(1,d+1) if asNumbers else "123456789"[:d]
    if topOnly:
        return product(alphabet, repeat=m)
    it=chain.from_iterable(product(alphabet, repeat=r) for r in range(m+1))
    return it
    
def randomElt(d,m,maxi=None):
    """a random Elt on d letters with m levels. If maxi not given, uniform[0,1] coeffs."""
    letters=range(1,d+1)
    ran = lambda:np.random.rand() if maxi is None else np.random.randint(maxi)
    out=[{Word(key):ran() for key in itertools.product(letters,repeat=lev)}
              for lev in range(0,m+1)]
    return Elt(out)

class EltElt:
    """An element of the tensor product of the tensor algebra n times with itself.
       data is a dictionary of (word,word,...)->coefficient"""
    def __init__(self, data, n):
        self.n=n
        assert type(data)==dict, data
        for k,v in data.items():
            assert type(k)==tuple and len(k)==n
            for i in k:
                assert isinstance(i,Word),(data,n,i,k)
        self.data=data
    def __eq__(self,other):
        return self.data==other.data
    def __repr__(self):
        return "EE"+str(self.data)
    def get_deg(self):
        if 0==len(self.data):
            return 0
        return max(sum(len(i.letters) for i in k) for k in self.data)
    def __rmul__(self, scale):
        assert_coefficient(scale)
        return self*scale
    def __mul__(self, scale):
        assert_coefficient(scale)
        if scale==0:
            return EltElt(dict(),self.n)
        out={k:scale*v
              for k,v in self.data.items()}
        return EltElt(out,self.n)
    def __add__(self, b):
        assert isinstance(b,EltElt), (b)
        assert self.n == b.n
        out=self.data.copy()
        for k,v in b.data.items():
            _increment_value_in_dict_to_coeff(out,k,v)
        return EltElt(out,self.n)
    def __neg__(self):
        return -1*self
    def __sub__(self,b):
        assert isinstance(b,EltElt), b
        assert self.n == b.n
        return self+(-1*b)
    def truncatedToTotalLength(self,total):
        out={k:v for k,v in self.data.items() if sum(len(i.letters) for i in k)<=total}
        return EltElt(out,self.n)
    def truncatedToLengths(self,lengths):
        """remove components where nth elt longer than lengths[n]"""
        assert self.n == len(lengths)
        out={k:v for k,v in self.data.items() 
             if all(j is None or len(i.letters)<=j for i,j in zip(k,lengths))}
        return EltElt(out,self.n)
    def restrictedToLengths(self,lengths):
        """remove components unless nth elt's length is lengths[n]"""
        assert self.n == len(lengths)
        out={k:v for k,v in self.data.items() 
             if all(j is None or len(i.letters)==j for i,j in zip(k,lengths))}
        return EltElt(out,self.n)
    def pretty(self):
        def key(x):
            length = sum(len(aa.letters) for aa in x)
            return (length,tuple(aa.letters for aa in x))
        a=sorted(self.data, key=key)
        #return [(self.data[i],i) for i in a]
        return " ".join("{:+}{}".format(self.data[i],i) for i in a)
        
def get_coefficient(a,word):
    """return the coefficient of the Word word in the Elt a"""
    assert isinstance(a,Elt),a
    assert isinstance(word,Word),word
    level=len(word.letters)
    if level<len(a.data):
        return a.data[level].get(word,zero_coefficient())
    return zero_coefficient()

def epsilon_numeric(a):
    """The coefficient of the empty word in the Elt a"""
    assert isinstance(a,Elt),a
    if len(a.data)==0 or emptyWord not in a.data[0]:
        return 0
    return a.data[0][emptyWord]

def epsilon(a):
    """The coefficient of the empty word in the Exp a, as an Elt"""
    assert isinstance(a,Elt),a
    return Elt(a.data[:1])

def _increment_value_in_dict_to_coeff(dict_,k,v):
    if k in dict_:
        if iszero_coefficient(dict_[k]+v):
            del dict_[k]
        else:
            dict_[k]+=v
    elif not iszero_coefficient(v):
        dict_[k]=v

def dotprod(a,b):
    """The scalar product of the Elts a and b in the word basis"""
    assert isinstance(a,Elt) and isinstance(b,Elt), (a,b)
    out=zero_coefficient()
    for x,y in zip(a.data,b.data):
        for k in x:
            if k in y:
                out += x[k]*y[k]
    return out

def make_dual(a, returnElt=True):
    """Turn the Elt a into the function mapping b to dotprod(a,b)
    We use Elts both for tensor space and its dual, so this makes sense.
    Returning an Elt by default makes sense because can pass to tensorProductFunctions"""
    #alternative - let conc take coefficients as well as Elts.
    assert isinstance(a,Elt), a
    def loc_dual(b):
        d=dotprod(a,b)
        if returnElt:
            return d*unitElt
        return d
    return loc_dual

def distance(a,b):
    """The distance between the Elts a and b in the word basis"""
    assert isinstance(a,Elt) and isinstance(b,Elt), (a,b)
    d=a-b
    return math.sqrt(dotprod(d,d))

def concatenationProduct(a,b,maxLevel=None):
    """The concatenation product of the Elts a and b. 
    This is the _internal_ tensor product in tensor space.
    Levels above maxLevel, if provided, are ignored."""
    assert isinstance(a,Elt) and isinstance(b,Elt), (a,b)
    topLevel = (len(a.data)-1)+(len(b.data)-1)
    maxLevel = _getMaxLevel(maxLevel)
    if maxLevel is None or maxLevel>topLevel:
        maxLevel = topLevel
    out=[dict() for i in range(maxLevel+1)]
    for level in range(0,maxLevel+1):
        for alevel in range(0,min(level+1,len(a.data))):
            blevel=level-alevel
            if blevel >= len(b.data) or b.data[blevel] is None or a.data[alevel] is None:
                continue
            for l1,l2 in a.data[alevel].items():
                for r1,r2 in b.data[blevel].items():
                    prod=l2*r2
                    w = concatenate(l1,r1)
                    _increment_value_in_dict_to_coeff(out[level],w,prod)
    return Elt(out)

def concatenationProductMany(a, maxLevel=None):
    """The concatenation product of many Elts (in the iterable a) all together"""
    return functools.reduce(
        lambda x,y : concatenationProduct(x,y,maxLevel),a)

def shuffleProduct(a,b,maxLevel=None):
    """The shuffle product of two Elts"""
    assert isinstance(a,Elt) and isinstance(b,Elt), (a,b)
    topLevel = (len(a.data)-1)+(len(b.data)-1)
    maxLevel = _getMaxLevel(maxLevel)
    if maxLevel is None or maxLevel>topLevel:
        maxLevel = topLevel
    out=[dict() for i in range(maxLevel+1)]
    for level in range(0,maxLevel+1):
        for alevel in range(0,min(level+1,len(a.data))):
            blevel=level-alevel
            if blevel >= len(b.data) or b.data[blevel] is None or a.data[alevel] is None:
                continue
            source=(0,)*alevel + (1,)*blevel
            for l1,l2 in a.data[alevel].items():
                for r1,r2 in b.data[blevel].items():
                    prod=l2*r2
                    out_=np.zeros(level,dtype="int32")
                    if level==0:
                        _increment_value_in_dict_to_coeff(out[0],emptyWord,prod)
                    else:
                        for mask in multiset_permutations(source):
                            mask=np.array(mask)
                            np.place(out_,1-mask,l1.letters)
                            np.place(out_,mask,r1.letters)
                            w = Word(out_)
                            _increment_value_in_dict_to_coeff(out[level],w,prod)
    return Elt(out)

def shuffleProductMany(a, maxLevel=None):
    """The shuffle product of many Elts (in the iterable a) all together"""
    return functools.reduce(
        lambda x,y : shuffleProduct(x,y,maxLevel),a)

def rightHalfShuffleProduct(a,b,maxLevel=None):
    """For two words a and b, their rightHalfShuffle is those shuffles
    of a and b for which the last element is the last element of b.
    This is extended to a bilinear operation on Elts.
    If c is a letter then rightHalfShuffle(a,bc) is (a shuffle b)c.
    Usually (a shuffle b) == rightHalfShuffle(a,b)+rightHalfShuffle(b,a) (*)
    In the current implementation, rightHalfShuffle(a,b) is zero if b is the empty word,
    even if a is the empty word.
    Note that this means that (*) is violated if a and b are both the empty word.
    This operation is often denoted $\mathbin{\succ}$, being a dendriform algebra operation.
    It is not mentioned in the book."""

    assert isinstance(a,Elt) and isinstance(b,Elt), (a,b)
    topLevel = (len(a.data)-1)+(len(b.data)-1)
    maxLevel = _getMaxLevel(maxLevel)
    if maxLevel is None or maxLevel>topLevel:
        maxLevel = topLevel
    out=[dict() for i in range(maxLevel+1)]
    for level in range(0,maxLevel+1):
        for alevel in range(0,min(level+1,len(a.data))):
            blevel=level-alevel
            if blevel >= len(b.data) or b.data[blevel] is None or a.data[alevel] is None:
                continue
            if blevel ==0:
                continue
            source=(0,)*alevel + (1,)*(blevel-1)
            for l1,l2 in a.data[alevel].items():
                for r1,r2 in b.data[blevel].items():
                    prod=l2*r2
                    out_=np.zeros(level,dtype="int32")
                    if level==1:#so r1 is a single letter
                        _increment_value_in_dict_to_coeff(out[1],r1,prod)
                    else:
                        for mask in multiset_permutations(source):
                            mask=np.array(mask+[1,])
                            np.place(out_,1-mask,l1.letters)
                            np.place(out_,mask,r1.letters)
                            w = Word(out_)
                            _increment_value_in_dict_to_coeff(out[level],w,prod)
    return Elt(out)

def _allValuesFromElt(a):
    assert isinstance(a,Elt), a
    return tuple(itertools.chain.from_iterable(j.items() for j in a.data))

def tensorProduct(*args):
    """construct an EltElt as a sequence of Elts and EltElts tensored together.
    This is the tensor product of Elts (and EltElts), returning an EltElt.
    It is the _external_ tensor product in tensor space"""
    assert 0<len(args)
    for a in args:
        assert isinstance(a,(Elt,EltElt)), a
    n_out = sum(1 if isinstance(a,Elt) else a.n for a in args)
    out=dict()
    vals=[[((i,),j) for i,j in _allValuesFromElt(a)] if isinstance(a,Elt) else a.data.items() for a in args]
    for p in itertools.product(*vals):
        k = functools.reduce(operator.concat,(i for i,j in p))
        v = functools.reduce(operator.mul,(j for i,j in p))
        _increment_value_in_dict_to_coeff(out,k,v)
    return EltElt(out,n_out)

def tensorProductFunctions(*args, **kwargs):
    """if f,g,h takes Elts and returns Elts or EltElts then
    tensorProductFunctions(f,g,h) is the function f\otimes g\otimes h.
    If some of f,g or h return EltElt with n>1, then provide a named argument n as the tensor exponent
    we use to return in the case of zero input.
    If you are using python 3 you should think of this function's signature as
          "def tensorProductFunctions(*args, n=None):"
    """
    assert 0<len(args)
    if "n" in kwargs:
        n=kwargs["n"]
    else:
        n=len(args)
    def loc_tensorProductFunctions(a):
        assert isinstance(a,EltElt),a
        assert len(args)==a.n
        out=None
        for k,v in a.data.items():
            val=tensorProduct(*[i(word2Elt(j)) for i,j in zip(args,k)])*v
            if out is None:
                out = val
            else:
                out = out + val
        if out is None:
            return EltElt(dict(),n)
        return out
    return loc_tensorProductFunctions

def concatenationProductEltElt(a,b):
    assert isinstance(a,EltElt) and isinstance(b,EltElt), (a,b)
    assert a.n == b.n
    out=dict()
    for k1,v1 in a.data.items():
        for k2,v2 in b.data.items():
            k=tuple(concatenate(i,j) for i,j in zip(k1,k2))
            _increment_value_in_dict_to_coeff(out,k,v1*v2)
    return EltElt(out,a.n)

def dot_EltElt(a,b):
    assert isinstance(a,EltElt) and isinstance(b,EltElt), (a,b)
    assert a.n == b.n
    out=zero_coefficient()
    for k,v1 in a.data.items():
        if k in b.data:
            out += v1 * b.data[k]
    return out

def distance_EltElt(a,b):
    """The distance between the EltElts a and b in the word basis"""
    assert isinstance(a,EltElt) and isinstance(b,EltElt), (a,b)
    assert a.n==b.n
    d=a-b
    return math.sqrt(dot_EltElt(d,d))


def log1p(a,maxLevel=None,useRational=None):
    """returns the tensor logarithm of (1+a) where a is an Elt with nothing in level 0.
    if maxLevel is not given, only go up to the maximum level already in a
     - there is no other way to pick a maximum level
    This follows the pattern of iisignature's logTensorHorner
    log(1+x) = x(1-x(1/2-x(1/3-x(1/4-...))))
             = x-x(x/2-x(x/3-x(x/4-...)))
    When inside p brackets, we only need the first m-p levels to be calculated,
    because when multiplying a tensor t by x (which has 0 in the zeroth level)
    level k of t only affects level k+1 and above of xt.
    """
    assert isinstance(a,Elt), a
    assert iszero_coefficient(get_coefficient(a,emptyWord)), a
    maxLevel = _getMaxLevel(maxLevel)
    if maxLevel is None:
        maxLevel = len(a.data)-1
    assert type(maxLevel) is int, maxLevel
    s=t=zeroElt
    for depth in range(maxLevel,0,-1):
        constant = _reciprocateInteger(depth, useRational)
        t=concatenationProduct(a,s,1+maxLevel-depth)
        if depth>1:
            s=a*constant-t
    return a-t

def log(a,maxLevel=None,useRational=None):
    """tensor logarithm of a where a is an Elt with 1 in level 0"""
    assert isinstance(a,Elt), a
    assert isunit_coefficient(get_coefficient(a,emptyWord)), a
    d=a.data[:]#Shallow copy, but ok, we won't return it
    d[0]={emptyWord:zero_coefficient()}
    return log1p(Elt(d), maxLevel,useRational)

#exp(x)=1+x(1+x/2(1+x/3(...
#=1+x+x/2(x+x/3(x+...))
def exp(a,maxLevel=None,useRational=None):
    """tensor exponential of the Elt a.
    You almost always need to specify a maxLevel here"""
    assert isinstance(a,Elt), a
    assert iszero_coefficient(get_coefficient(a,emptyWord)), a
    maxLevel = _getMaxLevel(maxLevel)
    if maxLevel is None:
        maxLevel = len(a.data)-1
    assert type(maxLevel) is int, maxLevel
    s=zeroElt
    for depth in range(maxLevel,0,-1):
        constant = _reciprocateInteger(1+depth, useRational)
        t=concatenationProduct(a*constant,s,1+maxLevel-depth)
        s=a+t
    d=[None] if s is zeroElt else s.data
    d[0]={emptyWord:unit_coefficient()}
    return Elt(d)

#This function was previously called 'id', which clashed with a python builtin
def id_Elt(a):
    """The identity on Elts, id"""
    assert isinstance(a,Elt), a
    return a

def I(a):
    """returns a with constant term removed"""
    assert isinstance(a,Elt), a
    out = a.data[:]
    if len(out)>0:
        out[0]=dict()
    return Elt(out)

def alpha(a):
    """The antipode.
    E.g. if X is the truncated signature of a path, then alpha(X) is the one for the reversed path."""
    assert isinstance(a,Elt), a
    out=[{Word(k.letters[::-1]):((-1)**level)*v
          for k,v in x.items()}
           for level, x in enumerate(a.data)]
    return Elt(out)

def reverseAllWords(a):
    """returns a version of the Elt a with all words reversed"""
    assert isinstance(a,Elt), a
    out=[{Word(k.letters[::-1]):v
          for k,v in x.items()}
           for level, x in enumerate(a.data)]
    return Elt(out)

def lieProduct(a,b, maxLevel=None):
    """The Lie product of Elts a and b"""
    assert isinstance(a,Elt) and isinstance(b,Elt), (a,b)
    return concatenationProduct(a,b,maxLevel)-concatenationProduct(b,a,maxLevel)

def deltaOfLetter(letter,p):
    w=Word((letter,))
    o=unit_coefficient()
    tuples=[(emptyWord,)*i+(w,)+(emptyWord,)*(p-1-i) for i in range(p)]
    return EltElt({i:o for i in tuples},p)

def delta(a,p=2):#sh*, adjoint of sh
    """delta(x) is $\delta(x)$. delta(x,p) is $\delta_p(x)$"""
    assert isinstance(a,Elt), a
    assert isinstance(p,int), p
    out=dict()
    c = get_coefficient(a,emptyWord)
    if not iszero_coefficient(c):
        out[(emptyWord,)*p]=c
    for i in range(1,len(a.data)):
        for k,v in a.data[i].items():
            x=[deltaOfLetter(j,p) for j in k.letters]
            prod=functools.reduce(concatenationProductEltElt,x)
            for k2,v2 in prod.data.items():
                _increment_value_in_dict_to_coeff(out,k2,v2*v)
    return EltElt(out,p)

def deltabar(a):
    assert isinstance(a,Elt), a
    d=delta(a)
    out={(i,Word(j.letters[::-1])):v*((-1)**len(j.letters)) for (i,j),v in d.data.items()}
    return EltElt(out,2)

def deltadash(a,p=2): #aka conc*, p27, deconcatenation coproduct
    """deltadash(x) is $\delta'(x)$. delta(x,p) is $\delta'_p(x)$"""
    assert isinstance(a,Elt), a
    assert isinstance(p,int), p
    
    #copied from sympy kbins but allow empty bins
    def partition(lista, bins):
        #  EnricoGiampieri's partition generator from
        #  http://stackoverflow.com/questions/13131491/
        #  partition-n-items-into-k-bins-in-python-lazily
        if bins == 1:
            yield [lista]
        elif bins > 1:
            for i in range(0, len(lista)+1):
                for part in partition(lista[i:], bins - 1):
                    if len([lista[:i]] + part) == bins:
                        yield [lista[:i]] + part
    out=dict()
    for x in a.data:
        for k,v in x.items():
            for i in partition(k.letters,p):
                k2=tuple(Word(j) for j in i)
                _increment_value_in_dict_to_coeff(out,k2,v)
    return EltElt(out,p)
    
def ad(a):
    assert isinstance(a,Elt), a
    return lambda b: lieProduct(a,b)

def r(a):
    """Right Lie-bracketing function, extended linearly to Elts.
    e.g. 123 -> [1,[2,3]]
    This is also known as the Dynkin map"""
    assert isinstance(a,Elt), a
    out = [dict() for i in a.data]
    for i,x in enumerate(a.data):
        if i>0:
            for k,v in x.items():
                rr = [letter2Elt(j) for j in reversed(k.letters)]
                f = functools.reduce(lambda y,z:lieProduct(z,y),rr)
                for k2,v2 in f.data[i].items():#We only need to look in level i
                    _increment_value_in_dict_to_coeff(out[i],k2,v2*v)
    return Elt(out)

def l(a):
    """Left Lie-bracketing function, extended linearly to Elts.
    e.g. 123 -> [[1,2],3]
    page 36."""
    assert isinstance(a,Elt), a
    out = [dict() for i in a.data]
    for i,x in enumerate(a.data):
        if i>0:
            for k,v in x.items():
                rr = [letter2Elt(j) for j in k.letters]
                f = functools.reduce(lieProduct,rr)
                for k2,v2 in f.data[i].items():#We only need to look in level i
                    _increment_value_in_dict_to_coeff(out[i],k2,v2*v)
    return Elt(out)

def Ad(a):
    assert isinstance(a,Elt), a
    def loc_Ad(b):
        assert isinstance(b,Elt), b
        out=zeroElt
        for x in a.data:
            for k,v in x.items():
                y=b
                for j in reversed(k.letters):
                   y=lieProduct(letter2Elt(j),y)
                out = out + v*y
        return out
    return loc_Ad

#test (?) Ad(exp(x))(y)=exp(ad(x)(y)) ?for Lie elts x and y
#Ad is the derivative of conjugation?

def D(a):
    assert isinstance(a,Elt), a
    out=[{k:level*v
          for k,v in x.items()}
           for level, x in enumerate(a.data)]
    out[0]=dict()
    return Elt(out)

def D_inv(a):
    """The inverse of (D restricted to elements which are 0 in level 0).
    possibly not in Reutenauer."""
    assert isinstance(a,Elt), a
    assert iszero_coefficient(get_coefficient(a,emptyWord)), a
    out=[{k:v*(0 if level==0 else _reciprocateInteger(level))
          for k,v in x.items()}
           for level, x in enumerate(a.data)]
    return Elt(out)

def conc(a):
    """This is both conc and conc_p, as we don't assert a.n==2"""
    assert isinstance(a,EltElt), a
    out=[dict() for i in range(a.get_deg()+1)]
    for k,v in a.data.items():
        w=functools.reduce(concatenate,k)
        _increment_value_in_dict_to_coeff(out[len(w.letters)],w,v)
    return Elt(out)

def sh(a):
    assert isinstance(a,EltElt), a
    assert a.n==2, a
    out=[dict() for i in range(a.get_deg()+1)]
    for (x,y),v in a.data.items():
        source=(0,)*len(x.letters)+(1,)*len(y.letters)
        level=len(source)
        out_=np.zeros(level,dtype="int32")
        if(level==0):
            _increment_value_in_dict_to_coeff(out[level],emptyWord,v)
        else:
            for mask in multiset_permutations(source):
                mask=np.array(mask)
                np.place(out_,1-mask,x.letters)
                np.place(out_,mask,y.letters)
                w = Word(out_)
                _increment_value_in_dict_to_coeff(out[level],w,v)
    return Elt(out)

def sh_p(a):
    """This is the generic sh_p, we don't need p as an input. sh_p(a)==sh(a) if a.n==2"""
    assert isinstance(a,EltElt), a
    out=zeroElt
    for x,v in a.data.items():
        y=functools.reduce(shuffleProduct,[word2Elt(i) for i in x])
        out = out + y*v
    return out

def star(f,g):
    """convolution"""
    t=tensorProductFunctions(f,g)
    def loc_star(a):
        assert isinstance(a,Elt), a
        return conc(t(delta(a)))
    return loc_star

def stardash(f,g):
    t=tensorProductFunctions(f,g)
    def loc_stardash(a):
        assert isinstance(a,Elt), a
        return sh(t(deltadash(a)))
    return loc_stardash

def lambda_(a):
    assert isinstance(a,EltElt), a
    assert a.n==2,a
    out=[dict() for i in range(a.get_deg()+1)]
    for k,v in a.data.items():
        w=Word(k[0].letters[::-1]+k[1].letters)
        _increment_value_in_dict_to_coeff(out[len(w.letters)],w,v*len(k[0].letters))
    return Elt(out)

def mu(a):
    assert isinstance(a,EltElt), a
    assert a.n==2,a
    def loc_mu(b):
        assert isinstance(b,Elt), b
        out=zeroElt
        for (x,y),v in a.data.items():
            out = out + concatenationProductMany([word2Elt(x),b,word2Elt(y)])*v
        return out
    return loc_mu

def rhoOfWord(letters): #p32
    l=len(letters)
    if l==0:
        return zeroElt
    if l==1:
        return letter2Elt(letters[0])
    return (concatenationProduct(letter2Elt(letters[ 0]),rhoOfWord(letters[1:]))-
                          concatenationProduct(letter2Elt(letters[-1]),rhoOfWord(letters[:-1])))

def rho(a):
    assert isinstance(a,Elt), a
    out=[dict() for i in a.data]
    for i,x in enumerate(a.data):
        for k,v in x.items():
            f = rhoOfWord(k.letters)
            for k2,v2 in f.data[i].items():#We only need to look in one level of f
                _increment_value_in_dict_to_coeff(out[i],k2,v2*v)
    return Elt(out)

#note: This definition of pi1 is following the remark just before eqn 3.2.4.
#Although this looks inefficient, the direct calculation  of pi1 using equation 3.2.3
#looks bad too, as we need to decompose words into all the shuffles (i.e. partitions) that make them.
#pi1adjoint I do do directly though, because the decomposition into concatenations feels easier
def pi1(a):
    """the unique linear map on Elts s.t. log(x)=pi1(x) for any grouplike x"""
    assert isinstance(a,Elt), a
    #p58: pi_1=log(id_Elt)=log(epsilon+I)=log1p(I) [because epsilon is the unit in the algebra of End(K<A>)]
    maxlevel=len(a.data)-1
    out=zeroElt
    fn=None
    for i in range(1,maxlevel+1):
        fn=I if i==1 else star(I,fn)
        out+=fn(a)*(_reciprocateInteger(i)*(-1)**(i-1))
    return out

def pi1adjointOfWord(word):
    assert isinstance(word,Word),word
    l=len(word.letters)
    if l==0:
        return zeroElt
    if l==1:
        return word2Elt(word)
    out=zeroElt
    lets = [letter2Elt(i) for i in word.letters]
    for k in range(1,l+1):
        constant = (-1)**(k-1)*_reciprocateInteger(k)
        for u in kbins(lets,k):
            v=[concatenationProductMany(i) for i in u]
            out += constant * shuffleProductMany(v)
    return out


#Previously called pi1star
#useful for the following reason.
#if L is a Lie element and G is grouplike, then
#(L,log(G)) = (L,pi1(G)) = (pi1adjoint(L),G)
#and L can of course be written in the pbw basis as a linear combination of P(w) for hall words w
#which is how the log signature is written.
#So the linear function on signatures which returns element w of the log signature is
#f(X) = dotprod(pistar(S(w)),X)
def pi1adjoint(a):
    """adjoint of pi1
    This is what is known as $\pi_1^*$ in Section 6.2 (p129) of Reutenauer.
    It is what Eric Gehrig and Matthias Kawski call $\pi_1'$ in their
    'A Hopf-Algebraic Formula for Compositions of Noncommuting Flows' """
    assert isinstance(a,Elt), a
    out=[dict() for i in a.data]
    for i,x in enumerate(a.data):
        for k,v in x.items():
            f = pi1adjointOfWord(k)
            for k2,v2 in f.data[i].items():#We only need to look in one level of f
                _increment_value_in_dict_to_coeff(out[i],k2,v2*v)
    return Elt(out)    

def pi(a,n):
    """\pi_n(a)
    NB: This is not the function for projecting to the nth level, or up to the nth level, 
    which some authors call \pi_n. For that, use the restrictedToLevel or truncatedToLevel 
    member functions of Elt."""
    #This implementation is simple but slow
    #equation 3.2.6
    assert isinstance(a,Elt), a
    if(n==0):
        return epsilon(a)
    b=delta(a,n)
    c=tensorProductFunctions(*([pi1]*n))(b)
    d=conc(c)
    return d*_reciprocateInteger(math.factorial(n))

###BEGIN HALL BASIS STUFF

def foliage_iter(x):
    if type(x) is int:
        yield x
        return
    assert type(x) is tuple, x
    for i in x:
        for j in foliage_iter(i):
            yield j

def foliageFromTree(tup):
    return "".join(str(i) for i in foliage_iter(tup))


def printTreeAsLieBrackets(tup):
    if len(tup)==1:
        return str(tup[0])
    return "["+printTreeAsLieBrackets(tup[0])+","+printTreeAsLieBrackets(tup[1])+"]"


def lessExpressionLyndon(a,b):
    return tuple(foliage_iter(a))<tuple(foliage_iter(b))

#this is the other way around from coropa
def lessExpressionStandardHall(a,b):
    ll=len(tuple(foliage_iter(a)))
    lr=len(tuple(foliage_iter(b)))
    if ll!=lr:
        return lr<ll
    if 1==ll:
        return a<b
    if a[0]==b[0]:
         return lessExpressionStandardHall(a[1],b[1])
    return lessExpressionStandardHall(a[0],b[0])

def basisElementToElt(b):
    assert type(b) in (int, tuple)
    if type(b) is int:
        return letter2Elt(b)
    if len(b)==1:
        return letter2Elt(b[0])
    return lieProduct(basisElementToElt(b[0]),basisElementToElt(b[1]))

#A basis knows about its elements as tuples.
class HallBasis:
    def __init__(self, d, m, lessExpression=lessExpressionLyndon):
        assert d>1
        assert m>0
        self.d=d
        self.m=m
        out=[[(i,) for i in range(1,d+1)]]
        for mm in range(2,m+1):
            out.append([])
            for firstLev in range(1,mm):
                for x in out[firstLev-1]:
                    for y in out[mm-firstLev-1]:
                        if lessExpression(x,y) and (firstLev==1 or not lessExpression(x[1],y)):
                            out[-1].append((x,y))
        self.data=out
        self.less=lessExpression

    #w must be str or tuple of ints
    def findAsFoliageOfHallWord(self, w):
        assert type(w) in (tuple,str), w
        assert 0<len(w)<=self.m
        if type(w)==str:
            w=tuple(int(i) for i in w)
        for i in self.data[len(w)-1]:
            if w == tuple(foliage_iter(i)):
                return i
        return None
    #w must be str or tuple of ints
    def factorIntoHallWords(self,w):
        assert type(w) in (tuple,str), w
        assert 0<len(w)<=self.m
        if type(w)==str:
            w=tuple(int(i) for i in w)
        l=len(w)
        if l==1:
            assert 1<=w[0]<=self.d, str(w[0])+" is not in my alphabet"
            return [w]
        best=(w[-1],)
        best_prefix_length=l-1
        for prefix_length in range(0,l-1):
            end=w[prefix_length:]
            endH=self.findAsFoliageOfHallWord(end)
            if endH is not None and self.less(endH,best):
                best=endH
                best_prefix_length=prefix_length
        if best_prefix_length==0:
            return [best]
        return self.factorIntoHallWords(w[:best_prefix_length])+[best]

    def foliageLookup(self,m):
        """return dict from foliage to hall word index in level m"""
        assert 0<m<=self.m
        out=dict()
        for i,tree in enumerate(self.data[m-1]):
            out[tuple(foliage_iter(tree))]=i
        return out
    
    def indicesOfHallWords(self, m):
        """returns an array of length d**m where elements which are HallWords correspond to their index
        and all other indices are -1"""
        assert 0<m<=self.m
        if m==1:
            return np.arange(self.d)
        d=self.foliageLookup(m)
        out=[d.get(wd,-1) for wd in wordIter(self.d,m,topOnly=True,asNumbers=True)]
        return np.array(out)

def arbitraryLieEltSympy(basis, m=None, symbol='x'):
    """return an arbitrary Lie element with Sympy coefficients"""
    assert isinstance(basis, HallBasis), basis
    if m==None:
        m=basis.m
    assert m<=basis.m
    assert 1<basis.d<10
    out = zeroElt
    for levelMinus1 in range(m):
        for tree in basis.data[levelMinus1]:
            name = foliageFromTree(tree)
            coeff = sympy.var(symbol+"_"+name)
            elt = basisElementToElt(tree)
            out += elt*coeff
    return out

def arbitraryGrouplikeEltSympy(basis, m=None, symbol='x'):
    """return an arbitrary grouplike element with Sympy coefficients"""
    l=arbitraryLieEltSympy(basis,m,symbol)
    return exp(l)

def P(w, basis):
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (tuple,str), w
    if 0==len(w):
        return unitElt
    assert 0<len(w)<=basis.m
    a=basis.factorIntoHallWords(w)
    out = functools.reduce(concatenationProduct,(basisElementToElt(i) for i in a))
    return out

def S(w, basis):
    """Dual PBW basis element, p108"""
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (tuple,str), w
    assert len(w)<=basis.m
    if type(w)==str:
        w=tuple(int(i) for i in w)
    else:
        for i in w:
            assert isinstance(i,(int, str)), "perhaps you supplied a tree not a word?"
    if len(w)==0:
        return unitElt
    a=basis.factorIntoHallWords(w)
    if len(a)==1:
        return concatenationProduct(letter2Elt(w[0]),S(w[1:],basis))
    factor=1.0
    out = unitElt
    for i,j in itertools.groupby(a):
        word=tuple(foliage_iter(i))
        num=len(tuple(j))
        factor *= math.factorial(num)
        base = S(word,basis)
        power = functools.reduce(shuffleProduct,(base for i in range(num)))
        out = shuffleProduct(out,power)
    out = out*_reciprocateInteger(factor)
    return out

def wordToShuffleOfLogSigElts(w,basis):
    """If w's factorisation into hall words is h_1..h_n, return shuffle product of {pi1adjoint(S(h_i,basis))}.
    This is possibly our idea."""
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (tuple,str), w
    if 0==len(w):
        return unitElt
    assert 0<len(w)<=basis.m
    a=basis.factorIntoHallWords(w)
    #out = functools.reduce(shuffleProduct,(basisElementToElt(i) for i in a))
    out = functools.reduce(shuffleProduct,(pi1adjoint(S(foliageFromTree(i),basis)) for i in a))
    return out

def Q(w, basis, ignoreFactor=False):#p128
    """Word w -> shuffle of its hall factors, with constant coefficients of repeats divided out"""
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (tuple,str), w
    assert len(w)<=basis.m
    assert basis.less == lessExpressionLyndon
    if type(w)==str:
        w=tuple(int(i) for i in w)
    if len(w)==0:
        return unitElt
    a=basis.factorIntoHallWords(w)
    factor=1
    out = unitElt
    for i,j in itertools.groupby(a):
        word=tuple(foliage_iter(i))
        num=len(tuple(j))
        factor *= math.factorial(num)
        base = word2Elt(word)
        power = functools.reduce(shuffleProduct,(base for i in range(num)))
        out = shuffleProduct(out,power)
    if not ignoreFactor:
        out = out*_reciprocateInteger(factor)
    return out

class TensorSpaceBasis:
    """A basis of tensor space up to level m on d letters, given by a function fn which maps words of
    length mm to each basis element of level mm. Trivial example is fn=word2Elt, but other cases work too,
    where we supply a hall basis too to fn. 
    Interesting cases are fn=P (the Poincare-Birkhoff-Witt basis) and fn=S (its dual)."""
    def __init__(self, fn, basis=None, d=None, m=None, checkFn=True):
        #would be nice to remove this assertion
        #but it's easy to get the arguments of this ctor in the wrong order
        if checkFn:
            assert fn in (word2Elt, P, S, Q), fn
        if basis is not None:
            assert isinstance(basis, HallBasis), basis
            if d is None:
                d=basis.d
            if m is None:
                m=basis.m
        assert d is not None
        assert m is not None
                
        if fn==word2Elt:
            self.fn=word2Elt
        else:
            assert isinstance(basis, HallBasis), basis
            assert 0<m<=basis.m
            assert 0<d<=basis.d
            self.fn= lambda x:fn(x,basis)
        self.d=d
        self.m=m
        if fn != word2Elt:
            from itertools import product
            alphabet=list(range(1,d+1))
            o=[]
            for r in range(m+1):
                words = list(product(alphabet,repeat=r))
                locdata=[]
                for w in words:
                    val=self.fn(w)
                    a=[val.data[r].get(Word(w2),0) for w2 in words]
                    locdata.append(a)
                o.append(np.array(locdata,dtype="float64"))
            self.data=o
        #print(self.data)

    #Simple constructors
    @staticmethod
    def wordBasis(d,m):
        return TensorSpaceBasis(word2Elt,d=d,m=m)
    @staticmethod
    def fromFunctionAndHallBasis(fn, hallBasis, m=None):
        """get a basis given a function on words which like P and S returns a basis element for each word."""
        return TensorSpaceBasis(fn,hallBasis,m=m,checkFn=False)

    def generateWordsElts(self):
        from itertools import product
        alphabet=list(range(1,self.d+1))
        for r in range(self.m+1):
            words = list(product(alphabet,repeat=r))
            for w in words:
                val=self.fn(w)
                yield "".join(str(i) for i in w),val
        
    def size(self):
#        return sum(i.shape[0] for i in self.data)
        return sum(self.d**r for r in range(1+self.m))
    
    def indexOfLevelStart(self, level):
        return sum(self.d**r for r in range(level))
        
    def toElt(self,x,m=None):
        """return the Elt corresponding to a list of coefficients of our elements"""
        """if m is specified, x only contains level m, instead of all levels up to self.m"""
        if m is None:
            assert len(x)==self.size()
        else:
            assert m<=self.m
            assert len(x)==self.d**m
        out=zeroElt
        it = iter(x)
        alphabet=list(range(1,self.d+1))
        if self.fn != word2Elt:
            R = enumerate(self.data) if m is None else [(m,self.data[m])]
            for r, mat in R:
                y=np.zeros_like(mat[:,0])
                for row in mat:
                    y=y+next(it)*row
                for i,j in zip(y,itertools.product(alphabet,repeat=r)):
                    out +=i*word2Elt(j)
        else:
            if m is None:
                o=[{Word(a):next(it) for a in itertools.product(alphabet,repeat=r)} for r in range(self.m+1)]
            else:
                o=[dict() for r in range(m)]
                o.append({Word(a):next(it) for a in itertools.product(alphabet,repeat=m)})
            out=Elt(o)
        return out
    def fromElt(self,a, checked=True, m=None, untyped=False):
        """Express a given Elt in our basis
           If m is given, we only return level m in terms of our basis of level m.
           Unless untyped is set, we assume the coefficients are convertible to float.
           When untyped is True, you need to check numpy will still cope, and you get a list.
           Unless checked is False, we check that the Elt can fit in the basis."""
        assert isinstance(a,Elt), a
        if checked:
            if m is None:
                assert len(a.data)<=self.m+1
            assert a.maxLetter()<=self.d
        if m is not None:
            assert m<=self.m
        #Something about asserting length
        alphabet=list(range(1,self.d+1))
        outIdx=0
        output_size=self.size() if m is None else self.d**m
        if untyped:
            #out=np.empty(output_size, dtype=object) #XXX
            out=[None]*output_size
        else:
            out=np.zeros(output_size)
        if self.fn!=word2Elt:
            R = zip(range(self.m+1),self.data,a.data) if m is None else [(m,self.data[m],a.data[m])] if m<len(a.data) else []
            for r,mat,d in R:
                x=[float(d.get(Word(i),0)) for i in itertools.product(alphabet,repeat=r)]
                v=scipy.linalg.lstsq(mat.T,x)
                assert v[2]==mat.shape[1]
                newOutIdx=outIdx+mat.shape[0]
                out[outIdx:newOutIdx]=v[0]
                outIdx=newOutIdx
        else:
            R=zip(range(self.m+1),a.data) if m is None else [(m,a.data[m])] if m<len(a.data) else []
            for r,dd in R:
                for a,b in dd.items():
                    locidx=0
                    for l in a.letters:
                        #if checked:
                        #    assert 0<l<=self.d
                        locidx*=self.d
                        locidx+=l-1
                    out[outIdx+locidx]=b
                outIdx+=self.d**r
        return out
    def inTermsOf(self,x,l,allowFailure=False):
        """Express a given Elt (or list of Elts) x as (a) linear combination(s) of the elts in l, 
           using our basis"""
        single = isinstance(x,Elt)
        if not single:
            for i in x:
                assert isinstance(i, Elt), i
        for b in l:
            assert isinstance(b,Elt), b
        sources = np.array([self.fromElt(b) for b in l])
        targets = self.fromElt(x) if single else np.transpose([self.fromElt(i) for i in x])
        v=scipy.linalg.lstsq(sources.T,targets)
        if not allowFailure:
            assert v[2]==len(l)#l is not LI
            assert np.amax(v[1])<1e-8#not in span
        return v[0] if single else [v[0][:,i] for i in range(v[0].shape[1])]
    def matrix(self,l):
        """return the matrix of the elts in l, 
           using our basis"""
        for b in l:
            assert isinstance(b,Elt), b
        sources = np.array([self.fromElt(b) for b in l])
        return sources
    def rank(self,l):
        """return the dimension of the span of the elts in l, 
           using our basis"""
        return np.linalg.matrix_rank(self.matrix(l))

    def indicesOfAnagramSet(self, counts, singleLevelOnly):
        """return the indices of elements which have the homogeneity given by counts"""
        #This currently doesn't depend on self.fn because
        #self.fn is always finely homogenous
        while len(counts)<self.d:
            counts=list(counts)+[0]
        assert len(counts)==self.d
        level = sum(counts)
        assert level <= self.m
        o=[]
        from itertools import product, chain
        alphabet=list(range(1,self.d+1))
        target=list(chain.from_iterable([(i,)*j for i,j in zip(alphabet,counts)]))
        for i,j in enumerate(itertools.product(alphabet,repeat=level)):
            if target == sorted(j):
                o.append(i)
        if not singleLevelOnly:
            offset = indexOfLevelStart(level)
            return [offset + i for i in o]
        return o                

#This function illustrates expressing an Elt which is known to be a Lie element
#in terms of a HallBasis. 
def bch_coefficients(bas):
    """Return the coefficients of the Baker-Campbell-Hausdorff (BCH)
    aka Campbell-Baker-Hausdorff (CBH) formula in the given Hall Basis."""
    assert isinstance(bas, HallBasis)
    assert bas.d==2
    x1=exp(letter2Elt(1),maxLevel=bas.m)
    x2=exp(letter2Elt(2),maxLevel=bas.m)
    x=log(concatenationProduct(x1,x2,maxLevel=bas.m))
    out = [ [dotprod(S(foliageFromTree(i),bas),x)
              for  i in lev]
               for lev in bas.data]
    return out

###END HALL BASIS STUFF

###BEGIN UTILITIES
def expandSetsThroughGrading(gradedList,level):
    """if gradedList is a list of lists, where gradedList[i] is things of weight i+1
    returns a list of tuples(representing sets) of things of total weight level"""
    sources=[[(i,j) for j in range(len(gradedList[i]))] for i in range(len(gradedList))]
    o=set()
    for p in ordered_partitions(level):
        if all(i<=len(gradedList) for i in p):
            for t in itertools.product(*(sources[i-1] for i in p)):
                o.add(tuple(sorted(t)))
    return [tuple(gradedList[i][j] for i,j in k) for k in o]

def expandThroughGrading(gradedList,level,fn):
    """if gradedList is a list of lists, where gradedList[i] is things of weight i+1 (*)
    returns [fn(t) for t a tuple (representing a set) of things of total weight level]
    by "representing a set" I mean that fn(t) should give the same value if you permute the elements of t
    (*) mnemonic: this function wouldn't be able to do anything with things of weight 0, so it provides no way to supply them!"""
    return [fn(i) for i in expandSetsThroughGrading(gradedList,level)]

def signature_of_path_manual(path,m):
    """calculate the signature of a path"""
    path=np.array(path)
    d=path.shape[-1]
    if np.shape(path)[0]<2:
        return unitElt
    displacements = path[1:,:]-path[:-1,:]
    displacements2 = [functools.reduce(operator.add,(letter2Elt(i+1)*float(dis[i]) for i in range(d))) for dis in displacements]
    sigsOfSegments = [exp(i,m) for i in displacements2]
    sig = functools.reduce(lambda x,y: concatenationProduct(x,y,m),sigsOfSegments)
    return sig

def signature_of_path_iisignature(path,m):
    """calculate the signature of a path using iisignature"""
    import iisignature
    d=np.shape(path)[-1]
    s=iisignature.sig(path,m,1)
    letters=range(1,d+1)    
    out=[{Word(key):float(val) for key,val in zip(itertools.product(letters,repeat=lev),vals)}
              for lev,vals in zip(range(1,m+1),s)]
    return Elt([{emptyWord:1}]+out)

def countHallWords(d,m):
    """necklace polynomial: number of hall words at depth m on d letters"""
    return sum(mobius(m//D)*(d**D) for D in divisors(m))/m

def expressFunctionInBasis(f,bas,d=None,m=None, basisForImage=None):
    """if f is a function from Elts to Elts, give the matrix which is its
    effect on level m on d letters, in the given basis/ between the given bases
    Normally you want to specify m but not d"""
    if basisForImage is None:
        basisForImage = bas
    if d is None:
        d=bas.d
    if m is None:
        m=bas.m
    assert d<=bas.d
    assert d<=basisForImage.d
    assert m<=bas.m
    assert m<=basisForImage.m
    alphabet=list(range(1,d+1))
    from itertools import product
    out=[]
    for word in product(alphabet,repeat=m):
        elt = bas.fn(word)
        im = basisForImage.fromElt(f(elt),m=m)
        out.append(im)
    return np.array(out).T


class MaxLevelContext():
    """Several functions have a maxLevel parameter.
    (concatenationProduct, shuffleProduct, log, log1p, exp and their friends)
    If you want to fix a maxLevel of say 4 for a block of code do something like this

    with MaxLevelContext(4):
        Block..
    """

    def __init__(self, level):
        self.level = level
    def __enter__(self):
        global _defaultMaxLevel
        self.origMaxLevel = _defaultMaxLevel
        _defaultMaxLevel = self.level
    def __exit__(self,a,b,c):
        global _defaultMaxLevel
        _defaultMaxLevel = self.origMaxLevel


class UseRationalContext:
    """If you want this library to use Sympy's rational numbers instead of floating point
    during a block, you can do
    
    with UseRationalContext():
         Block..
    """
    def __init__(self, use=True):
        self.use = use
    def __enter__(self):
        global _defaultUseRational
        self.origUse=_defaultUseRational
        _defaultUseRational = self.use
    def __exit__(self,a,b,c):
        global _defaultUseRational
        _defaultUseRational = self.origUse

_expn=dict()
def parse(s, sympyCoeffs=False):
    """parse("[3]+2+[4]11") -> 3*unitElt + word2Elt("2") + 4*word2Elt("11")"""
    global _expn
    if sympyCoeffs not in _expn:
        import pyparsing as pp

        if sympyCoeffs:
            from sympy.parsing.sympy_parser import parse_expr
            coeff_s = pp.QuotedString("[",endQuoteChar="]")
            coeff_s.setParseAction(lambda t: [parse_expr(t[0])])
            coeff = pp.Optional(coeff_s,1)
        else:
            coeff_i=pp.Suppress("[")+pp.Word(pp.nums)+pp.Suppress("]")
            coeff_i.setParseAction(lambda t: [int(t[0])])
            coeff_f=pp.Suppress("[")+pp.Combine(pp.Optional(pp.Word(pp.nums))+
                                                "."+
                                                pp.Optional(pp.Word(pp.nums)))+pp.Suppress("]")
            coeff_f.setParseAction(lambda t: [float(t[0])])
            coeff=pp.Optional(coeff_i|coeff_f,1)
        if six.PY2:
            minus = pp.Literal("-")
        else:
            #In python 3, where str is unicode, it is easy to allow the minus sign character.
            #This means you can copy from a formula in a pdf
            minus = pp.Literal("-")|pp.Literal(chr(0x2212))
            minus.setParseAction(lambda t:["-"])
        firstTerm=pp.Optional(minus,"+")+coeff+pp.Optional(pp.Word(pp.nums),"")
        otherTerm=(pp.Literal("+")|minus)+coeff+pp.Optional(pp.Word(pp.nums),"")
        _expn[sympyCoeffs] = pp.Group(firstTerm)+pp.ZeroOrMore(pp.Group(otherTerm))
    #print(s)
    exp=_expn[sympyCoeffs].parseString(s,True)
    #print(exp)
    x=[(b if a=="+" else -b)*word2Elt(c) for a,b,c in exp]
    #print(x)
    out = functools.reduce(operator.add,x)
    return out  

def randomLieElt(d,m):
    """Just a way to make an Elt which is in the FLA, no particular distribution"""
    a = randomElt(d,m)
    out = r(a)
    return out

def randomGrouplikeElt(d,m):
    """Just a way to make an Elt whose log is in the FLA, no particular distribution"""
    return exp(randomLieElt(d,m))

###END UTILITIES


###BEGIN TESTING/EXAMPLES

#verify the adjoint properties, p27
def testHopfAlgebraProperty(P,Q,R):
    a=dotprod(sh(tensorProduct(P,Q)),R)
    a_=dotprod(shuffleProduct(P,Q),R)
    assert np.allclose(a,a_) #this is obvious
    b=dot_EltElt(tensorProduct(P,Q),delta(R))
    #print("testHopf1:",a,b)
    assert np.allclose(a,b)

    c=dotprod(concatenationProduct(P,Q),R)
    c_=dotprod(conc(tensorProduct(P,Q)),R)
    assert np.allclose(c,c_)
    d=dot_EltElt(tensorProduct(P,Q),deltadash(R))
    #print("testHopf2:",c,d)
    assert np.allclose(c,d)

#grouplike elts obviously obey the coalgebra axioms with delta and epsilon

def testPBWdual(words,basis):
    Ps = [P(w,basis) for w in words]
    Ss = [S(w,basis) for w in words]
    if 0:
        for i in range(len(words)):
            print(Ps[i])
            print(Ss[i])
            print(dotprod(Ps[i],Ss[i]))
    for i in range(len(words)):
        for j in range(len(words)):
            assert np.allclose(1 if i==j else 0, dotprod(Ps[i],Ss[j]))

def testSympy():
    from sympy.parsing.sympy_parser import parse_expr
    """A simple example of using symbolic coefficients"""
    x=parse("2+[x]12",sympyCoeffs=True)
    rx = r(x)+x+unitElt
    assert "[1]+[2]2+[2*x]12+[-x]21"==rx.prettySympy()

    g=arbitraryGrouplikeEltSympy(HallBasis(2,2))
    coeff=get_coefficient(g*2,Word("21"))
    assert(coeff==parse_expr("x_1*x_2 - 2*x_12"))

def testRational():
    assert exp(parse("1"),3).pretty(dp=3)=="[1]+1+[0.5]11+[0.167]111"
    with UseRationalContext():
        assert exp(parse("1"),3).pretty()=="[1]+1+[1/2]11+[1/6]111"
            
def test():
    a=Word("121")
    b=letter2Elt("1")
    c=Word("2")
    A=word2Elt(a)
    B=word2Elt("31")
    C=word2Elt("2")
    p=parse
    assert Word("123").crudeNumber()==123
    assert parse("[3]+2+[4]11") == 3*unitElt + word2Elt("2") + 4*word2Elt("11")
    onetwo=lieProduct(letter2Elt(1),letter2Elt(2))
    assert onetwo == parse("12-21")
    assert 1==get_coefficient(A,a)
    assert zeroElt==(B-B)
    assert concatenationProduct(word2Elt(a),b)==word2Elt("1211")
    assert shuffleProduct(A,b)==(word2Elt("1211")+word2Elt("1121"))*2
    ee=EltElt({(a,c):7},2)
    ee0=EltElt({},2)
    assert ee0==ee.truncatedToLengths((2,None))==ee.truncatedToTotalLength(3)
    assert ee.pretty()=="+7(121, 2)"
    assert conc(ee)==word2Elt("1212")*7
    assert conc(tensorProduct(B,C,B))==p("31231")
    rA=r(A)
    assert lambda_(deltabar(A))==rA
    assert 12==dotprod(rA*2,rA)==epsilon_numeric(make_dual(rA)(rA*2))

    x=unitElt+alpha(shuffleProduct(A,word2Elt("1"))+A)
    assert D(x).data==[{},{},{},{a:-3},{Word("1211"):8,Word("1121"):8}]
    assert D_inv(D(x))==I(x)

    assert 0 <= distance(b,log(exp(b,6),6)) < 1e-10
    assert 0 <= distance(A,log(exp(A,6),6)) < 1e-10
    onetwo=lieProduct(letter2Elt(1),letter2Elt(2))
    assert onetwo==rho(word2Elt("12"))

    testHopfAlgebraProperty(p("12"),onetwo,p("[7]1221+[3]1212"))
    #assert (sh(deltadash(A*4+unitElt)))
    assert ad(word2Elt("12"))(word2Elt("4")) == parse("124-412")
    assert Ad(word2Elt("12"))(word2Elt("4")) == parse("124-142-241+421")
    #oneonetwo=r(word2Elt("112"))
    testHopfAlgebraProperty(randomElt(2,3),randomElt(2,3),randomElt(2,6))

    H=HallBasis(2,12)
    assert basisElementToElt(HallBasis(2,3).data[2][1])==parse("122+221-[2]212")
    assert H.findAsFoliageOfHallWord("112")==((1,),((1,),(2,)))
    assert np.allclose(H.indicesOfHallWords(3),[-1,0,-1,1,-1,-1,-1,-1])
    #print(H.factorIntoHallWords("2121211222"))
    HH=HallBasis(2,12,lessExpressionStandardHall)
    #print(HH.factorIntoHallWords("2121211222"))
    assert P("121",H) == concatenationProduct(onetwo,word2Elt("1"))
    assert P("121",HH) == parse("[2]121-211-112")
    testPBWdual(["1122","1212"],H)
    testPBWdual(["11221","22222","11122","11212"],HH)
    assert S("1212",H) == parse("1212+[2]1122")==Q("1212",H)
    assert S("111",H) == parse("111")==Q("111",H)
    assert parse("11212")==Q("11212",H)!=S("11212",H)==parse("11212+[2]11122")
    H23=HallBasis(2,3)
    H25=HallBasis(2,5,lessExpressionStandardHall)
    for t in (TensorSpaceBasis(word2Elt,None,2,3),TensorSpaceBasis(P,H23),
              TensorSpaceBasis(Q,H23),TensorSpaceBasis(S,H23)):
        z=np.random.rand(15)
        assert np.allclose(z,t.fromElt(t.toElt(z)))
    
    bch=bch_coefficients(H25)
    assert len(bch)==5
    assert np.allclose(720*np.array(bch[4]),[-2,-6,1,4,-4,-1])

    #group stuff in the Lie group
    g1 = randomGrouplikeElt(2,5)
    g1inv = alpha(g1) #alpha is the inverse
    g2 = randomGrouplikeElt(2,2)
    assert distance(unitElt,concatenationProduct(g1,g1inv,5))<=1e-10
    #Ad is conjugation
    assert distance(Ad(g1)(g2).truncatedToLevel(5),
                        concatenationProductMany([g1,g2,g1inv],5))<=1e-10

    #The formula 
    # Ad(exp(X))(Y)=exp(ad(X))(Y) := Y + [X,Y]+0.5 [X,[X,Y]] + ...
    # is clear from our definition of Ad

    r1=randomElt(2,4)
    r2=randomElt(2,4)
    #check lemma 1.6(i)
    assert distance(ad(r1)(r2),mu(tensorProduct(r1,unitElt)-tensorProduct(unitElt,r1))(r2))<1e-10
    assert distance(Ad(r1)(r2),mu(deltabar(r1))(r2))<1e-10

    #formula in middle of p32
    rr1=r(r1)
    assert distance(rr1,conc(tensorProductFunctions(D,alpha)(delta(r1))))<1e-14
    assert distance(rr1,star(D,alpha)(r1))<1e-10

    assert distance(l(r1),conc(tensorProductFunctions(alpha,D)(delta(r1))))<1e-14

    def tensorWithSelf(x):
        return tensorProduct(x,x)
    f=tensorProductFunctions(tensorWithSelf,n=2)
    assert f(tensorProduct(unitElt))*0 == f(tensorProduct(zeroElt))

    #This is equation 1.6.5 in Reutenauer. It doesn't hold for arbitrary P and Q,
    #but only for P having nothing in level 0. I use I(r1) to get a version of r1 which then works.
    #AKA Baker's identity
    assert (distance(l(concatenationProduct(I(r1),l(r2))),lieProduct(l(r1),l(r2))))<1e-12
    #version of the same formula for r
    assert (distance(r(concatenationProduct(r(r1),I(r2))),lieProduct(r(r1),r(r2))))<1e-12

    #equation 1.6.8 on the kernel of l. i.e. It's a right ideal generated by {Pl(P) for P an Elt}
    #Note that the statement should say that P has nothing in level 0, any right ideal containing
    #the empty word would be everything. Hence the use of I here.
    r22=I(randomElt(2,2))
    assert distance(l(concatenationProductMany([r22,l(r22),randomElt(2,2)])),zeroElt)<1e-12
    #version of the same formula for r
    assert distance(r(concatenationProductMany([randomElt(2,2),r(r22),r22])),zeroElt)<1e-12

    #In each level, the image of rho is the orthogonal complement of the kernel of r
    assert dotprod(rho(r1),concatenationProduct(r(r22),r22))<1e-14
    r21=randomLieElt(2,1)
    assert dotprod(rho(r1),concatenationProductMany([r22,r21,r21]))<1e-14

    r3=p("[9]+1+12+123")
    assert reverseAllWords(r3)==p("[9]+1+21+321")
    assert distance(r3,functools.reduce(operator.add,(pi(r3,i) for i in range(4))))<1e-10
    r4=randomLieElt(2,4)
    assert distance(r4,pi(r4,1))<1e-10
    assert distance(r4,pi1(exp(r4)))<1e-10#pi1 is a linear fn which coincides with log on the gp

    assert rightHalfShuffleProduct(p("12"),p("23"))==p("[2]1223+2123")
    assert rightHalfShuffleProduct(p(""),p("2"))+rightHalfShuffleProduct(p("2"),p(""))==shuffleProduct(p(""),p("2"))

    bas=TensorSpaceBasis.wordBasis(3,3)
    parts=[p("1+123"),p("[3]+[3]12"),p("12")]
    assert parts[1].pretty()=="[3]+[3]12"
    assert np.allclose(bas.inTermsOf(r3,parts),[1,3,-8])
    assert bas.rank(parts)==3
    assert [1,3,9]==bas.indicesOfAnagramSet([2,1], True)
    bas2=TensorSpaceBasis.fromFunctionAndHallBasis(P,HallBasis(3,5),m=4)
    for i, j in enumerate([[0],[0,1,0],[0]*9]):
        assert np.allclose(bas.fromElt(C,m=i),j)
        assert np.allclose(bas2.fromElt(C,m=i),j)

    have_iisignature = False
    try:
        import iisignature
        have_iisignature=True
    except ImportError:
        pass
    if have_iisignature:
        path=[[0,0],[1,0.3],[1,1]]
        p1=signature_of_path_manual(path,3)
        p2=signature_of_path_iisignature(path,3)
        assert distance(p1,p2)<1e-7 #not too close as iisignature uses float32s
    
    assert pi1adjoint(S((1,2),H)) == 0.5*onetwo
    assert np.allclose(dotprod(pi1(r1),r2),dotprod(r1,pi1adjoint(r2)))

    vol=p("123+231+312-132-213-321")
    assert r(vol)==zeroElt #so vol isn't a Lie element

    #Calculate the signature of the unit square two ways up to level 4
    with MaxLevelContext(4):
        s1 = concatenationProductMany([exp(b),exp(C),exp(-b),exp(-C)])
    s2 = signature_of_path_manual([[0,0],[1,0],[1,1],[0,1],[0,0]],4)
    assert distance(s1,s2)<1e-10

    mat_r = expressFunctionInBasis(r, bas)
    mat_rho=expressFunctionInBasis(rho, bas)
    assert np.allclose(mat_r.T,mat_rho)

    testSympy()
    testRational()
    
    
class TestFLA(unittest.TestCase):
    def testall(self):
        test()
    
if __name__=="__main__":
    test()
    np.set_printoptions(suppress=True, precision=4)
    a = Word("121")
    b = Word("1")
    A=word2Elt(a)
    B=word2Elt("31")
    C=word2Elt("2")
    onetwo=lieProduct(letter2Elt(1),letter2Elt(2))
    p=parse

    def pp (x):
        print(x.pretty(dp=7))

