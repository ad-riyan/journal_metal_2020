"""
pyScrew4Mobility.py
    is a module for application of screw theory to determine the mobility of parallel manipulators
    (PMs) written entirely in python using symbolic python (sympy) package.

This module is developed by Adriyan <adriyan0686|at|gmail|dot|com>, 2019, using Python 3.7 and
Sympy 1.4. Python 3.6+ and Sympy 1.3 are able to run this module even-though it is not fully
tested. To the best developer knowledge the built-in functions that used in this package is not
altered since the previous version which already mentioned.


History:
--------
    2020.03.20 - New


This software is released under 3-clause BSD (Berkeley Software Distribution)
License.

Copyright (c) 2020, Adriyan
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sympy as sym
from sympy.physics import matrices as mat


#
# SCREW
#
class Screw:
    """
    Screw is a class to construct a screw based on
        * the parameters of the screw itself, namely: s as the the direction of a screw in a 3 by
          1 unit column vector; r as the position vector of the variable 's' with respect to a
          reference frame in a 3 by 1 unit column vector; and h as the pitch of the screw.
        * a 6 by 1 column vector.

    When the screw is constructed, it will have five instances of attribute, namely:
        * the screw itself in 6 rows by 1 column of a column vector,
        * the first vector of the screw from its first three row in 3 row by 1 column of a column
          vector,
        * the second vector of the screw from its last three row in 3 row by 1 column of a column
          vector,
        * the pitch of the screw in a scalar.
        * the kind of screw, which can be 'twist' or 'wrench' or None for the unassigned kind.

    Also, the screw can be operated algebraically using scalar multiplication, negation,
    addition, and reciprocal product.

    For example:
    ------------

    Creating a screw from the three parameters, i.e. s, r, and h.

        >>> import sympy as sym
        >>> import sympy.physics.matrices as mat
        >>> a, b, c, pitch = sym.symbols("a b c h")
        >>> s = mat.Matrix([1, 0, 0])  # The direction of a screw in a 3 by 1 unit column vector
        >>> r = mat.Matrix([a, b, c])  # The position vector of the variable 's' with a, b, and c \
                                       # as the component of the position of the vector from a \
                                       # reference frame, in a 3 by 1 unit column vector
        >>> h = 0  # The pitch of the screw, h = 0 for zero pitch screw
        >>> import pyScrew4Mobility as s4m
        >>> args = [s, r, h]
        >>> kind = "twist"  # the kind of screw, must be either 'twist' or 'wrench', default \
                            # is None for unassigned argument.
        >>> screw_1 = s4m.Screw(args, kind)
        >>> type(screw_1)
        screw_theory_for_mobility.Screw
        >>> screw_1.screw
        Matrix([
        [ 1],
        [ 0],
        [ 0],
        [ 0],
        [ c],
        [-b]])
        >>> screw_1.vector_1st
        Matrix([
        [1],
        [0],
        [0]])
        >>> screw_1.vector_2nd
        Matrix([
        [ 0],
        [ c],
        [-b]])
        >>> screw_1.pitch
        0
        >>> screw_1.kind
        'twist'
        >>>
        >>> h = sym.oo  # The pitch of the screw, h = sym.oo for infinity pitch screw
        >>> screw_2 = s4m.Screw([s, r, h], kind)
        >>> screw_2.screw
        Matrix([
        [0],
        [0],
        [0],
        [1],
        [0],
        [0]])
        >>> screw_2.vector_1st
        Matrix([
        [0],
        [0],
        [0]])
        >>> screw_2.vector_2nd
        Matrix([
        [1],
        [0],
        [0]])
        >>> screw_2.pitch
        oo
        >>> screw_2.kind
        'twist'
        >>>
        >>> h = pitch  # The pitch of the screw, h = pitch (h as the defined symbol for pitch) \
                       # for general pitch screw
        >>> screw_3 = s4m.Screw([s, r, h], kind)
        >>> screw_3.screw
        Matrix([
        [ 1],
        [ 0],
        [ 0],
        [ h],
        [ c],
        [-b]])
        >>> screw_3.vector_1st
        Matrix([
        [1],
        [0],
        [0]])
        >>> screw_3.vector_2nd
        Matrix([
        [ h],
        [ c],
        [-b]])
        >>> screw_3.pitch
        h
        >>> screw_3.kind
        'twist'
        >>>

    Creating a screw object if you only have a screw that already defined by 6 rows by 1 column,

        >>> S4 = mat.Matrix([0, 1, 0, a, b, c])
        >>> type(S4)
        sympy.matrices.dense.MutableDenseMatrix
        ## or it can be
        sympy.matrices.immutable.ImmutableDenseMatrix
        >>> screw_4 = s4m.Screw(S4, kind="wrench")
        >>> screw_4.screw
        Matrix([
        [0],
        [1],
        [0],
        [a],
        [b],
        [c]])
        >>> screw_4.vector_1st
        Matrix([
        [0],
        [1],
        [0]])
        >>> screw_4.vector_2nd
        Matrix([
        [a],
        [b],
        [c]])
        >>> screw_4.pitch
        b
        >>> screw_4.kind
        'wrench'

    Algebraic calculation for the screws.

    [Scalar Multiplication]. Each screw can be multiplied by a scalar whether it a number or a
    symbolic representation of the physical worlds
        >>> screw_1_m_scalar = 2.5 * screw_1  # It also can be performed as screw_1 * 2.5
        >>> screw_1_m_scalar.screw
        Matrix([
        [   2.5],
        [     0],
        [     0],
        [     0],
        [ 2.5*c],
        [-2.5*b]])
        >>> screw_1_m_scalar.pitch
        0
        >>> screw_1_m_scalar.kind
        'twist'
        >>> v, omega, f = sym.symbols("v \\omega f")
        >>> screw_1_m_symbol_omega = omega * screw_1
        >>> screw_1_m_symbol_omega.screw
        Matrix([
        [   \omega],
        [        0],
        [        0],
        [        0],
        [ c*\omega],
        [-b*\omega]])
        >>> screw_1_m_symbol_omega.pitch
        0
        >>> screw_1_m_symbol_omega.kind
        'twist'
        >>> screw_2_m_symbol_v = v * screw_2
        >>> screw_2_m_symbol_v.screw
        Matrix([
        [0],
        [0],
        [0],
        [v],
        [0],
        [0]])
        >>> screw_2_m_symbol_v.pitch
        oo
        >>> screw_2_m_symbol_v.kind
        'twist'
        >>> screw_4_m_symbol_f = f * screw_4
        >>> screw_4_m_symbol_f.screw
        Matrix([
        [  0],
        [  f],
        [  0],
        [a*f],
        [b*f],
        [c*f]])
        >>> screw_4_m_symbol_f.pitch
        b
        >>> screw_4_m_symbol_f.kind
        'wrench'

    [Addition]. When adding two screw with assigned kind, either 'twist' or 'wrench', the kind for
    both screws must be same, unless for the un-assigned ones, (kind=None) will still return the
    addition.
        >>> screw_add = screw_1_m_symbol_omega + screw_2_m_symbol_v
        >>> # the previous command is equivalent to \
            # screw_add = omega * screw_1 + v * screw_2
        >>> screw_add.screw
        Matrix([
        [   \omega],
        [        0],
        [        0],
        [        v],
        [ c*\omega],
        [-b*\omega]])
        >>> screw_add.pitch
        v/\omega

    [Reciprocal Product]. Two screws can be found its reciprocal product, but it must be a
    different in kind if each screw has the assigned kind. For the un-assigned ones (kind=None),
    it still returns the product.
        >>> rss = screw_2_m_symbol_v @ screw_4_m_symbol_f
        \omega*a*f + \omega*c*f

    If two screws have a reciprocal product is equal to zero, it means that both screw are
    reciprocal to each other. In physical system, this is equivalent to what called as the
    principal of virtual works. The instantaneous change of position or angular position is due
    to the application of force or moment do not change the equilibrium of the system.
    """

    def __init__(self, args, kind=None):
        """
        Constructor for the Screw class.

        :param args: list of screw parameters, i.e: s; r; and h, in sequence as [s, r, h]. Or,
        it can be a screw in 6 rows by 1 column
        :type args: (list, sympy.matrices.dense.MutableDenseMatrix,
        sympy.matrices.immutable.ImmutableDenseMatrix).

        :param kind: the kind of screw, must be either 'twist' or 'wrench', default is None for
        unassigned argument.
        :type kind: (str, None), optional.
        """

        self.vector_1st = None
        self.vector_2nd = None
        self.pitch = None
        self.screw = None
        self.kind = None

        # Checking the input arguments
        if type(args).__name__ not in ["list", "MutableDenseMatrix", "ImmutableDenseMatrix"]:
            raise TypeError("Expected the 'args' data type as 'list', 'MutableDenseMatrix', "
                            "'ImmutableDenseMatrix', but got '%s'!" % type(args).__name__)

        if type(kind).__name__ not in ["str", "NoneType"]:
            raise TypeError("Expected the 'kind' data type as either 'None' for undefined kind "
                            "or 'str', but got '%s'!" % type(kind).__name__)

        # ----------------------------------------------------------------------------------------
        # For the 'list' data type of length 3 --> for constructing the screw from its parameters,
        #                                          i.e.: s, r, and h
        # ----------------------------------------------------------------------------------------
        if type(args).__name__ == "list":
            if len(args) != 3:
                raise TypeError("Expected the length of 'args' is 3 for the 'list' data type, "
                                "but got %d!" % len(args))

            if type(args[0]).__name__ not in ["MutableDenseMatrix", "ImmutableDenseMatrix"] \
                    and args[0].shape != (3, 1):
                raise TypeError("Expected the first 'args' is a unit direction of screw, s, in "
                                "the sympy 'MutableDenseMatrix' or 'ImmutableDenseMatrix' data "
                                "type of shape 3 by 1, but got '%s' data type of shape %d by %d!"
                                % (type(args[0]).__name__, args[0].shape[0], args[0].shape[1]))

            if type(args[1]).__name__ not in ["MutableDenseMatrix", "ImmutableDenseMatrix"] \
                    and args[1].shape != (3, 1):
                raise TypeError("Expected the second 'args' is a position vector of the unit "
                                "screw, r, in the sympy 'MutableDenseMatrix' or "
                                "'ImmutableDenseMatrix' data type of shape 3 by 1, but got '%s' "
                                "data type of shape %d by %d!"
                                % (type(args[1]).__name__, args[1].shape[0], args[1].shape[1]))

            if type(args[2]).__name__ not in ["int", "float", "Symbol", "Infinity"]:
                raise TypeError("Expected the third 'args' is a pitch of the screw in one of "
                                "following data type, i.e. 'int', 'float', sympy 'Symbol', or "
                                "sympy 'Infinity', but got '%s'!" % type(args[2]).__name__)

            # -----------------------------------------------------------
            #  CONSTRUCTING SCREW from ITS PARAMETERS, i.e. s, r, and h.
            # -----------------------------------------------------------
            s, r, h = args
            if h == sym.oo:
                self.vector_1st = mat.Matrix.zeros(3, 1)
                self.vector_2nd = s
            else:
                self.vector_1st = s
                self.vector_2nd = sym.expand(r.cross(s) + h * s).trigsimp()

            self.pitch = h
            self.screw = mat.Matrix.vstack(self.vector_1st, self.vector_2nd)

        # ----------------------------------------------------------------------------------------
        # For the 'MutableDenseMatrix' data type whose has the shape of 6 by 1.
        # ----------------------------------------------------------------------------------------
        elif type(args).__name__ in ["MutableDenseMatrix", "ImmutableDenseMatrix"]:
            if args.shape != (6, 1):
                raise TypeError("Expected the shape of 'args' data type is 6 by 1 for either the "
                                "'MutableDenseMatrix' or the 'ImmutableDenseMatrix' , but got the "
                                "shape of %d by %d!" % (args.shape[0], args.shape[1]))

            self.screw = args
            self.vector_1st = args[:3, :]
            self.vector_2nd = args[3:, :]
            if self.vector_1st == mat.Matrix.zeros(3, 1):
                self.pitch = sym.oo
            else:
                num = sym.trigsimp(self.vector_1st.dot(self.vector_2nd))
                den = sym.trigsimp(self.vector_1st.dot(self.vector_1st))
                self.pitch = sym.simplify(num / den)

        # ----------------------------------------------------------------------------------------
        # The kind of screw: its must be either 'twist' or 'wrench'.
        #                    'None' for the unassigned one.
        # ----------------------------------------------------------------------------------------
        if type(kind).__name__ == "str":
            if kind.lower() not in ["twist", "wrench"]:
                raise ValueError("Expected the 'kind' must be either 'twist' or 'wrench', "
                                 "but got '%s'!" % type(kind).__name__)
            self.kind = kind.lower()
        return

    def __neg__(self):
        """
        To negate the screw.

        :return: The negation of screw.
        :rtype: Screw
        """
        return Screw(-1 * self.screw, self.kind)

    def __mul__(self, scalar):
        """
        Multiplying a screw with a scalar = $ * q

        :param scalar: a scalar to multiply with the screw
        :type scalar: int, float, sympy.core.symbol.Symbol

        :return: multiplication of a screw with a scalar
        :rtype: Screw
        """
        if type(scalar).__name__ not in ["int", "float", "Symbol"]:
            raise TypeError("Expected the 'scalar' must follow one of 'int', 'float', or "
                            "'Symbol', but got '%s'!" % type(scalar).__name__)

        # Returning the multiplication of a screw with a scalar
        return Screw(self.screw * scalar, self.kind)

    def __rmul__(self, scalar):
        """
        Multiplying a screw with a scalar = q * $

        :param scalar: a scalar to multiply with the screw
        :type scalar: int, float, sympy.core.symbol.Symbol

        :return: multiplication of a screw with a scalar
        :rtype: Screw
        """
        if type(scalar).__name__ not in ["int", "float", "Symbol"]:
            raise TypeError("Expected the 'scalar' must follow one of 'int', 'float', or "
                            "'Symbol', but got '%s'!" % type(scalar).__name__)

        # Returning the multiplication of a screw with a scalar
        return Screw(self.screw * scalar, self.kind)

    def __add__(self, screw):
        """
        Adding a screw with another screw = $1 + $2

        :param screw: Another screw to add which must be the same 'kind'.
        :type screw: Screw

        :return: addition of two screws
        :rtype: Screw
        """
        if type(screw).__name__ != "Screw":
            raise TypeError("Expected the 'screw' must be 'Screw' data type, but got '%s'!"
                            % type(screw).__name__)

        if [self.kind, screw.kind] != [None, None]:
            if self.kind != screw.kind:
                raise ValueError("Expected two screws must be the same 'kind' for the addition, "
                                 "but got different '%s' and '%s'!" % (self.kind, screw.kind))

        # Returning the addition of two screws
        return Screw(self.screw + screw.screw, self.kind)

    def __radd__(self, screw):
        """
        Adding a screw with another screw = $1 + $2

        :param screw: Another screw to add which must be the same 'kind'.
        :type screw: Screw

        :return: addition of two screws
        :rtype: Screw
        """
        if type(screw).__name__ != "Screw":
            raise TypeError("Expected the 'screw' must be 'Screw' data type, but got '%s'!"
                            % type(screw).__name__)

        if [self.kind, screw.kind] != [None, None]:
            if self.kind != screw.kind:
                raise ValueError("Expected two screws must be the same 'kind' for the addition, "
                                 "but got different '%s' and '%s'!" % (self.kind, screw.kind))

        # Returning the addition of two screws
        return Screw(self.screw + screw.screw, self.kind)

    def __matmul__(self, screw):
        """
        Reciprocal product of two screws = $1 @ $2.

        :param screw: Another screw for reciprocal product
        :type screw: Screw

        :return: reciprocal product of two screws which be a scalar
        :rtype: int, float, sympy.core.symbol.Symbol
        """
        if type(screw).__name__ != "Screw":
            raise TypeError("Expected the 'screw' must be 'Screw' data type, but got '%s'!"
                            % type(screw).__name__)

        if [self.kind, screw.kind] != [None, None]:
            if self.kind == screw.kind:
                raise ValueError("Expected two screws must be not the same 'kind' for the "
                                 "reciprocal product, but got same '%s' and '%s'!"
                                 % (self.kind, screw.kind))

        # Returning the reciprocal product of two screws
        return (self.screw.T * pi_matrix() * screw.screw)[0]


#
# SCREW SYSTEM
#
class ScrewSystem:
    """
    ScrewSystem is a class to construct a screw system based on

        *  list of the screw parameters, or
        *  list of the 6 by 1 column vector,

    When a system of screw (or a screw system) is constructed, it will have three instances and
    three methods. These three instances are

        *  the screw system itself, which comprises of a list of screw that already formed using
           the Screw class,
        *  the screw system list, which contains a list of 6 by 1 column vector

    For example:

        >>> import sympy as sym
        >>> import sympy.physics.matrices as mat
        >>> a = sym.symbols("a0:4")
        >>> b = sym.symbols("b0:4")
        >>> c = sym.symbols("c0:4")
        >>> s = [mat.Matrix([1, 0, 0]),
        ...      mat.Matrix([1, 0, 0]),
        ...      mat.Matrix([0, 1, 0]),
        ...      mat.Matrix([0, 0, 1])]
        >>> r = [mat.Matrix([a[ii], b[ii], c[ii]]) for ii in range(len(a))]
        >>> h = [sym.oo, 0, 0, sym.oo]
        >>> screw_parameters = [[item[0], item[1], item[2]] for item in zip(s, r, h)]

        >>> import pyScrew4Mobility as s4m
        >>> ss_1 = s4m.ScrewSystem(screw_parameters, kind="twists")
        >>> type(ss_1)
        screw_theory_for_mobility.ScrewSystem
        >>> ss_1.screw_system  # It returns a list of Screw's object
        [<screw_theory_for_mobility.Screw at 0xddd5a6ce80>,
         <screw_theory_for_mobility.Screw at 0xddd5a6c978>,
         <screw_theory_for_mobility.Screw at 0xddd5a6c320>,
         <screw_theory_for_mobility.Screw at 0xddd5a6ccf8>]
         >>> ss_1.screw_system_list  # It returns a list of each screw in the screw system which \
                                     # defined in Plucker line coordinates
         [Matrix([
         [0],
         [0],
         [0],
         [1],
         [0],
         [0]]), Matrix([
         [  1],
         [  0],
         [  0],
         [  0],
         [ c1],
         [-b1]]), Matrix([
         [  0],
         [  1],
         [  0],
         [-c2],
         [  0],
         [ a2]]), Matrix([
         [0],
         [0],
         [0],
         [0],
         [0],
         [1]])]
         >>> ss_1.kind
         'twists'
    """
    def __init__(self, args, kind=None):
        """
        Constructor for forming the screw system of physical system.

        :param args: List of list of screw parameters, i.e: s; r; and h, in sequence as [s, r, h].
        It can be a screw in 6 by 1 column
        :type args: list, sympy.matrices.dense.MutableDenseMatrix,
        sympy.matrices.immutable.ImmutableDenseMatrix.

        :param kind: the kind of screw, must be either 'twists' or 'wrenches', default is None
        for unassigned argument.
        :type kind: str, None, optional.
        """

        self.screw_system = None
        self.screw_system_list = None
        self.kind = kind

        # Checking the input arguments
        if type(args).__name__ not in ["list", "MutableDenseMatrix", "ImmutableDenseMatrix"]:
            raise TypeError("Expected the 'args' data type as 'list', 'MutableDenseMatrix', "
                            "'ImmutableDenseMatrix', but got '%s'!" % type(args).__name__)

        if type(kind).__name__ not in ["str", "NoneType"]:
            raise TypeError("Expected the 'kind' data type as either 'None' for undefined kind "
                            "or 'str', but got '%s'!" % type(kind).__name__)

        if type(args).__name__ in ["MutableDenseMatrix", "ImmutableDenseMatrix"]:
            if args.shape[0] != 6:
                raise TypeError("Expected the number of row of 'args' is 6, but got %d!"
                                % args.shape[0])
            args = args.columnspace()

        if type(kind).__name__ == "str":
            if kind.lower() not in ["twists", "wrenches"]:
                raise ValueError("Expected the 'kind' must be either 'twists' or 'wrenches', "
                                 "but got '%s'!" % type(kind).__name__)
            self.kind = kind.lower()

        # ----------------------------------------------------------------------------------------
        # To construct screw system
        # ----------------------------------------------------------------------------------------
        if type(args).__name__ == "list":
            for item in args:
                if type(item).__name__ not in ["list", "MutableDenseMatrix",
                                               "ImmutableDenseMatrix"]:
                    raise TypeError("Expected the 'args' data type as 'list', "
                                    "'MutableDenseMatrix', 'ImmutableDenseMatrix', but got '%s'!"
                                    % type(args).__name__)

            if len(args) == 0:
                self.screw_system = []
                self.screw_system_list = []
            else:
                if self.kind == "twists":
                    kind = self.kind[:-1]
                elif self.kind == "wrenches":
                    kind = self.kind[:-2]

                self.screw_system = [Screw(item, kind) for item in args]
                self.screw_system_list = [item.screw for item in self.screw_system]
        return

    def find_reciprocal_screw_system(self):
        """
        To determine the reciprocal screw system from a screw system.

        :return: Reciprocal screw system of a screw system.
        :rtype: ScrewSystem
        """
        kind = None
        if self.kind == "twists":
            kind = "wrenches"
        elif self.kind == "wrenches":
            kind = "twists"

        Pi = pi_matrix()
        if len(self.screw_system) == 0:
            reciprocal_screw_system = [Pi[:, ii] for ii in range(Pi.shape[1])]
        else:
            reciprocal_screw_system = mat.Matrix.hstack(*self.screw_system_list).T.nullspace()
            reciprocal_screw_system = [sym.simplify(Pi * item)
                                       for item in reciprocal_screw_system]

        # Returning the reciprocal screw system
        return ScrewSystem(reciprocal_screw_system, kind)

    def find_unique_screw_system(self):
        """
        To determine a unique screw system from a screw system.

        :return: The unique screw system
        :rtype ScrewSystem
        """
        # for empty screw system --> len(self.screw_system) == 0
        unique_screw_system = []

        # for non empty screw system --> len(self.screw_system) > 0
        if len(self.screw_system) != 0:
            _h = [item.pitch for item in self.screw_system]
            _v1 = [item.vector_1st for item in self.screw_system]
            _v2 = [item.vector_2nd for item in self.screw_system]
            _ss = [mat.Matrix(item.screw) for item in self.screw_system]

            id_non_unique = set()
            for ii in range(len(self.screw_system)):
                for jj in range(ii+1, len(self.screw_system)):
                    if _h[ii] == _h[jj]:
                        if [_h[ii], _h[jj]] == 2 * [sym.oo, ] and _v2[ii] == _v2[jj]:
                            id_non_unique.add(jj)
                        elif [_h[ii], _h[jj]] == [0, 0] and _v1[ii] == _v1[jj]:
                            _v2ii, _v2jj = 2 * [mat.Matrix.zeros(3, 1), ]
                            for kk in range(len(_v2[ii])):
                                if _v2[ii][kk] != 0:
                                    _v2ii[kk] = 1  # sym.sign(_v2[ii][kk])
                                if _v2[jj][kk] != 0:
                                    _v2jj[kk] = 1  # sym.sign(_v2[jj][kk])
                            if _v2ii == _v2jj:
                                id_non_unique.add(jj)
                                _ss[ii][3:, :] = _v2[ii] + _v2[jj]

            id_unique = set(range(len(self.screw_system))).difference(id_non_unique)
            unique_screw_system = mat.Matrix.hstack(*[_ss[ii] for ii in id_unique])

        # Returning the unique screw system
        return ScrewSystem(unique_screw_system, self.kind)

    def __matmul__(self, screw_system):
        """
        Reciprocal product of two screw systems that are SS_1 @ SS_2, for instance if
        SS_1 = [$_1, $_2] is a screw system with 6 rows by 2 columns, and
        SS_2 = [$_a, $_b, $_c] is the another screw system with with 6 rows by 3 columns.
        Thus, it produces a reciprocal product od two screw system in a matrices with 2 rows by
        3 columns, as follow

            | rss_1a, rss_1b, rss_1c |
            | rss_2a, rss_2b, rss_2c |

        where rss_1a is a scalar yielded by reciprocal product of $_1 and $S_a, and so on.

        :param screw_system: Another screw system for reciprocal product calculation.
        :type screw_system: ScrewSystem

        :return: reciprocal product of two screw system.
        :rtype sympy.matrices.dense.MutableDenseMatrix,
        sympy.matrices.immutable.ImmutableDenseMatrix
        """

        if type(screw_system).__name__ != "ScrewSystem":
            raise TypeError("Expected the 'screw_system' must be 'ScrewSystem' data type, "
                            "but got '%s'!" % type(screw_system).__name__)

        if [self.kind, screw_system.kind] != [None, None]:
            if self.kind == screw_system.kind:
                raise ValueError("Expected two screw system must be not the same 'kind' for the "
                                 "reciprocal product, but got same '%s' and '%s'!"
                                 % (self.kind, screw_system.kind))

        # Returning the reciprocal product of two screw system
        ss_1 = mat.Matrix.hstack(*self.screw_system_list)
        ss_2 = mat.Matrix.hstack(*screw_system.screw_system_list)
        return ss_1.T * (pi_matrix() * ss_2)


class ManipulatorsMobility:
    """
    To determine the mobility of a robotic manipulator.

    Users need to supplied information about the unit direction vector, s, position vector, r,
    and pitch, h, of every joint on each limb had by the parallel or serial manipulators.

    The supplied information for the input arguments (args) need to be casted as list of list of
    those parameters s, r, and h.

    For example: The 3-PRRR Parallel Manipulator a.k.a Tripteron

        >>> import sympy as sym
        >>> import sympy.physics.matrices as mat

        >>> def PM_3PRRR():
        >>>     # Number of limbs
        >>>     n_limbs = 3
        >>>     # Declaring sympy symbol for its kinematic parameter
        >>>     a, b, Ly, Lz, r = sym.symbols("a b L_y L_z r")
        >>>     # Joint displacement and/or joint angular displacement
        >>>     act_spc = sym.symbols("d_{1:"+str(n_limbs+1)+"}")
        >>>     thetas = sym.symbols("\\theta_{{1:"+str(n_limbs+1)+"}{2:4}}")
        >>>     # The unit vector of a unit screw pointing at for each limb
        >>>     _s = [4 * [mat.Matrix([1, 0, 0]),],
        >>>           4 * [mat.Matrix([0, 1, 0]),],
        >>>           4 * [mat.Matrix([0, 0, 1])]]
        >>>     # pitch of the each joint for every limb
        >>>     _h = [sym.oo,] + 3 * [0,]
        >>>     # A fixed reference of frame O-XYZ
        >>>     G = mec.ReferenceFrame("G")
        >>>     # Position vector of OA_i
        >>>     rOA = [mat.Matrix([act_spc[0], Ly, 0]),
        >>>            mat.Matrix([0, act_spc[1], Lz]),
        >>>            mat.Matrix([0, 0, act_spc[2]])]
        >>>     # An auxiliary vector for determining position vector AB and BC
        >>>     _ = [mat.Matrix([0, 1, 0]),
        >>>          mat.Matrix([0, 0, 1]),
        >>>          mat.Matrix([1, 0, 0])]
        >>>     # Within this loop, it is constructed all parameters required for
        >>>     # mobility calculation
        >>>     list_of_parameters = [] # Variable for containing all parameters
        >>>                             # required for mobility calculation
        >>>     ii = 0
        >>>     for _1, _2, _3 in zip(thetas[::2], thetas[1::2], [G.x, G.y, G.z]):
        >>>         # Moving reference of frame
        >>>         H = G.orientnew("H"+str(ii+1), "Axis", [_1, _3])
        >>>         K = G.orientnew("K"+str(ii+1), "Axis", [_2, _3])
        >>>         # Position vector AB and BC in fixed reference of frame
        >>>         rAB = G.dcm(H) * (a * _[ii])
        >>>         rBC = G.dcm(K) * (b * _[ii])
        >>>         # Contruct all parameters for each limb
        >>>         s_limb = []
        >>>         for jj in range(len(_s[ii])):
        >>>             _r = rOA[ii]
        >>>             if jj == 2:
        >>>                 _r += rAB
        >>>             elif jj == 3:
        >>>                 _r += rAB + rBC
        >>>             s_limb.append([_s[ii][jj], _r, _h[jj]])
        >>>         list_of_parameters.append(s_limb)
        >>>         ii += 1
        >>>     return list_of_parameters

        >>> import pyScrew4Mobility as s4m
        >>> Mobility_PM_3PRRR = s4m.ManipulatorsMobility(PM_3PRRR(), "A 3-PRRR Parallel Manipulator")

        Mobility of the manipulator

        >>> Mobility_PM_3PRRR.Mobility
        {'Number': 3, 'Type': '3T', 'Direction': ['Tx', 'Ty', 'Tz']}

        Displaying all screw systems (for the best, use Jupyter Notebook or Jupyter Lab because
        it can render the mathematical presentation of those screw systems using LaTeX on a
        browser)
        >>> def all_screw_system(obj):
        >>>     print("Limb Twist System")
        >>>     print("=================")
        >>>     for ii, item in enumerate(obj.LimbTwistSystem):
        >>>         print("Limb: ", ii+1)
        >>>         display(item.screw_system_list)
        >>>     print("\n\nLimb Wrench System")
        >>>     print("==================")
        >>>     for ii, item in enumerate(obj.LimbWrenchSystem):
        >>>         print("Limb: ", ii+1)
        >>>         display(item.screw_system_list)
        >>>     print("\n\nPlatform Wrench System")
        >>>     print("======================")
        >>>     display(obj.PlatformWrenchSystem.screw_system_list)
        >>>     print("\n\nPlatform Twist System")
        >>>     print("=====================")
        >>>     display(obj.PlatformTwistSystem.screw_system_list)
        >>>
        >>> all_screw_system(Mobility_PM_3PRRR)

        The result of this function will not be displayed here.

    """
    def __init__(self, args, name=""):
        """
        Constructor to determine the mobility of a robotic manipulator.

        :param args: List of list of list of screw parameters
        :type args: list

        :param name: Name of a manipulator (default is an empty string, "")
        :type name: str, optional
        """
        self.Mobility = None

        if type(args).__name__ != "list" and len(args) == 0:
            raise TypeError("Expected the 'args' as a 'list' data type with the length greater "
                            "than 0, but got '%s' with length = %d!"
                            % (type(args).__name__, len(args)))

        if type(name).__name__ != "str":
            raise TypeError("Expected the 'name' of manipulator as 'str' data type, "
                            "but got '%s'!" % type(name).__name__)
        self.ManipulatorName = name

        # ----------------------------------------------------------------------------------------
        # SERIAL MANIPULATOR
        # ----------------------------------------------------------------------------------------
        if len(args) == 1:
            self.ManipulatorType = "Serial"
            self.ArmTwistSystem = ScrewSystem(args[0], kind="twists")
            self.ArmWrenchSystem = self.ArmTwistSystem.find_reciprocal_screw_system()
            self.EndEffectorTwistSystem = self.ArmWrenchSystem.find_reciprocal_screw_system()

        # ----------------------------------------------------------------------------------------
        # PARALLEL MANIPULATOR
        # ----------------------------------------------------------------------------------------
        elif len(args) > 1:
            self.ManipulatorType = "Parallel"
            self.LimbTwistSystem = [ScrewSystem(item, kind="twists") for item in args]
            self.LimbWrenchSystem = [item.find_reciprocal_screw_system()
                                     for item in self.LimbTwistSystem]
            _combine_screw_system_ = []
            for item in self.LimbWrenchSystem:
                _combine_screw_system_ += item.screw_system_list
            self.PlatformWrenchSystem = ScrewSystem(_combine_screw_system_, kind="wrenches")
            self.PlatformWrenchSystem = self.PlatformWrenchSystem.find_unique_screw_system()
            self.PlatformTwistSystem = self.PlatformWrenchSystem.find_reciprocal_screw_system()

        # ----------------------------------------------------------------------------------------
        # ACQUIRING THE MOBILITY of THE MANIPULATOR
        #   from End-effector Twist System or Platform Twist System for Serial and Parallel
        #   Manipulator, respectively
        # ----------------------------------------------------------------------------------------
        _screw_system_ = []
        if self.ManipulatorType == "Parallel":
            _screw_system_ = self.PlatformTwistSystem.screw_system
        elif self.ManipulatorType == "Serial":
            _screw_system_ = self.EndEffectorTwistSystem.screw_system

        self.Mobility = dict()
        self.Mobility["Number"] = len(_screw_system_)
        T_occurrences, R_occurrences = 0, 0
        direction = []
        for item in _screw_system_:
            if item.pitch == 0:
                R_occurrences += 1
                for ii, _ in enumerate("xyz"):
                    if item.vector_1st[ii] == 1:
                        direction.append("R" + _)
            elif item.pitch == sym.oo:
                T_occurrences += 1
                for ii, _ in enumerate("xyz"):
                    if item.vector_2nd[ii] == 1:
                        direction.append("T" + _)

        if T_occurrences == 0:
            self.Mobility["Type"] = str(R_occurrences) + "R"
        elif R_occurrences == 0:
            self.Mobility["Type"] = str(T_occurrences) + "T"
        else:
            self.Mobility["Type"] = str(T_occurrences) + "T" \
                                    + str(R_occurrences) + "R"

        self.Mobility["Direction"] = direction
        return


def pi_matrix():
    """
    Auxiliary function for matrix of Pi.
    | 0, 0, 0, 1, 0, 0 |
    | 0, 0, 0, 0, 1, 0 |
    | 0, 0, 0, 0, 0, 1 |
    | 1, 0, 0, 0, 0, 0 |
    | 0, 1, 0, 0, 0, 0 |
    | 0, 0, 1, 0, 0, 0 |

    :return: Matrix of Pi, 6 by 6.
    :rtype: sympy.matrices.dense.MutableDenseMatrix
    """
    Pi = mat.Matrix.eye(6)

    # Returning the matrix of Pi
    return mat.Matrix.hstack(Pi[:, 3:], Pi[:, :3])


# END OF FILE
