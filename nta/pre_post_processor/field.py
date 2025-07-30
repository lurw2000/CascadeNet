import numpy as np
from enum import Enum
import pandas as pd
import warnings

class OutputType(Enum):
    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"


class Normalization(Enum):
    ZERO_ONE = "ZERO_ONE"
    MINUSONE_ONE = "MINUSONE_ONE"


class Output(object):
    def __init__(self, type_, dim, normalization=None, is_gen_flag=False):
        self.type_ = type_
        self.dim = dim
        self.normalization = normalization
        self.is_gen_flag = is_gen_flag

        if type_ == OutputType.CONTINUOUS and normalization is None:
            raise Exception("normalization must be set for continuous output")
        

class Field(object):
    def __init__(self, name):
        self.name = name

    def normalize(self):
        raise NotImplementedError

    def denormalize(self):
        raise NotImplementedError

    def getOutputType(self):
        raise NotImplementedError


class ContinuousField(Field):
    def __init__(self, norm_option, min_x=None, max_x=None, dim_x=1,
                 log1p_norm=False, *args, **kwargs):
        super(ContinuousField, self).__init__(*args, **kwargs)

        self.min_x = min_x
        self.max_x = max_x
        if self.min_x == self.max_x:
            warnings.warn("min_x and max_x are the same! Normalize everything to 1.0")

        self.norm_option = norm_option
        self.dim_x = dim_x
        self.log1p_norm = log1p_norm
        if not self.min_x == self.max_x and self.log1p_norm:
            self.min_x = np.log1p(self.min_x)
            self.max_x = np.log1p(self.max_x)

    # Normalize x in [a, b]: x' = (b-a)(x-min x)/(max x - minx) + a
    def normalize(self, x):

        # corner case: min_x == max_x
        if self.min_x == self.max_x:
            return np.ones_like(x)

        try:
            if self.log1p_norm:
                x = np.log1p(x)
        except: # TODO
            pass
        
        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            return np.asarray((x - self.min_x) / (self.max_x - self.min_x))

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            return np.asarray(2 * (x - self.min_x)
                              / (self.max_x - self.min_x) - 1)

        else:
            raise Exception("Not valid normalization option!")

    def denormalize(self, norm_x):

        # corner case: min_x == max_x
        if self.min_x == self.max_x:
            return np.ones_like(norm_x) * self.min_x

        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            to_return = norm_x * float(self.max_x - self.min_x) + self.min_x

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            to_return = (norm_x+1) / 2.0 * \
                float(self.max_x - self.min_x) + self.min_x

        else:
            raise Exception("Not valid normalization option!")

        try:
            if self.log1p_norm:
                to_return = np.expm1(to_return)
        except: # TODO
            pass
        
        return to_return

    def getOutputType(self):
        return Output(
            type_=OutputType.CONTINUOUS,
            dim=self.dim_x,
            normalization=self.norm_option
        )


class DiscreteField(Field):
    def __init__(self, choices, *args, **kwargs):
        super(DiscreteField, self).__init__(*args, **kwargs)

        if not isinstance(choices, list):
            raise Exception("choices should be a list")
        self.choices = choices

    def normalize(self, x):
        index = self.choices.index(x)

        norm_x = np.zeros_like(self.choices, dtype=float)
        norm_x[index] = 1.0

        return list(norm_x)

    def denormalize(self, norm_x):
        index = np.argmax(norm_x)

        return self.choices[index]

    def getOutputType(self):
        return Output(
            type_=OutputType.DISCRETE,
            dim=len(self.choices)
        )

class StringField(DiscreteField):
    def __init__(self, strings, *args, **kwargs):
        """
        e.g. proto
            strings = ["TCP", "UDP", "ICMP"]
            choices = [0, 1, 2]
            normalize("TCP") = [1, 0, 0]
            normalize(["TCP", "UDP"]) = [[1, 0, 0], [0, 1, 0]]
            denormalize([0, 1, 0]) = "UDP"
        """
        choices = list(range(len(strings)))
        super(StringField, self).__init__(choices, *args, **kwargs)
        self.choices = choices
        self.strings = strings
        self.choice2string = dict(zip(choices, strings))
        self.string2choice = dict(zip(strings, choices))
    
    def normalize(self, x):
        if isinstance(x, str):
            index = self.string2choice[x]

            norm_x = np.zeros_like(self.choices, dtype=float)
            norm_x[index] = 1.0

        # else, x is an iterable of strings
        elif isinstance(x, list) or isinstance(x, pd.Series):
            index = [self.string2choice[s] for s in x]

            norm_x = np.zeros((len(x), len(self.choices)), dtype=float)
            for i, idx in enumerate(index):
                norm_x[i, idx] = 1.0

        return norm_x         

    def denormalize(self, norm_x):
        if len(norm_x.shape) == 1:
            index = np.argmax(norm_x)
            return self.choice2string[index]

        elif len(norm_x.shape) == 2:
            index = np.argmax(norm_x, axis=1)
            return [self.choice2string[idx] for idx in index]    


class BitField(Field):
    def __init__(self, num_bits, *args, **kwargs):
        super(BitField, self).__init__(*args, **kwargs)

        self.num_bits = num_bits

    def normalize(self, input_x, input_type="decimal"):
        if input_type == "decimal":
            decimal_x = input_x
        elif input_type == "ip":
            decimal_x = int(input_x.replace(".", ""))
        else:
            raise Exception("{} is not a valid input type".format(input_type))
        
        bin_x = bin(int(decimal_x))[2:].zfill(self.num_bits)
        bin_x = [int(b) for b in bin_x]

        bits = []
        for b in bin_x:
            if b == 0:
                bits += [1.0, 0.0]

            elif b == 1:
                bits += [0.0, 1.0]

            else:
                print("Binary number is zero or one!")

        return bits

    def denormalize(self, bin_x, output_type="decimal"):
        if not isinstance(bin_x, list):
            # raise Exception("Bit array should be a list")
            bin_x = list(bin_x)

        assert len(bin_x) == 2*self.num_bits, "length of bit array is wrong!"

        bits = "0b"
        for i in range(self.num_bits):
            index = np.argmax(bin_x[2*i:2*(i+1)])

            if index == 0:
                bits += "0"

            elif index == 1:
                bits += "1"

            else:
                raise Exception("Bits array is ZERO or ONE!")

        decimal_x = int(bits, 2)

        if output_type == "decimal":
            return decimal_x
        elif output_type == "ip":
            # e.g. 3232235777 -> 
            return ".".join([str((decimal_x >> (8 * i)) & 0xFF) for i in range(3, -1, -1)])
        else:
            raise Exception("{} is not a valid output type".format(output_type))

    def getOutputType(self):
        outputs = []

        for i in range(self.num_bits):
            outputs.append(
                Output(
                    type_=OutputType.DISCRETE,
                    dim=2
                ))

        return outputs
