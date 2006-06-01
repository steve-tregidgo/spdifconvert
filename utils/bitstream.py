
class BitstreamReader:
    def __init__(self, fname):
        self.fname = fname
        self._chunk_size = 32
        self.file = open(fname, 'rb')
        self.reset()


    def bits(self, n=1):
        bits = []
        while n:
            if not self.buffer:
                self.buffer = map(ord, self.file.read(self._chunk_size))
                self.unused_bits = 8
            if not self.buffer:
                raise ValueError("Can't supply %s bits; only %s left!" % (n, len(bits)))
            in_this_byte = min(self.unused_bits, n)
            shifted_bits = self.buffer[0] >> (self.unused_bits - in_this_byte)
            while in_this_byte:
                bits.append((shifted_bits >> (in_this_byte-1)) & 1)
                in_this_byte -= 1
                self.unused_bits -= 1
                n -= 1
            if self.unused_bits == 0:
                self.buffer = self.buffer[1:]
                self.unused_bits = 8

        return bits


    def bits_and_decimal(self, n=1):
        bits = self.bits(n)

        decimal_value = 0
        for bit in bits:
            decimal_value = (decimal_value << 1) + bit

        grouped_bits = []
        while bits:
            grouped_bits.append(''.join(map(str, bits[-8:])))
            bits = bits[:-8]
        bits_string = ' '.join(grouped_bits)

        return decimal_value, bits_string


    def print_bits_and_decimal(self, n=1, first_column=16):
        decimal_value, bits_string = self.bits_and_decimal(n)
        str_value = '%d' % (decimal_value,)
        print '%s%s%s' % (str_value, ' '*(first_column-len(str_value)), bits_string)


    def print_bits_and_hex(self, n=1, first_column=16):
        decimal_value, bits_string = self.bits_and_decimal(n)
        str_value = '%x' % (decimal_value,)
        print '%s%s%s' % (str_value, ' '*(first_column-len(str_value)), bits_string)


    def print_bits_hex_and_decimal(self, n=1, first_column=16, second_column=16):
        decimal_value, bits_string = self.bits_and_decimal(n)
        str_value_1 = '%d' % (decimal_value,)
        str_value_2 = '%x' % (decimal_value,)
        print '%s%s%s%s%s' % (str_value_1, ' '*(first_column-len(str_value_1)), str_value_2, ' '*(second_column-len(str_value_2)), bits_string)


    def reset(self):
        self.file.seek(0, 0)
        self.buffer = []
        self.unused_bits = 0


'''
# Applying this reader to a test DTS file:

>>> dts = bitstream.BitstreamReader('test.dts')
>>> get = dts.print_bits_and_hex_and_decimal
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
AttributeError: BitstreamReader instance has no attribute 'print_bits_and_hex_an
d_decimal'
>>> get = dts.print_bits_hex_and_decimal
>>> get(32) # SYNC
2147385345      7ffe8001        00000001 10000000 11111110 01111111
>>> get(1) # FTYPE
1               1               1
>>> get(5) # SHORT
31              1f              11111
>>> get(1) # CPF -- CRC Present Flag
0               0               0
>>> get(7) # NBLKS -- Number of PCM Sample Blocks
15              f               0001111
>>> get(14) # FSIZE -- Primary Frame Byte Size
2012            7dc             11011100 000111
>>> get(6) # AMODE -- Audio Channel Arrangement
9               9               001001
>>> # = C + L + R + SL + SR ... no LFE?
...
>>> get(4) # SFREQ -- Core Audio Sampling Frequency
13              d               1101
>>> # = 48 kHz
...
>>> get(5) # RATE -- Transmission Bit Rate
24              18              11000
>>> # 1536 kbit/s
...
>>> get(1) # MIX
0               0               0
>>> get(1) # DYNF
0               0               0
>>> get(1) # TIMEF
0               0               0
>>> get(1) # AUXF
0               0               0
>>> get(1) # HDCD
0               0               0
>>> get(3) # EXT_AUDIO_ID
0               0               000
>>> get(1) # EXT_AUDIO
0               0               0
>>> get(1) # ASPF
1               1               1
>>> get(2) # LFF -- Low Frequency Effects Flag
2               2               10
>>> get(1) # HFLAG
1               1               1
>>> get(1) # FILTS
0               0               0
>>> get(4) # VERNUM
7               7               0111
>>> get(2) # CHIST
1               1               01
>>> get(3) # PCMR
6               6               110
>>> get(1) # SUMF
0               0               0
>>> get(4) # DIALNORM
0               0               0000


# Read 4 bytes, check sync.
# Read 4 more bytes, then apply flen:
#def flen(bytes):
#    return (
#            ((ord(bytes[1]) & 0x03) << 14)
#            |
#            (ord(bytes[2]) << 4)
#            |
#            ((ord(bytes[3]) >> 4) & 0x0f)
#    )
#...to get frame length.
# Read that many bytes, minus 4 for sync and 4 for flen, plus 1 (see FSIZE in spec).
# Repeat.

'''
