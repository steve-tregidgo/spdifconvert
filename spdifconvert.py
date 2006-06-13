#!python
# Copyright 2005 Steve Tregidgo -- smst@bigfoot.com
#
# Utility to prepare DTS/AC3 files for SPDIF output.
#
# Contains a table derived from the Xine project.  The table is copyright 2001
# James Courtier-Dutton, and copyright 2000-2005 the Xine project.
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
#
# References:
# [1]: "Digital Audio Compression Standard (AC-3)"
#      - Document A/52, 2005-12-20
#      - Advanced Television Systems Committee
#
# [2]: "DTS Coherent Acoustics; Core And Extensions"
#      - ETSI TS 102 114 v1.2.1, 2002-12
#
# [3]: "xine_decoder.c"
#      - revision 1.63, 2005-05-28 11:22:05
#
# XXX TODO:
#
# - When printing file length, also print estimate of time...
#
# - Better justification for DTS output frame size than currently... I'm
#   hard-coding it to 512x4 because that's what works, but it's different to
#   AC3's 1536x4.  It may even differ for DTS files of a different source
#   bitrate, so ideally we should choose the output rate based on the input
#   rate somehow...
#
# - We only support the 16-bit packing of DTS streams, not the 14-bit version,
#   and only one endianness.  That's the one I've encountered on DVDs!  I can
#   add support for the others... we can tell what sort we've got by the magic
#   number.

VERSION = '0.1'

class SPDIFConverter:

    # Ref [1]
    ac3_sample_rate_table = [
        48000, # '00' = 0
        44100, # '01' = 1
        32000, # '10' = 2
        None,  # '11' = 3 (reserved)
    ]

    # Ref [1]
    ac3_frame_size_table = [
        ( 32, (  64,   69,   96)), # '000000' =  0
        ( 32, (  64,   70,   96)), # '000001' =  1
        ( 40, (  80,   87,  120)), # '000010' =  2
        ( 40, (  80,   88,  120)), # '000011' =  3
        ( 48, (  96,  104,  144)), # '000100' =  4
        ( 48, (  96,  105,  144)), # '000101' =  5
        ( 56, ( 112,  121,  168)), # '000110' =  6
        ( 56, ( 112,  122,  168)), # '000111' =  7
        ( 64, ( 128,  139,  192)), # '001000' =  8
        ( 64, ( 128,  140,  192)), # '001001' =  9
        ( 80, ( 160,  174,  240)), # '001010' = 10
        ( 80, ( 160,  175,  240)), # '001011' = 11
        ( 96, ( 192,  208,  288)), # '001100' = 12
        ( 96, ( 192,  209,  288)), # '001101' = 13
        (112, ( 224,  243,  336)), # '001110' = 14
        (112, ( 224,  244,  336)), # '001111' = 15
        (128, ( 256,  278,  384)), # '010000' = 16
        (128, ( 256,  279,  384)), # '010001' = 17
        (160, ( 320,  348,  480)), # '010010' = 18
        (160, ( 320,  349,  480)), # '010011' = 19
        (192, ( 384,  417,  576)), # '010100' = 20
        (192, ( 384,  418,  576)), # '010101' = 21
        (224, ( 448,  487,  672)), # '010110' = 22
        (224, ( 448,  488,  672)), # '010111' = 23
        (256, ( 512,  557,  768)), # '011000' = 24
        (256, ( 512,  558,  768)), # '011001' = 25
        (320, ( 640,  696,  960)), # '011010' = 26
        (320, ( 640,  697,  960)), # '011011' = 27
        (384, ( 768,  835, 1152)), # '011100' = 28
        (384, ( 768,  836, 1152)), # '011101' = 29
        (448, ( 896,  975, 1344)), # '011110' = 30
        (448, ( 896,  976, 1344)), # '011111' = 31
        (512, (1024, 1114, 1536)), # '100000' = 32
        (512, (1024, 1115, 1536)), # '100001' = 33
        (576, (1152, 1253, 1728)), # '100010' = 34
        (576, (1152, 1254, 1728)), # '100011' = 35
        (640, (1280, 1393, 1920)), # '100100' = 36
        (640, (1280, 1394, 1920)), # '100101' = 37
    ]

    # Ref [2]
    dts_sample_rate_table = [
        None,  # '0000' =  0
        8000,  # '0001' =  1
        16000, # '0010' =  2
        32000, # '0011' =  3
        None,  # '0100' =  4
        None,  # '0101' =  5
        11025, # '0110' =  6
        22050, # '0111' =  7
        44100, # '1000' =  8
        None,  # '1001' =  9
        None,  # '1010' = 10
        12000, # '1011' = 11
        24000, # '1100' = 12
        48000, # '1101' = 13
        None,  # '1110' = 14
        None,  # '1111' = 15
    ]

    # The stream type codes are taken from the Xine project, and were (as far
    # as I know) determined by James Courtier-Dutton.
    #
    # Ref [3], 'dts_decode_frame'.
    dts_stream_types = {
        512: 0x0b,
        1024: 0x0c,
        2048: 0x0d,
    }

    magic_numbers = {
        'ac3': '\x0b\x77',
        'dts': '\x7f\xfe\x80\x01',
    }
    frame_readers = {}

    spdif_magic = '\x72\xf8\x1f\x4e'

    # Have we written a WAV header?
    written_wav_header = False

    # Initialised by 'read_frame' (when it's seen at least one frame so
    # know the WAV details).  This will be a dict of various WAV
    # parameters.
    wav_header_info = None

    # Position in the WAV file of the actual data (after the header).
    wav_data_start_pos = None

    # Endianness of WAV file and SPDIF data.
    bigendian_wrapper = False
    bigendian_spdif_data = False

    # Number of frames we've read from the source file.
    num_frames_read = 0

    # Running total of output frames with no padding.
    running_unpadded = 0

    # Internal identifier for the incoming stream type.  This must be a key in
    # 'magic_numbers'.
    stream_type = None

    def __init__(self, options, arguments):
        self.options = options
        self.initialised = False

        if self.options['ac3'] and self.options['dts']:
            self.message(1, "Both 'ac3' and 'dts' options specified!")
            self.message(1, "Aborting.")
            return

        # 'stream_type' will either be set by a user option, or auto-detected
        # with magic numbers in 'read_frame'.
        if self.options['ac3']:
            self.stream_type = 'ac3'
        elif self.options['dts']:
            self.stream_type = 'dts'

        self.spdif_metadata = {
            # Size of each SPDIF frame; we'll pad to this.  'read_frame' will
            # set it based on the incoming stream, or it can be set as a user
            # option.
            'frame_size': None,

            # 5-bit value which is put into an SPDIF frame.  May change for
            # each frame; can be overwritten in 'read_frame'.  XXX This does
            # imply a limitation of only being able to read/write one frame at
            # a time.
            'preamble_data_dependent': 0,

            # 5-bit value indicating the type of data in the SPDIF stream.  Set
            # in 'read_frame' later.
            'data_type': None,
        }

        if self.options['frame_size'] is not None:
            self.spdif_metadata['frame_size'] = self.options['frame_size']

        self.fin_name = self.fout_name = None
        self._eof = {}
        self._seekable = {}

        if self.options['stdin']:
            pass
        elif len(arguments) > 0:
            self.fin_name = arguments[0]
        else:
            self.message(1, "Not using stdin, and no input file name specified!")
            self.message(1, "Aborting.")
            return

        if self.options['stdout']:
            pass
        elif len(arguments) > 1:
            self.fout_name = arguments[1]
        elif self.fin_name is not None:
            if self.options['wave']:
                self.fout_name = self.fin_name + '.wav'
            else:
                self.fout_name = self.fin_name + '.spdif'
        else:
            self.message(1, "Not using stdin, and no output file name specified!")
            self.message(1, "Aborting.")
            return

        if self.fin_name is None:
            self.fin = sys.stdin
            self.message(0, "Reading input from stdin")
        else:
            self.fin = open(self.fin_name, 'rb')
            self.message(0, "Reading input from file ", self.fin_name)

        if self.fout_name is None:
            self.fout = sys.stdout
            self.message(0, "Writing output to stdout")
        else:
            self.fout = open(self.fout_name, 'w+b')
            self.message(0, "Writing output to file ", self.fout_name)

        if self.options['wave'] \
        and not self.is_seekable(self.fin) \
        and not self.is_seekable(self.fout):
            self.message(2, "Require either input or output file to be seekable for WAV")
            return

        self.initialised = True


    def is_eof(self, file):
        return self._eof.get(file, False)


    def is_seekable(self, file):
        if not self._seekable.has_key(file):
            try:
                file.seek(0, 1) # does nothing!
            except IOError:
                self._seekable[file] = False
            else:
                self._seekable[file] = True
        return self._seekable[file]


    def read_raw(self, file, num_bytes):
        if self.is_eof(file):
            return ''
        chunks = []
        eof = False
        to_read = num_bytes
        max_pos = self.options.get('truncate_input')
        tell = file.tell
        while to_read > 0 and \
        (max_pos is None or tell() <= max_pos):
            data = file.read(to_read)
            if not data: # EOF
                eof = True
                break
            chunks.append(data)
            to_read = to_read - len(data)
        if eof or \
        (max_pos is not None and tell() > max_pos):
            self._eof[file] = True
        return ''.join(chunks)


    # Return data string, or None if it failed.
    def read_frame(self):
        import struct

        current_fin_pos = self.fin.tell()
        is_first_frame = self.num_frames_read == 0

        magic = ''
        discarded = ''

        # No stream type has been set, so we detect one by looking for any one
        # of the known magic numbers.
        if self.stream_type is None:
            checks = []
            for stream_type, want_magic in self.magic_numbers.items():
                checks.append( (len(want_magic), want_magic, stream_type) )
            checks.sort()
            checks.reverse() # longest magic numbers first
            if not checks:
                return None # No magic numbers defined!

            slice_magic = 1 - checks[0][0]

            while self.stream_type is None:
                for length, want_magic, stream_type in checks:
                    if magic[:length] == want_magic:
                        self.stream_type = stream_type
                        break # inner loop break
                else: # didn't break
                    next_byte = self.read_raw(self.fin, 1)
                    if self.is_eof(self.fin):
                        return None
                    if checks[0][0] > 1:
                        discarded = discarded + magic[:1 - checks[0][0]]
                        magic = magic[1 - checks[0][0]:] + next_byte
                    else:
                        discarded = discarded + magic
                        magic = next_byte

            self.message(0, 'Detected stream type: ', self.stream_type)

        # Known stream type: look for next magic number to find start of frame.
        else:
            want_magic = self.magic_numbers[self.stream_type]
            slice_magic = 1 - len(want_magic)
            if slice_magic == 0:
                slice_magic = 1
            while 1:
                if magic == want_magic:
                    break
                next_byte = self.read_raw(self.fin, 1)
                if self.is_eof(self.fin):
                    return None
                if len(want_magic) > 1:
                    discarded = discarded + magic[:1 - len(want_magic)]
                    magic = magic[1 - len(want_magic):] + next_byte
                else:
                    discarded = discarded + magic
                    magic = next_byte

        # Did we have to skip some data to get here?
        if discarded:
            self.message(-1, 'Discarded ', len(discarded), ' bytes before magic number, starting at pos ', current_fin_pos)
            self.message(-1, 'Bytes were: ', ('\\x%.2x'*len(discarded)) % tuple(map(ord, discarded)))

        # If we've never filled WAV header info before (so self doesn't know
        # anything about it) we'll do so now.  Otherwise, we set the local
        # 'wav_header_info' to None and don't record any of the relevant
        # values.
        if self.wav_header_info is None and self.options['wave']:
            wav_header_info = self.wav_header_info = {}
        else:
            wav_header_info = None

        # Find the relevant method to dispatch to.
        frame_reader = getattr(self, self.frame_readers.get(self.stream_type), None)
        if frame_reader is None:
            self.message(1, 'Unknown stream type ', self.stream_type)
            return None

        # Read the complete frame.  We pass on the magic number data we've
        # already read; 'frame_data' will probably include this at its start.
        frame_data = frame_reader(magic, self.spdif_metadata, wav_header_info, is_first_frame)
        if frame_data is not None:
            self.num_frames_read = self.num_frames_read + 1
        return frame_data


    # Ref [1] contains all the AC-3 info we could possible want.
    def read_ac3_frame(self, magic, spdif_metadata, wav_metadata, is_first_frame):
        import struct

        data = magic + self.read_raw(self.fin, 3)
        if self.is_eof(self.fin):
            return None

        sync_code = ord(data[4])
        fscod = (sync_code >> 6) & 0x03
        sample_rate = self.ac3_sample_rate_table[fscod]
        if sample_rate is None:
            self.message(1, 'Unsupported sample rate; skipping frame')
            return None
        if is_first_frame:
            self.message(-1, 'Sample rate: ', sample_rate)

        # wav_metadata is None => caller doesn't need it to be filled in.
        if wav_metadata is not None:
            wav_metadata['sample_rate'] = sample_rate
            wav_metadata['bytes_per_sample'] = 2
            wav_metadata['nchannels'] = 2

        frame_size = self.ac3_frame_size_table[sync_code & 0x3f][1][fscod]

        got_num_bytes = len(data) # got magic, CRC, sync
        to_read = (frame_size * 2) - got_num_bytes
        data = data + self.read_raw(self.fin, to_read)
        if len(data) != to_read + got_num_bytes:
            self.message(-1, 'Read short frame of ', len(data), ', expected ', to_read + got_num_bytes)

        # Didn't get enough header data!
        if len(data) <= 5:
            self.message(1, 'Couldn\'t read 5 bytes of header data; skipping frame.')
            return None

        spdif_metadata['preamble_data_dependent'] = ord(data[5]) & 0x07
        spdif_metadata['data_type'] = 0x01
        if spdif_metadata.get('frame_size') is None:
            spdif_metadata['frame_size'] = 1536 * 4
            self.message(-1, 'Padding output file to frame size ', spdif_metadata['frame_size'])

        # If we can seek in the input file, determine its length and estimate
        # the size of the output file.  We can use this to write an accurate
        # WAV header without seeking back later.
        if self.is_seekable(self.fin):
            current_position = self.fin.tell()
            self.fin.seek(0, 2)
            file_length = self.fin.tell()
            if is_first_frame:
                self.message(-1, 'AC3 length: ', file_length, '; frame length: ', len(data))

            self.fin.seek(current_position, 0)
            if wav_metadata is not None:
                nframes, remainder = divmod(file_length, len(data))
                if remainder:
                    nframes = nframes + 1
                # Guess output size: nframes worth of padded frames
                wav_metadata['data_length'] = spdif_metadata['frame_size'] * nframes
                self.message(-1, 'Guessing ', nframes, ' AC3 frames, WAV length ', wav_metadata['data_length'], ' bytes')

        return data

    frame_readers['ac3'] = 'read_ac3_frame'


    def read_dts_frame(self, magic, spdif_metadata, wav_metadata, is_first_frame):
        import struct

        # Ref [2] for DTS decoding info...
        data = magic + self.read_raw(self.fin, 5)
        if self.is_eof(self.fin) or len(data) < 9:
            return None

        # Ref [2], section 5.4.1, FSIZE.
        # 5 bytes after magic number are:
        #   abbbbbcd ddddddee eeeeeeee eeeeffff ffggggxx
        # a: FTYPE (Frame Type)
        # b: SHORT (Deficit Sample Count)
        # c: CPF (CRC Present Flag)
        # d: NBLKS (Number Of PCM Sample Blocks)
        # e: FSIZE: Primary Frame Byte Size
        # f: AMODE (Audio Channel Arrangement)
        # g: SFREQ (Core Audio Sampling Frequency)
        #
        # FSIZE+1 is the byte size of the current frame (including sync).
        # NBLKS+1 is the number of blocks, each containing 32 PCM samples,
        # in the current frame.
        fsize = int((struct.unpack('>L', data[4:8])[0] >> 4) & 0x3fff)
        nblks = int((struct.unpack('>H', data[4:6])[0] >> 2) & 0x7f)

        if is_first_frame:
            self.message(-1, 'DTS frame data: FSIZE ', fsize)
            self.message(-1, 'DTS frame data: NBLKS ', nblks)

        got_num_bytes = len(data) # got magic and a bit more
        to_read = fsize + 1 - got_num_bytes
        data = data + self.read_raw(self.fin, to_read)
        if len(data) != to_read + got_num_bytes:
            self.message(-1, 'Read short frame of ', len(data), ', expected ', to_read + got_num_bytes)

        sfreq = (ord(data[8]) >> 2) & 0x0f
        sample_rate = self.dts_sample_rate_table[sfreq]
        if is_first_frame:
            self.message(-1, 'DTS frame data: SFREQ ', sfreq, '; sample rate is ', sample_rate)
        if sample_rate is None:
            self.message(1, 'Unsupported sample rate; skipping frame')
            return None

        num_samples = 32 * (nblks + 1)
        if not self.dts_stream_types.has_key(num_samples):
            self.message(1, 'Unsupported number of samples ', num_samples)
            return None
        spdif_metadata['data_type'] = self.dts_stream_types[num_samples]
        spdif_metadata['preamble_data_dependent'] = 0

        if spdif_metadata.get('frame_size') is None:
            # XXX This next bit should be justified somehow, probably by
            # considering sample rate etc.
            spdif_metadata['frame_size'] = 512 * 4
            self.message(-1, 'Padding output file to frame size ', spdif_metadata['frame_size'])

        if wav_metadata is not None:
            wav_metadata['sample_rate'] = sample_rate
            wav_metadata['bytes_per_sample'] = 2
            wav_metadata['nchannels'] = 2

        if self.is_seekable(self.fin):
            current_position = self.fin.tell()
            self.fin.seek(0, 2)
            file_length = self.fin.tell()
            if is_first_frame:
                self.message(-1, 'DTS length: ', file_length, '; frame length: ', len(data))
            self.fin.seek(current_position, 0)
            if wav_metadata is not None:
                nframes, remainder = divmod(file_length, fsize + 1)
                if remainder:
                    nframes = nframes + 1
                # Guess output size: nframes worth of padded frames
                wav_metadata['data_length'] = spdif_metadata['frame_size'] * nframes
                self.message(-1, 'Guessing ', nframes, ' DTS frames, WAV length ', wav_metadata['data_length'], ' bytes')

        return data

    frame_readers['dts'] = 'read_dts_frame'


    def write_wav_header_if_not_written(self):
        if not self.written_wav_header:
            self.write_wav_header()


    def write_wav_header(self):
        import struct

        data_length = self.wav_header_info.get('data_length', 0)
        endian_char = self.bigendian_wrapper and '>' or '<'

        if self.written_wav_header:
            return
        if self.wav_header_info is None:
            raise RuntimeError("Asked to write WAV header, but no info available.")

        current_position = self.fout.tell()
        if current_position:
            raise RuntimeError("WAV header not at start of output file.", current_position)

        self.fout.write('RIFF')
        self.fout.write(struct.pack(endian_char + 'L', data_length + 36))
        self.fout.write('WAVEfmt ')

        nchannels = self.wav_header_info['nchannels'] = 2
        block_size = nchannels * self.wav_header_info['bytes_per_sample']
        bytes_per_second = block_size * self.wav_header_info['sample_rate']
        self.fout.write(struct.pack(endian_char + 'LHHLLHH',
            16, # minimal fmt header size
            1, # compression type
            nchannels,
            self.wav_header_info['sample_rate'], # eg 48000
            bytes_per_second,
            block_size,
            self.wav_header_info['bytes_per_sample'] * 8, # bits per sample
        ))
        self.fout.write('data')
        self.fout.write(struct.pack(endian_char + 'L', data_length))

        self.wav_data_start_pos = self.fout.tell()
        self.written_wav_header = True


    def fix_wav_header(self):
        import struct

        if not self.is_seekable(self.fout):
            return

        # XXX The 'is_seekable' test doesn't always seem to be sufficient...
        if self.options['stdout']:
            return

        if self.wav_header_info is None:
            raise RuntimeError("Asked to fix WAV header, but no info available.")

        current_position = self.fout.tell()
        self.fout.seek(0, 2)
        data_length = self.fout.tell() - self.wav_data_start_pos
        if data_length == self.wav_header_info['data_length']:
            # No fixing needed!
            self.message(-1, 'WAV header does not need fixing (good estimate!)')
            self.fout.seek(current_position, 0)
            return

        self.message(-1, 'Fixing WAV header. Data length ', data_length, ' was marked as ', self.wav_header_info.get('data_length', 0))

        endian_char = self.bigendian_wrapper and '>' or '<'
        self.fout.seek(self.wav_data_start_pos - 4, 0)
        self.fout.write(struct.pack(endian_char + 'L', data_length))

        self.fout.seek(4, 0)
        self.fout.write(struct.pack(endian_char + 'L', data_length + self.wav_data_start_pos - 8))

        self.fout.seek(current_position, 0)


    def write_spdif_frame(self, data):
        import array
        import struct

        endian_char = self.bigendian_spdif_data and '>' or '<'

        # Ref [1], Annex B, section 4.5: if we've had a run of data with
        # absolutely no padding, we need to insert a pair of zeroes to
        # facilitate auto-detection of stream type.
        if self.running_unpadded >= (4096 * 32):
            self.fout.write('\000\000')
            self.running_unpadded = 0

        # First 32 bits of preamble = sync word
        header = self.spdif_magic

        # Next 16 bits of preamble = burst info.
        #   nnnddddd errttttt
        #   nnn: data stream number (0b000; main stream)
        # ddddd: data type dependent...
        #     e: error flag, set to 0b0
        #    rr: reserved, equal to 0b00
        # ttttt: data type (eg 0b00001 for AC3)
        header = header + struct.pack('B', self.spdif_metadata['data_type'] & 0x1f)
        header = header + struct.pack('B', self.spdif_metadata['preamble_data_dependent'] & 0x1f)

        # Next 16 bits of preamble = burst bits
        header = header + struct.pack(endian_char + 'H', len(data) * 8)

        padding_length = self.spdif_metadata['frame_size'] - len(data) - len(header)
        if padding_length == 0:
            self.running_unpadded = self.running_unpadded + len(data) + len(header)

        data_array = array.array('H', data + ('\000' * padding_length))
        data_array.byteswap()

        self.fout.write(header)
        data_array.tofile(self.fout)


    def go(self):

        import time
        start_time = time.time()

        while 1:
            frame = self.read_frame()
            if frame is None:
                break
            if self.options['wave']:
                self.write_wav_header_if_not_written()
            self.write_spdif_frame(frame)

        if self.options['wave']:
            self.fix_wav_header()

        end_time = time.time()

        spdif.message(-1, "Completed transformation.  Took %dm%.2ds." % divmod(end_time-start_time, 60))


    def message(self, msg_level, *message_items):
        import sys
        if msg_level >= self.options['quiet']:
            sys.stderr.write(''.join(map(str, message_items)) + '\n')


def print_help(program_name):
    assume_max_option_width = 18
    option_format = '%%%ds' % (assume_max_option_width,)

    print
    print "  Copyright 2005 Steve Tregidgo -- smst@bigfoot.com"
    print "  Version:", VERSION
    print
    print "  Usage: %s [options] [input_file [output_file]]" % (program_name,)
    print
    print "  Given an AC3 or DTS file from a DVD-Video, this utility will write a WAV file"
    print "  encapsulating that digital data in an IEC61937 stream.  If that WAV is sent"
    print "  out over an S/PDIF link in the normal IEC60958 way, without modification for"
    print "  volume etc, a digital receiver should be able to play the multichannel audio"
    print "  therein."
    print
    print "  Start with a file taken from a DVD-Video; de-multiplex the AC3 or DTS file"
    print "  required, then run this utility on it to generate a WAV file.  It is"
    print "  recommended that the WAV be further transformed to FLAC, which supports the"
    print "  storage of metadata values, although the compression will likely be very poor."
    print "  This utility was written for the benefit of SqueezeBox2 owners; playback of"
    print "  such a WAV or FLAC on the SB2, with digital output, will result in the"
    print "  original multichannel audio being passed through to the receiver."
    print
    print "  If no output file is specified, an output name will be derived from the input"
    print "  file name.  If no input file is specified, you must specify the '--stdin'"
    print "  option to read from stdin.  If you want to read from stdin but write to an"
    print "  output file, just use a dummy input file name along with '--stdin'."
    print
    print "  Options:"
    for option_string, text in [
        ('--stdin', 'Read input from stdin.'),
        ('--stdout', 'Write output to stdout.'),
        ('', 'NOTE: using both stdin and stdout with WAV output will give\na WAV file with invalid data length.'),
        ('', ''),
        ('-a, --ac3', 'The input stream is expected to be AC-3.',),
        ('-d, --dts', 'The input stream is expected to be 16-bit big-endian DTS\n(such as is obtained from ripping a DVD).',),
        ('', 'NOTE: in most cases \'--ac3\' and \'--dts\' can be omitted,\nand the stream type will be auto-detected.'),
        ('', ''),
        ('-w, --wave', 'Prefix output with a WAV header (default).  If not using\nstdout for output, the WAV header will definitely contain an\naccurate data length; if using stdout, the recorded length\nwill be an estimate, which may be wrong if frame sizes vary\nin the source file.  If using both stdin and stdout, the WAV\nheader will definitely be wrong (the length will be zero).'),
        ('-r, --raw', 'Don\'t write a WAV header.  If both \'--raw\' and \'--wave\' are\nspecified, the last one supplied takes precedence.'),
        ('', ''),
        ('-q, --quiet', 'Print fewer messages to stderr.  Repeat this option to\nsilence all messages.'),
        ('-v, --verbose', 'Print more messages to stderr.'),
        ('--frame-size=SIZE', 'Advanced users\' setting: override the frame size in the\noutput file.  We normally pad to a frame size of 6144\n(=1536*4) or 2048 (=512*4); if the resultant WAV file\ndoesn\'t appear to be the correct length (as indicated by any\nnormal audio software which can read WAV files -- but don\'t\nplay the file!) then adjusting this value might help.  Use\n\'--verbose\' to discover the frame size used, and then try\nrunning again with the \'--frame-size\' option set to a\ndifferent value (multiplied up or down by an integer).  If\nyou discover something which works for your file, please let\nme know so I can try setting that size automatically\ndepending on the input file.'),
        ('--truncate-input=M', 'Don\'t read more than M bytes from the input file.  If the\nMth byte falls within a frame, that partial frame will not\nbe converted.'),
        ('', ''),
        ('-h, -?, --help', 'Display this help text.'),
        ('-V, --version', 'Display the version number.'),
    ]:
        print option_format % (option_string,),
        for i, line in enumerate(text.split('\n')):
            if i:
                print (' ' * assume_max_option_width),
            print line

    print
    print "  If you have any comments or improvements, please email smst@bigfoot.com"
    print

if __name__ == '__main__':

    import getopt
    import os
    import struct
    import sys

    program_name = os.path.split(sys.argv[0])[-1]

    short_options = ''.join([
        'a', # == 'ac3'
        'd', # == 'dts'
        'w', # == 'wave'
        'r', # == 'raw'
        'q', # == 'quiet'
        'v', # == 'verbose'
        'V', # == 'version'
        'h', # == 'help'
        '?', # == 'help'
    ])
    long_options = [
        'ac3',
        'dts',
        'stdin',
        'stdout',
        'wave',
        'raw',
        'quiet',
        'verbose',
        'frame-size=',
        'truncate-input=',
        'help',
        'version',
    ]

    try:
        cli_options, cli_arguments = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error, err:
        sys.stderr.write("Usage error!\n" + str(err) + '\n\n')
        sys.stderr.write("For help, type: %s --help\n" % (program_name,))
        sys.exit(1)

    options = {
        'wave': True,
        'stdin': False,
        'stdout': False,
        'quiet': 0, # 0, 1, 2
        'ac3': False,
        'dts': False,
        'frame_size': None,
        'truncate_input': None,
        'help': False,
        'version': False,
    }

    for option, value in cli_options:
        if option in ['--ac3', '-a']:
            options['ac3'] = True
        elif option in ['--dts', '-d']:
            options['dts'] = True
        elif option in ['--wave', '-w']:
            options['wave'] = True
        elif option in ['--raw', '-r']:
            options['wave'] = False
        elif option in ['--quiet', '-q']:
            options['quiet'] = options['quiet'] + 1
        elif option in ['--verbose', '-v']:
            options['quiet'] = options['quiet'] - 1
        elif option in ['--stdin']:
            options['stdin'] = True
        elif option in ['--stdout']:
            options['stdout'] = True
        elif option in ['--frame-size']:
            try:
                options['frame_size'] = int(value)
            except ValueError:
                sys.stderr.write("'frame-size' option not an int: " + value)
                sys.exit(1)
        elif option in ['--truncate-input']:
            try:
                options['truncate_input'] = long(value)
            except ValueError:
                sys.stderr.write("'truncate-input' option not an int: " + value)
                sys.exit(1)
        elif option in ['--help', '-h', '-?']:
            options['help'] = True
        elif option in ['--version', '-V']:
            options['version'] = True

    if options['version']:
        sys.stderr.write("Version: %s\n" % (VERSION,))
        sys.exit(0)

    if options['help']:
        print_help(program_name)
        sys.exit(0)

    spdif = SPDIFConverter(options, cli_arguments)
    if not spdif.initialised:
        spdif.message(0, "There was an error initialising.  Aborting.")
        sys.exit(1)

    try:
        spdif.go()
    except RuntimeError, error:
        spdif.message(0, "Received RuntimeError during conversion:")
        spdif.message(0, error)

    except KeyboardInterrupt:
        print "Interrupted!  Read this many bytes:", spdif.fin.tell()
