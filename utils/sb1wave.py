# Basic beginning of a modified spdifconvert.py, the idea being to support
# SB1-friendly output.


class to_be_gutted:

    def __init__(self, options, arguments):

        self.fin_name = self.fout_name = None

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
        chunks = []
        eof = False
        to_read = num_bytes
        while to_read > 0:
            data = file.read(to_read)
            if not data: # EOF
                eof = True
                break
            chunks.append(data)
            to_read = to_read - len(data)
        if eof:
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
        ('', ''),
        ('-h, -?, --help', 'Display this help text.'),

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
    import sys

    program_name = os.path.split(sys.argv[0])[-1]

    short_options = ''.join([
        'q', # == 'quiet'
        'v', # == 'verbose'
        'h', # == 'help'
        '?', # == 'help'
    ])
    long_options = [
        'stdin',
        'stdout',
        'quiet',
        'verbose',
        'help',
    ]

    try:
        cli_options, cli_arguments = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error, err:
        sys.stderr.write("Usage error!\n" + str(err) + '\n\n')
        sys.stderr.write("For help, type: %s --help\n" % (program_name,))
        sys.exit(1)

    options = {
        'stdin': False,
        'stdout': False,
        'quiet': 0, # 0, 1, 2
        'help': False,
    }

    for option, value in cli_options:
        if option in ['--quiet', '-q']:
            options['quiet'] = options['quiet'] + 1
        elif option in ['--verbose', '-v']:
            options['quiet'] = options['quiet'] - 1
        elif option in ['--stdin']:
            options['stdin'] = True
        elif option in ['--stdout']:
            options['stdout'] = True
        elif option in ['--help', '-h', '-?']:
            options['help'] = True

    if options['help']:
        print_help(program_name)
        sys.exit(0)

    try:
        go(options, cli_arguments)
    except:
        if options['quiet'] > -1:
            sys.stderr.write("There was an error.  Aborting.\n")

def go(options, arguments):
    import struct
    import sys

    ifname = ofname = None
    if options['stdin']:
        input_file = sys.stdin
        if options['quiet'] > -1:
            sys.stderr.write("Reading input from stdin\n")
    elif len(arguments) > 0:
        ifname = arguments[0]
        try:
            input_file = open(ifname, 'rb')
        except (IOError, OSError):
            if options['quiet'] > -1:
                sys.stderr.write("Could not open file for reading: %s\n" % (ifname,))
                sys.stderr.write("Aborting.\n")
            sys.exit(1)
        if options['quiet'] > -1:
            sys.stderr.write("Reading input from %s\n" % (ifname,))
    else:
        if options['quiet'] > -1:
            sys.stderr.write("Not using stdin, and no input file name specified!\n")
            sys.stderr.write("Aborting.\n")
        sys.exit(1)

    if options['stdout']:
        output_file = sys.stdout
        if options['quiet'] > -1:
            sys.stderr.write("Writing output to stdout\n")
    else:
        if len(arguments) > 1:
            ofname = arguments[1]
        elif ifname is not None:
            ofname = ifname + '.sb1.wav'
        else:
            if options['quiet'] > -1:
                sys.stderr.write("No output file name specified, and no input file name to guess from.\n")
                sys.stderr.write("Aborting.\n")
            sys.exit(1)
        try:
            output_file = open(ofname, 'w+b')
        except (IOError, OSError):
            if options['quiet'] > -1:
                sys.stderr.write("Could not open file for writing: %s\n" % (ofname,))
                sys.stderr.write("Aborting.\n")
            sys.exit(1)
        if options['quiet'] > -1:
            sys.stderr.write("Writing output to %s\n" % (ofname,))
    else:
        if options['quiet'] > -1:
            sys.stderr.write("Not using stdin, and no output file name specified!\n")
            sys.stderr.write("Aborting.\n")
        sys.exit(1)
        return

    riff_header = input_file.read(8)
    if riff_header[:4] != 'RIFF':
        if options['quiet'] > -1:
            sys.stderr.write("Missing RIFF header; not a valid WAV file.\n")
        sys.exit(1)

    wave_header = input_file.read(28)
    if wave_header[:4] != 'WAVE':
        if options['quiet'] > -1:
            sys.stderr.write("Missing WAVE header; not a valid WAV file.\n")
        sys.exit(1)
    if wave_header[4:8] != 'fmt ':
        if options['quiet'] > -1:
            sys.stderr.write("Missing 'fmt ' header; not a valid WAV file.\n")
        sys.exit(1)
    fmt_data = struct.unpack('LHHLLHH', wave_header[8:])

    # XXX Here we need to examine 'fmt_data' to see what sizes of values we're
    # to read from the data chunks.

    data_header = input_file.read(8)
    if data_header[:4] != 'data':
        if options['quiet'] > -1:
            sys.stderr.write("Missing data header; not a valid WAV file.\n")
        sys.exit(1)

    output_file.write(riff_header + wave_header + data_header)

    # XXX Now we read all the data in from the file (not necessarily in one
    # go!) and unpack it into a series of values (using knowledge of the format
    # to decide what to grab).  Then we invert the values-- that's the bit I
    # need to find out how to do!! --and pack them up into the output file.

