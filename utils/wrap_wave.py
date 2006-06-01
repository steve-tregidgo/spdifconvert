PROGRESS_BAR_ITEMS = 50
READ_CHUNK_SIZE = 8192

if __name__ == '__main__':
    import getopt
    import struct
    import sys

    short_options = ''
    long_options = [
        'rate=',
    ]

    try:
        options, arguments = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error, err:
        print 'Usage error!'
        print err
        print
        print 'Hit ENTER to close.'
        raw_input()
        sys.exit(1)

    opt_rate = 48000

    for option, value in options:
        if option in ['--rate']:
            try:
                opt_rate = int(value)
            except ValueError:
                print "Invalid int for sample rate:", value
                sys.exit(1)

    if len(arguments) < 1:
        print "Supply input file name as argument."
        sys.exit(1)

    fin_name = arguments[0]
    print "Input file:", fin_name

    if len(arguments) > 1:
        fout_name = arguments[1]
    else:
        fout_name = fin_name + '.wav'
    print "Output file:", fout_name

    print "Using sample rate:", opt_rate

    fin = open(fin_name, 'rb')
    fin.seek(0, 2)
    file_length = fin.tell()
    fin.seek(0, 0)

    fmt_values = (1, 2, opt_rate, opt_rate*4, 4, 16)
    fmt_header = 'fmt ' +  struct.pack('LHHLLHH', 16, *fmt_values)
    data_header = 'data' + struct.pack('L', file_length)
    wave_header = 'WAVE' + fmt_header + data_header
    riff_header = 'RIFF' + struct.pack('L', file_length + len(wave_header))

    fout = open(fout_name, 'wb')
    fout.write(riff_header)
    fout.write(wave_header)

    print "Written RIFF header.  Copying main data..."

    total_bytes_to_process = file_length
    sys.stdout.write('[' + ('.'*PROGRESS_BAR_ITEMS) + ']\r[')
    wrote_progress = -1

    while 1:
        data = fin.read(READ_CHUNK_SIZE)
        new_progress = int(PROGRESS_BAR_ITEMS * (fin.tell() / float(total_bytes_to_process)))
        if new_progress > wrote_progress:
            sys.stdout.write('#'*(new_progress-wrote_progress))
            wrote_progress = new_progress
        if not data:
            break
        fout.write(data)

    sys.stdout.write((PROGRESS_BAR_ITEMS - wrote_progress)*'#')
    sys.stdout.write(']\n')

    fin.close()
    fout.close()

    print "Done."
