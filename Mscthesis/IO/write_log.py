
"""
Log functionalities in order to save every comment in a log.
"""


class Logger:
    """
    Logger class provides a functionality to write in a logfile and display in
    screen a given message.
    """

    def __init__(self, logfile):
        self.logfile = logfile

    def write_log(self, message, display=True):
        """Function to write and/or display in a screen a message."""
        if display:
            print message
        append_line_file(self.logfile, message+'\n')


def append_line_file(filename, line):
    f = open(filename, 'a')
    f.write(line)
    f.close()
