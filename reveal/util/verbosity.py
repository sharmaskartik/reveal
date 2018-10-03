
class Verbosity():

    def __init__(self, verbose_level):
        super(Verbosity, self).__init__()
        self.verbose_level = verbose_level

    def print_msg(self, verbose_level, *argv):
        if(verbose_level >= self.verbose_level):
            for arg in argv:
                print(arg)
