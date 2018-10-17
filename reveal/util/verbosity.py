import re

class Verbosity():

    def __init__(self, verbose_level):
        super(Verbosity, self).__init__()
        self.verbose_level = verbose_level

    def print(self, verbose_level, *argv, **kargs):

        #get tabs value if passed
        tabs = kargs.get("tabs", 0)


        #to insert tabs at the beginning of the strings
        tab_str = ""
        for i in range(tabs):
            tab_str += "\t"

        if(verbose_level >= self.verbose_level):
            for arg in argv:
                #if there are new line characters \n at the beginning
                #insert tabs after all \n
                arg = str(arg)
                new_line_substrs = re.findall("^[\n]+", arg)
                if len(new_line_substrs) == 0:
                    print(tab_str + arg)
                else:
                    splits = arg.split(new_line_substrs[0])
                    print(new_line_substrs[0]+ tab_str + splits[1])
