# coding: utf-8
# coding: utf-8
#from . import irmfpro
#from . import irmfnmf
import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],'modules')))
if cmd_subfolder not in sys.path:
   sys.path.insert(0, cmd_subfolder)

import nmtf.modules.nmtf
