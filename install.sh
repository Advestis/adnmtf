pip3 uninstall nmtf -y || echo "No nmtf to uninstall"
pip3 install setuptools
python3 setup.py install
if [ -d "dist" ] ; then rm -r dist ; fi
if [ -d "build" ] ; then rm -r build ; fi
if ls nmtf.egg-info* &> /dev/null ; then rm -r nmtf.egg-info* ; fi
