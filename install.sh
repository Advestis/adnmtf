pip3 uninstall nmtf -y || echo "No nmtf to uninstall"
pip3 install setuptools
python3 setup.py install
rm -r dist
rm -r build
rm -r nmtf.egg-info*
