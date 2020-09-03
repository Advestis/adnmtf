sudo pip3 uninstall nmtf -y || echo "No nmtf to uninstall"
sudo pip3 install setuptools
sudo python3 setup.py install
sudo rm -r dist
sudo rm -r build
sudo rm -r nmtf.egg-info*
