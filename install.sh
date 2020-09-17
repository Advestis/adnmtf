#!/bin/bash

publish=false
PACKAGE="nmtf"
VERSION="3"
COMMAND="install"

while true; do
  case "$1" in
    -p | --publish) publish=true ; shift 1 ;;
    -v) VERSION=$2 ; shift 2 ;;
    -c) COMMAND=$2 ; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -f apt-requirements.txt ] ; then
  if command -v sudo > /dev/null ; then
    sudo apt-get install $(grep -vE "^\s*#" apt-requirements.txt  | tr "\n" " ")
  else
    apt-get install $(grep -vE "^\s*#" apt-requirements.txt  | tr "\n" " ")
  fi
fi

if [ -f gspip-requirements.txt ] ; then
  if command -v gspip > /dev/null ; then
    gspip --upgrade install $(grep -vE "^\s*#" gspip-requirements.txt  | tr "\n" " ")
  else
   gspip --upgrade install $(grep -vE "^\s*#" gspip-requirements.txt  | tr "\n" " ")
  fi
fi

pip3 uninstall "$PACKAGE" -y
pip3 install setuptools
python$VERSION setup.py $COMMAND
if $publish && [ -f "$HOME/bin/gspip" ] ; then
  gspip push -s "  "
fi
if [ -d "dist" ] && [ "$COMMAND" != "sdist" ] ; then rm -r dist ; fi
if [ -d "build" ] ; then rm -r build ; fi
if ls "$PACKAGE".egg-info* &> /dev/null ; then rm -r "$PACKAGE".egg-info* ; fi

pdoc --force --html --config show_source_code=False --output-dir docs nmtf
