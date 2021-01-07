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
    if ! sudo apt-get install -y $(grep -vE "^\s*#" apt-requirements.txt  | tr "\n" " ") ; then exit 1 ; fi
  else
    if ! apt-get install -y $(grep -vE "^\s*#" apt-requirements.txt  | tr "\n" " ") ; then exit 1 ; fi
  fi
fi

if [ -f gspip-requirements.txt ] ; then
  if command -v gspip > /dev/null ; then
    if ! gspip --upgrade install $(grep -vE "^\s*#" gspip-requirements.txt  | tr "\n" " ") ; then exit 1 ; fi
  else
    if ! gspip --upgrade install $(grep -vE "^\s*#" gspip-requirements.txt  | tr "\n" " ") ; then exit 1 ; fi
  fi
fi

pip3 uninstall "$PACKAGE" -y
pip3 install setuptools
if ! python$VERSION setup.py $COMMAND ; then exit 1 ; fi
if $publish && [ -f "$HOME/bin/gspip" ] ; then
  gspip push -s "  "
fi
if [ -d "dist" ] && [[ "$COMMAND" == "install" ]] ; then rm -r dist ; fi
if [ -d "build" ] && [[ "$COMMAND" == "install" ]] ; then rm -r build ; fi
if ls "$PACKAGE".egg-info* &> /dev/null && [[ "$COMMAND" == "install" ]] ; then rm -r "$PACKAGE".egg-info* ; fi

if [ "$COMMAND" == "install" ] ; then
  pdoc --force --html --config show_source_code=False --output-dir docs nmtf
fi
