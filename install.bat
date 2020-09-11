python setup.py bdist_wininst
IF exist dist\ ( rmdir dist )
IF exist build\ ( rmdir build )
pdoc --force --html --config show_source_code=False --output-dir docs nmtf
