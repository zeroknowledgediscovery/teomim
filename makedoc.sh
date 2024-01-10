#!/bin/bash

pdoc --html teomim -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/teomim/* docs
