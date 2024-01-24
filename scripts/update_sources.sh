#!/bin/bash

rm -r ./outputs/tmp/source/src
rm -r ./outputs/tmp/source/configs

cp -r src ./outputs/tmp/source/src
cp -r configs ./outputs/tmp/source/configs

kaggle datasets version -p ./outputs/tmp/source -m "Update" --dir-mode "zip"