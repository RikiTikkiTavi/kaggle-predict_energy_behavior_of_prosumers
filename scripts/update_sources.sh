#!/bin/bash

cp src ./outputs/tmp/source/src
cp configs ./outputs/tmp/source/configs

kaggle datasets version -p ./outputs/tmp/source -m "Update"