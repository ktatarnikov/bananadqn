#!/usr/bin/env bash

set -e

wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip
wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip VisualBanana_Linux.zip -d pixels
unzip Banana_Linux.zip
