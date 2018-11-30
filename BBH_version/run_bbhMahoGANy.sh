#!/bin/bash

# Copyright (C) 2018  Hunter Gabbard, Chris Messenger, Michael Williams, Jordan McGinn
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# This script is for running bbhMahoGANy.py

###############
# Directories #
###############

# Output directory which will contain all outputs
outdir="/data/public_html/2136420/GenNet"
# Directory with templates
template_dir="templates/"

########
# CUDA #
########

# Cuda device to use
cuda_device="0"

./bbhMahoGAN.py -outdir=$outdir -templatedir=$template_dir

