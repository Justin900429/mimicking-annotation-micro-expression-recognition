#! /usr/bin/env bash

# Download the file with gdown
# `gdown` is included in requirements.txt and has no need to download it again
echo "Decompressed the five-class CASME II weights"
gdown 1w20GqDhBnuflU_xF9Oyi5FDoOafDaBdl -O weight/five_casme_weight.tar.gz
tar zxf weight/five_casme_weight.tar.gz -C weight

echo "Decompressed the five-class SAMM weights"
gdown 184jlfO6bATZ6OGV5Ns-oZpXDeBEvhQtI -O weight/five_samm_weight.tar.gz
tar zxf weight/five_samm_weight.tar.gz -C weight

echo "Decompressed the three-class CASME II weights"
gdown 1QewnYwmiy8v5Ev6mA46pRQugZrSyBC3y -O weight/three_casme_weight.tar.gz
tar zxf weight/three_casme_weight.tar.gz -C weight

echo "Decompressed the three-class SAMM weights"
gdown 1vS_88Z5dYsIe3Zu_NgN9mY3AruTHr248 -O weight/three_samm_weight.tar.gz
tar zxf weight/three_samm_weight.tar.gz -C weight
