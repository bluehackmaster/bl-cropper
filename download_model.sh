#!/usr/bin/env bash

mkdir -p /dataset/deepfashion/fig-$1
scp -i ~/bluehack.pem ubuntu@13.124.221.27:/dataset/deepfashion/fig-$1/frozen_inference_graph.pb ./model/
scp -i ~/bluehack.pem ubuntu@13.124.221.27:/dataset/deepfashion/data/label_map.pbtxt ./model/

#scp -i ~/bluehack.pem ubuntu@13.124.221.27:/dataset/deepfashion/fig-644228.tar.bz2 .
