#!/usr/bin/env bash

scp -P 5664 bluehack@magi:/dataset/deepfashion/frozen_inference_graph.pb .
scp -P 5664 bluehack@magi:/dataset/deepfashion/data/label_map.pbtxt .
