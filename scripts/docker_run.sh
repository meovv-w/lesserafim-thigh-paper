#!/bin/bash
BLADE_DIR=/mnt/d/HERMES/study/lesserafim-thigh-paper

sg docker -c "docker run --gpus all --rm \
  -v $BLADE_DIR:/workspace \
  -w /workspace \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel \
  bash /workspace/scripts/docker_setup.sh" 2>&1
