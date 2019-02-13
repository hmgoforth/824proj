# 824proj

2/12 meeting:

Jen: Wrap face GAN module: (image and gaussian/openpose poses in, generated face and face mask out)

Hunter: Wrap pose-warp GAN into function: (image and gaussian poses in, generated output out)

Kevin: Wrap function for OpenPose outputs: (image in, gaussians out)

It looks like you have to turn on Python (off by default)
cmake .. -DBUILD_PYTHON=ON

or with python2

cmake .. -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python2.7 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7m.so
