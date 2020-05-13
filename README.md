# Webcam Eye Tracker
This is some fairly old code that I found on a USB and as such I haven't included a requirements.txt file, but the setup should be easy as it is.

A webcam eye tracker, a work in progress. Intended to perform comparably to more expensive hardware/software for a mere fraction of the price. This began as an experimental project that was to be used during my grad research, however at some point other endeavors took priority. There is much improvement to be done here, first some kind of smoothing/estimation algorithm should be put to work so that the iris detection isn't so jittery/inconsistent, I know that the Kalman filter would be a good candidate. Next, the pupul should be extracted from the iris and drawn as a point. Finally, the pupil location should be mapped to (x,y) coordinates on a plane - ie: the computer screen.



