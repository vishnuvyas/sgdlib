SGDLIB
======
[![Build Status](https://travis-ci.org/vishnuvyas/sgdlib.svg?branch=master)](https://travis-ci.org/vishnuvyas/sgdlib)

An implementation of stochastic gradient descent in java. Also includes implementation of a linear svm with
L2-regularized loss. (and also scala test specs to see how this is used). This implementation tries to have
as little runtime dependencies as possible.


Building
========

Building requires sbt (http://www.scala-sbt.org/) and java version >= 1.6. To create the jar simply type
````
sbt package
````

That should create a jar that can be included in your own projects.


License
=======
This project is under GPL v3 (https://www.gnu.org/copyleft/gpl.html)


