# ION GNSS+ 2024 Factor Graph Tutorial
<br>
<br>

This repository contains a set of GNSS specific factor graph examples implemented in python using the GTSAM library.

<br>

### What is Georgia Tech Smoothing and Mapping (GTSAM):
 
- GTSAM is a C++ library (with Python and MATLAB bindings) for factor graphs
- https://github.com/borglab/gtsam
- "BSD-licensed C++ library that implements sensor fusion for robotics and computer vision applications, including SLAM (Simultaneous Localization and Mapping), VO (Visual Odometry), and SFM (Structure from Motion)."
- Developers are actively maintaining and developing the code base.
- Frank Dellaert and Michael Kaess. ``Factor graphs for robot perception.'' Foundations and Trends® in Robotics 6.1-2 (2017): 1-139.

<br>

## Examples:
---

- Example 1: 
    - Single epoch ( static GNSS localization with pseudorange data )
    - batch estimation
- Example 2/3:
    - Incorporating dynamics ( multi-epoch GNSS localization with pseudorange data )
    - batch estimation
- Example 4:
    - Incorporating robust estimation techniques
    - batch estimation
    - additional background on robust estimation in `robust_est_overview.pdf`
- Example 5:
    - Moving from batch to incremental estimation ( using ISAM2 )

<br>

## Running the software:
---

<br>

```bash
$ mkdir path/to/where/you/want/plots/saved
$ docker build -t ion_gnss_2024 .
$ docker run -v path/to/where/you/want/plots/saved:/ion_gnss_2024/plots -it ion_gnss_2024
```

<br>

after docker run, you should see all the examples listed above in the examples directory:

```
➜  /ion_gnss_2024 ls -l examples
total 88
drwxr-xr-x 2 root root  4096 Sep 15 06:04 __pycache__
-rw-r--r-- 1 root root 15920 Sep 16 18:46 example_1.py
-rw-r--r-- 1 root root 12705 Sep 16 06:55 example_2.py
-rw-r--r-- 1 root root 13237 Sep 16 06:55 example_3.py
-rw-r--r-- 1 root root 13742 Sep 16 06:54 example_4.py
-rw-r--r-- 1 root root 12365 Sep 16 16:36 example_5.py
drwxr-xr-x 3 root root  4096 Sep 16 18:46 helpers
```