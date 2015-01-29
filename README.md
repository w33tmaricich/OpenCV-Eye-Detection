OpenCV Eye Detection Implementation
===================================

About
-----
This project is a simple implementation of the OpenCV framework. OpenCV provides simple methods to manipulate different types of media. This program should be passed an image or list of images, each with a single face. It will then attempt to find the eye centers of the face within the images passed. There weren't many examples that I could find of Haar Cascades within OpenCV. Because of this, I am releasing this under the MIT licence for anyone who would like a working example.

Compiling it yourself
---------------------
If you wish to compile it yourself, you must first install OpenCV on your system. Here is a link to the OpenCV website if you wish to download it and install it yourself. http://opencv.org/

If you are on a mac and have Homebrew installed, getting OpenCV working is extremely simple. Just open up a terminal and run the following brew commands:
```
brew tap homebrew/science
brew install opencv
```
If this fails, you may have ffmpeg installed which tends to conflict. If so, try this:
```
brew install opencv --env=std
```

After all that is installed, a simple GCC should do the trick!

If you are compiling on windows, good luck. Ill update this readme at a later time on the easiest way to get it compiled in Visual Studio. 

Flags and features
------------------
```
usage: eyedetection [--help] [--display] [-f <image-path>] [-o <path-with-name>]
flags:
        --multi-image   Every trailing perameter is a path to an image to find eye centers.
        --file, -f      Path to image you want to run the engine on.
        --output, -o    Location/name of output file. If not specified, the file
                        will save in the same directory as the script in 'out.txt'.
        --display, -d   Show a graphical representation of what was run. This only
                        works when running without the --multi-image flag.
        --help          Display this help menu
```

The MIT License (MIT) :: Copyright (c) 2014 Alexander Maricich