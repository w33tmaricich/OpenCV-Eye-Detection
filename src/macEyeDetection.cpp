//// eyes.cpp : Defines the entry point for the console application.
////
//
//#include "stdafx.h"
//
//
//
//int _tmain(int argc, _TCHAR* argv[])
//{
//	return 0;
//}
//

/*
	usage: eyedetection [--help] [--display] [-f <image-path>] [-o <path-with-name>]

	flags:
		--multi-image	Every trailing perameter is a path to an image to find eye centers.
	    --file, -f		Path to image you want to run the engine on.
	    --output, -o	Location/name of output file. If not specified, the file
						will save in the same directory as the script in 'out.txt'.
	    --display, -d	Show a graphical representation of what was run.
	    --help			Display this help menu

	The MIT License (MIT) :: Copyright (c) 2014 Alexander Maricich
*/

/*
	file output format
	==================
	count_value\n					<-- this is the number of registration points that the engine found, in this case: 2[cr]
	x_value\ty_value\tpoint_name\n	<-- example: 35[tab]156[tab]Left Eye[cr]
	x_value\ty_value\tpoint_name\n	<-- example  145[tab]161[tab]Right Eye[cr]
 */
#include <opencv2/opencv.hpp>

// This was an attempt to make things universal. Maybe i should come back to this later $$$
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "StdAfx.h"
#endif

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
	
using namespace std;
using namespace cv;

// image matrixes of bits
Mat img;
Mat imgCopy;

// paths to the files we are using
// ENSURE DOUBLE SLASHES FOR WINDOWS
string imgPath;
char* fileName = (char *)"\\XLoupeEyesOut.txt";

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
char* tempPath = getenv("Temp");
#else
char* tempPath = getenv("TMPDIR");
#endif

//int outputPathCharLength = (int)strlen(tempPath) + (int)strlen(fileName)+1;
char* outputPathChar = (char*)malloc(strlen(tempPath)+strlen(fileName)+1);
string outputPath = string(outputPathChar);
string xmlLeft = "haarcascade_mcs_lefteye.xml";
string xmlRight = "haarcascade_mcs_righteye.xml";
string xmlPairSmall = "haarcascade_mcs_eyepair_small.xml";
string xmlPairBig = "haarcascade_mcs_eyepair_big.xml";
string outputString = "";

// Output file
ofstream outputFile;
bool fileExists;

// string converter
ostringstream convert;

// classifier objects to find matches
CascadeClassifier haar_left;
CascadeClassifier haar_right;
CascadeClassifier haar_pair;

// vectors that hold the matches we find
vector< Rect_<int> > lefts;
vector< Rect_<int> > rights;
vector< Rect_<int> > pairs;

int maxint = numeric_limits<int>::max();

// index number of above vectors that we will be using
int leftNumber = maxint;
int rightNumber = maxint;
int pairNumber = maxint;
int pointCount = 0;

// If 1, we will display the output to the screen in a graphical form
int eyePairsFound = 0;
int showImage = 0;
int imgFound = 0;
int flagsFound = 0;
int multiImage = 0;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define CV_RGB( r, g, b )  (int)((uchar)(b) + ((uchar)(g) << 8) + ((uchar)(r) << 16))
#endif

/**
 * help - displays help screen for the user
 */
void help() {
	cout << "usage: eyedetection [--multi-image <image-path> <image-path> ...][--help]\n\t\t    [--display] [-f <image-path>] [-o <path-with-name>]"<< endl << endl
		 << "flags:" << endl
		 << "    --multi-image\tEvery trailing perameter is a path to an image to find\n\t\t\teye centers." << endl
		 << "    --file, -f\t\tPath to image you want to run the engine on." << endl
		 << "    --output, -o\tLocation/name of output file. If not specified, the file\n\t\t\twill save in the same directory as the script\n\t\t\tin \'out.txt\'." << endl
		 << "    --display, -d\tShow a graphical representation of what was run." << endl
		 << "    --help\t\tDisplay this help menu" << endl
		 << endl << "The MIT License (MIT) :: Copyright (c) 2014 Alexander Maricich" << endl;
}

/**
 * exists - Checks the file system to see if the ile exists.
 * @param  name String path to the file
 * @return      Completion: True or False
 */
bool exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

/**
 * parseParameters - Make some sense out of what the user passes in.
 * @param  argc       Number of arguments passed.
 * @param  argv       Arguments in an array.
 * @param  showImage  Flag for if a graphical representation of the eye finding should be displayed.
 * @param  imgFound   Number of images found.
 * @param  flagsFound Number of flags found.
 * @param  multiImage Flag for if multiple images are to be processed.
 * @param  imgPath    Path to one image at any given time.
 * @param  outputPath Path to the file that data will be written to.
 * @return            bool on correct termination.
 */
bool parseParameters(int argc, char const *argv[]) {
	// loop through the input to see what needs to be done
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[1], "--help") == 0) {
			help();
			// cout << "Help called" << endl;
			exit(1);
			return false;
		}
		// change the display flag so it will be run.
		else if (strcmp(argv[i], "--display") == 0 || strcmp(argv[i], "-d") == 0) {
			showImage = 1;
			flagsFound++;
			//cout << "Display enabled: showImage=" << showImage << endl;
		}
		else if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0) {
			imgPath = argv[i+1];
			imgFound++;
			flagsFound++;
			//cout << "Passing one file: imgPath=" << imgPath << " imgFound=" << imgFound << endl;
		}
		else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
			outputPath = argv[i+1];
			flagsFound++;
			//cout << "Changing output file: outputPath=" << outputPath << endl;
		}
		else if (strcmp(argv[i], "--multi-image") == 0) {
			multiImage = i+1;
		}
		//cout << "flagsFound=" << flagsFound << endl;
	}
	return true;
}

void findEyeCenters(int argc, char const *argv[]) {

	// if there is only one image being passed in, we only want to run the loop once
	if (multiImage == 0 && argc == 2) {
		multiImage = 1;
	}

	int processedCounter = 0;
	cout << "Processing  " << argc - multiImage << " images: \n--------------" << endl;
	for (int l = multiImage; l < argc; l++) {
		processedCounter++;

		if (imgFound || multiImage != 0) {
			imgPath = argv[l];
			cout << "[" << processedCounter << "] Processing image at path: " << imgPath << endl;
			// check to see if the image exsists
			fileExists = (ifstream(imgPath.c_str()) != 0);


			if (fileExists) {

				// read in the image
				img = imread(imgPath, 0);
				// clone the image
				imgCopy = img.clone();

				// find everything you need
				haar_left.load(xmlLeft);
				haar_right.load(xmlRight);
				haar_pair.load(xmlPairSmall);

				haar_left.detectMultiScale(imgCopy, lefts);
				haar_right.detectMultiScale(imgCopy, rights);
				haar_pair.detectMultiScale(imgCopy, pairs);

				// loop through each pair
				
				if (int(pairs.size() != 0)) {
					eyePairsFound = int(pairs.size());

					for (unsigned int k = 0; k < pairs.size(); k++) {
						int useLeft = -1;
						int useRight = -1;
						int small_left_diff = numeric_limits<int>::max();
						int small_right_diff = numeric_limits<int>::max();
						Rect pair = pairs[k];

						// find the left eye that matches the closest to the pair
						if (int(lefts.size() != 0)){
							for (unsigned int i = 0; i < lefts.size(); i++) {
								Rect left = lefts[i];
								int lxdiff;
								int lydiff;
								int ldiff;
		
								// find how close the left eye is to the left corner of the pair
								if(left.x > pair.x) {
									lxdiff = left.x - pair.x;
								}
								else {
									lxdiff = pair.x - left.x;
								}
		
								if(left.y > pair.y) {
									lydiff = left.y - pair.y;
								}
								else {
									lydiff = pair.y - left.y;
								}
		
								// find the total number of pixels away it is
								ldiff = lxdiff + lydiff;
		
								if(small_left_diff > ldiff) {
									small_left_diff = ldiff;
									useLeft = i;
								}
							}
						}

						leftNumber = useLeft;

						// find the right eye that matches the closest to the pair
						if (int(rights.size()) != 0) {
							for (unsigned int j = 0; j < rights.size(); j++) {
								Rect right = rights[j];
								int rxdiff;
								int rydiff;
								int rdiff;
								
								// find how close the right eye is to the right corner of the pair
								if((right.x+right.width) > (pair.x+pair.width)) {
									rxdiff = (right.x+right.width) - (pair.x+pair.width);
								}
								else {
									rxdiff = (pair.x+pair.width) - (right.x+right.width);
								}
		
								if(right.y > pair.y) {
									rydiff = right.y - pair.y;
								}
								else {
									rydiff = pair.y - right.y;
								}
		
								// find the total number of pixels away it is
								rdiff = rxdiff + rydiff;
		
								if(small_right_diff > rdiff) {
									small_right_diff = rdiff;
									useRight = j;
								}
							}	
						}

						rightNumber = useRight;

						// some check to see if this is the closest match $$$
						pairNumber = k;	
					}
				}

				// only run the output / display code if something can be displayed
				if (eyePairsFound) {
					// save the pair we are using
					Rect usedPair = pairs[pairNumber];
					// create ints for the data we have a potential to pass
					int left_eye_x_pt = -1, left_eye_y_pt = -1, right_eye_x_pt = -1, right_eye_y_pt = -1;
					// create rectangle objects for drawing if needed
					Rect usedLeft, usedRight;

					// if a left eye was found
					if (leftNumber != maxint) {
						pointCount++;
						// get the rectangle we are using
						usedLeft = lefts[leftNumber];
						// calcualte the x and y coordinates
						left_eye_x_pt = usedLeft.x + (usedLeft.width/2);
						left_eye_y_pt = usedLeft.y + (usedLeft.height/2);
					}
					if (rightNumber != maxint) {
						pointCount++;
						usedRight = rights[rightNumber];
						right_eye_x_pt = (usedRight.x) + (usedRight.width/2);
						right_eye_y_pt = usedRight.y + (usedRight.height/2);
					}

					// display the image to the screen
					if (showImage) {
						// draw the pair of eyes finding 
						rectangle(imgCopy, usedPair, CV_RGB(0, 255, 0), 1);

						// if a left eye match was found
						if (leftNumber != maxint) {
							// draw the eye match
							rectangle(imgCopy, usedLeft, CV_RGB(0, 255, 0), 1);
							// create a point telling where the circle is going to go
							Point lefteye = Point(left_eye_x_pt, left_eye_y_pt);
							// draw the circle
							circle(imgCopy, lefteye, 10, Scalar(0, 0, 255), -1, 8);

						}
						if (rightNumber != maxint) {
							rectangle(imgCopy, usedRight, CV_RGB(0, 255, 0), 1);
							Point righteye = Point(right_eye_x_pt, right_eye_y_pt);
							circle(imgCopy, righteye, 10, Scalar(0, 0, 255), -1, 8);
							
						}


						while(true) {
							// show the window with drawings
							imshow(argv[0], imgCopy);
							// hold the window open until you hit escape
							char key = (char) waitKey(20);
							if(key == 27)
								break;
						}
					}
					// used to convert to string
					


					// Put the string together one bit at a time
					string strPoint;

					

					// if the left eye is found
					if (leftNumber != maxint) {
						// add the number to the convert object
						convert << left_eye_x_pt;
						// convert it to a string
						strPoint = convert.str();
						// append it to the output string
						outputString += strPoint + "\t";
						// cear the convert object of the number we just used
						convert.str("");
						convert.clear();

						convert << left_eye_y_pt;
						strPoint = convert.str();
						outputString += strPoint + "\t";
						convert.str("");
						convert.clear();
					}

					if (rightNumber != maxint) {
						convert << right_eye_x_pt;
						strPoint = convert.str();
						outputString += strPoint + "\t";
						convert.str("");
						convert.clear();

						convert << right_eye_y_pt;
						strPoint = convert.str();
						outputString += strPoint + "\n";
						convert.str("");
						convert.clear();
					}

					outputFile.open(outputPath.c_str());
					outputFile << outputString;
					outputFile.close();

				}
				else {
					outputString += "0\t0\t0\t0\n";
				}
			}
			else {
				outputString += "0\t0\t0\t0\n";
			}
		}
		else {
			cout << "Error: No file was passed\n\tPlease insure to pass the data as described below" << endl;
			help();
			exit(1);
		}
	
	}
	// string strNumberOfPoints;
	string finalString;
	// convert << pointCount/2;
	// strNumberOfPoints = convert.str();
	// convert.str("");
	// convert.clear();

	// outputString += strNumberOfPoints + "\n";
	// finalString = strNumberOfPoints + "\n" + outputString + ":ok";
	finalString = outputString + ":ok";
	// finalString = ":ok";

	cout << endl << "Output data:\n--------------\n" << finalString << endl << endl;
	cout << "Outputing data to file: ";
	outputFile.open(outputPath.c_str());
	outputFile << finalString;
	outputFile.close();
	cout << "Complete." << endl;
}

/**
 * main - where the application launches at startup
 * @param  argc the total number of arguments
 * @param  argv each argument passed in as a string
 * @return      0 on successful completion
 */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
int _tmain(int argc, _TCHAR* argv[]) 
#else
int main(int argc, char const *argv[]) 
#endif
{
	strcpy(outputPathChar, tempPath);
	strcat(outputPathChar, fileName);
	
	bool hasCorrectInput = parseParameters(argc, argv);
	
	if (hasCorrectInput) {
		if (!multiImage) {
			// use the second perameter as the file if nothing else was provided
			if(!imgFound && argc == 2) {
				imgFound++;
				imgPath = argv[1];
			}
		}
		findEyeCenters(argc, argv);
	}

	return 0;
}