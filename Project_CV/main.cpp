#include <iostream>
#include "stdio.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

CvHaarClassifierCascade *cascade_f;
CvHaarClassifierCascade *cascade_m;
CvMemStorage   *storage;

int detectFaces(IplImage *img, CvRect *face);
int detectFaceFeatures(IplImage *img, CvRect *face);
int calcHist(int number, string face_mouth);
int rgbHistComp(string face_mouth);

string fd = "img/"; //Folder for inputs & outputs
int filenumber = 1; // Number of file
string filename; char *fn; stringstream ssfn; char k;

int main(int argc, char** argv)
{
	IplImage *img;
	const char *file1 = "C:/opencv248/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	const char *file2 = "C:/opencv248/sources/data/haarcascades/haarcascade_mcs_mouth.xml";

	cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);
	cascade_m = (CvHaarClassifierCascade*)cvLoad(file2, 0, 0, 0);

	printf("Menu:\n");
	printf("ESC for EXIT.\nSPACE for NEXT IMG.\nF for face histograms.\nM for mouth histograms.\nH for compare histograms.\n\n");

start:
	/* setup memory storage, needed by the object detector */
	storage = cvCreateMemStorage(0);

	/* load image */
	ssfn.str("");
	filename = "";
	ssfn << fd <<filenumber << ".jpg";
	filename = ssfn.str();

	fn = new char[filename.size() + 1];
	fn[filename.size()] = 0;
	memcpy(fn, filename.c_str(), filename.size());

	img = cvLoadImage(fn, CV_LOAD_IMAGE_COLOR);

	assert(cascade_f && cascade_m && storage); //Check for null.

	const char* name = "Features-Detection";
	cvNamedWindow(name, 1);

	Mat frame = img;
	CvRect face;
	if (detectFaces(img, &face)) {
		if (detectFaceFeatures(img, &face)) {
			cout << "Features detected" << endl;
		}
	}
	cvShowImage(name, img);

label:
	k = cvWaitKey(0);
	if (k == 27){ //ESC
		cvDestroyWindow(name);
		cvReleaseImage(&img);
		cvClearMemStorage(storage);
		goto stop;
	}
	if (k == 32){ //Space
		filenumber++;
		if (filenumber < 7){
			cvDestroyWindow(name);
			cvReleaseImage(&img);
			cvClearMemStorage(storage);
			goto start;
		}
		else{ printf("No more images.\n"); goto label; }
	}
	if (k == 102){ //f for face histograms
		for (int j = 1; j < filenumber + 1; j++){
			calcHist(j, "face");
		}
		printf("Histograms for Faces DONE!\n");
		goto label;
	}
	if (k == 109){ //m for mouth histograms
		for (int j = 1; j < filenumber + 1; j++){
			calcHist(j, "mouth");
		}
		printf("Histograms for Mouths DONE!\n");
		goto label;
	}
	if (k == 104){ //h for compare histograms
		rgbHistComp("face");
		printf("Results in res_Face.txt File!\n");
		rgbHistComp("mouth");
		printf("Results in res_Mouth.txt File!\n");
		goto label;
	}
	else{ goto label; }
stop:
	return 0;
}

int rgbHistComp(string face_mouth){

	FILE * fFile, *mFile;
	fFile = fopen("img/res_Face.txt", "a+");
	mFile = fopen("img/res_Mouth.txt", "a+");
	Mat src_base;
	Mat src_test1, src_test2, src_test3, src_test4, src_test5;

	if (face_mouth == "face"){

		src_base = imread("img/face6.jpg", 1);
		src_test1 = imread("img/face1.jpg", 1);
		src_test2 = imread("img/face2.jpg", 1);
		src_test3 = imread("img/face3.jpg", 1);
		src_test4 = imread("img/face4.jpg", 1);
		src_test5 = imread("img/face5.jpg", 1);
	}
	else if (face_mouth == "mouth"){

		src_base = imread("img/mouth6.jpg", 1);
		src_test1 = imread("img/mouth1.jpg", 1);
		src_test2 = imread("img/mouth2.jpg", 1);
		src_test3 = imread("img/mouth3.jpg", 1);
		src_test4 = imread("img/mouth4.jpg", 1);
		src_test5 = imread("img/mouth5.jpg", 1);
	}
	else{ return -1; }

	/// Establish the number of bins
	int histSize[] = { 30, 32 };

	/// Set the ranges ( for B,G,R) )
	float hranges[] = { 0, 180 }; //For the i-th channel there are 30 bins in a range from 0 to 180, therefore 6 levels of intensity for each bin
	float sranges[] = { 0, 256 }; //For the j-th channel there are 32 bins in a range from 0 to 255, therefore 8 levels of intensity for each bin

	const float* ranges[] = { hranges, sranges };

	bool uniform = true; bool accumulate = false;
	MatND currentHistogram, currentHistogram1, currentHistogram2, currentHistogram3, currentHistogram4, currentHistogram5;
	int channels[] = { 0, 1, 2 };

	/// Compute the histograms:
	calcHist(&src_base, 1, channels, Mat(), currentHistogram, 2, histSize, ranges, uniform, accumulate);
	calcHist(&src_test1, 1, channels, Mat(), currentHistogram1, 2, histSize, ranges, uniform, accumulate);
	calcHist(&src_test2, 1, channels, Mat(), currentHistogram2, 2, histSize, ranges, uniform, accumulate);
	calcHist(&src_test3, 1, channels, Mat(), currentHistogram3, 2, histSize, ranges, uniform, accumulate);
	calcHist(&src_test4, 1, channels, Mat(), currentHistogram4, 2, histSize, ranges, uniform, accumulate);
	calcHist(&src_test5, 1, channels, Mat(), currentHistogram5, 2, histSize, ranges, uniform, accumulate);

	/// Apply the histogram comparison methods
	for (int i = 0; i < 4; i++)
	{
		int compare_method = i;
		double base_base = compareHist(currentHistogram, currentHistogram, compare_method);
		double base_test1 = compareHist(currentHistogram, currentHistogram1, compare_method);
		double base_test2 = compareHist(currentHistogram, currentHistogram2, compare_method);
		double base_test3 = compareHist(currentHistogram, currentHistogram3, compare_method);
		double base_test4 = compareHist(currentHistogram, currentHistogram4, compare_method);
		double base_test5 = compareHist(currentHistogram, currentHistogram5, compare_method);

		if (face_mouth == "face"){
			fprintf(fFile, " Method [%d] Perfect, Base-Test(1), Base-Test(2), Base-Test(3), Base-Test(4), Base-Test(5) :\n \t %f, \t%f,   %f, \t  %f, \t%f, \t%f \n", i, base_base, base_test1, base_test2, base_test3, base_test4, base_test5);
		}
		else if (face_mouth == "mouth"){
			fprintf(mFile, " Method [%d] Perfect, Base-Test(1), Base-Test(2), Base-Test(3), Base-Test(4), Base-Test(5) :\n \t %f, \t%f,   %f, \t  %f, \t%f, \t%f \n", i, base_base, base_test1, base_test2, base_test3, base_test4, base_test5);
		}
		else{ return -1; }
	}

	fclose(fFile);
	fclose(mFile);
	return 1;
}

int calcHist(int number, string face_mouth){

	ssfn.str("");
	filename = "";
	ssfn << fd << face_mouth << number << ".jpg";
	filename = ssfn.str();

	fn = new char[filename.size() + 1];
	fn[filename.size()] = 0;
	memcpy(fn, filename.c_str(), filename.size());

	Mat src = imread(fn, CV_LOAD_IMAGE_COLOR);
	if (!src.data)
	{
		return -1;
	}

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	ssfn.str("");
	filename = "";
	ssfn << fd << "histogram_" << face_mouth << number << ".jpg";
	filename = ssfn.str();

	fn = new char[filename.size() + 1];
	fn[filename.size()] = 0;
	memcpy(fn, filename.c_str(), filename.size());

	/// Display
	namedWindow(fn, WINDOW_AUTOSIZE);
	imwrite(fn, histImage);
	imshow(fn, histImage);
	return 1;
}

int detectFaces(IplImage *img, CvRect *face) {
	/* detect faces */
	CvSeq *faces = cvHaarDetectObjects(
		img, cascade_f, storage,
		1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50));
	CvRect face1;

	//if (faces->total == 0) {
	//return 0;
	//}
	CvRect* _face = (CvRect*)cvGetSeqElem(faces, 0);
	face->x = _face->x;
	face->y = _face->y;
	face->width = _face->width;
	face->height = _face->height;

	face1.x = face->x;
	face1.y = face->y;
	face1.width = face->width;
	face1.height = face->height;

	Mat crop(img);
	crop = crop(face1);

	ssfn.str("");
	filename = "";
	ssfn << fd << "face" << filenumber << ".jpg";
	filename = ssfn.str();

	fn = new char[filename.size() + 1];
	fn[filename.size()] = 0;
	memcpy(fn, filename.c_str(), filename.size());

	imwrite(fn, crop);
	//namedWindow(fn, CV_WINDOW_NORMAL && CV_WINDOW_AUTOSIZE);
	imshow(fn, crop);

	return 1;
}

int detectFaceFeatures(IplImage *img, CvRect *face)
{
	int i;
	bool hasMouth = false;
	cvRectangle(img,
		cvPoint(face->x, face->y),
		cvPoint(face->x + face->width, face->y + face->height),
		CV_RGB(255, 0, 0), 1, 8, 0);

	CvRect mouthROI = cvRect(face->x, face->y + (face->height / 1.5), face->width, face->height / 2.5);
	CvRect mouth1;

	CvRect *r;

	/* detect Mouth */
	cvSetImageROI(img, mouthROI);
	CvSeq* mouths = cvHaarDetectObjects(
		img, cascade_m, storage,
		1.1, 3, 0, Size(30, 30));
	cvResetImageROI(img);

	/* draw a rectangle for each mouth found */
	for (i = 0; i < (mouths ? mouths->total : 0); i++) {
		int margin_left = 10;
		int margin_right = 0;
		r = (CvRect*)cvGetSeqElem(mouths, i);
		int x1 = r->x + mouthROI.x;
		int y1 = r->y + mouthROI.y;
		int x2 = x1 + r->width;
		int y2 = y1 + r->height;
		int x1c = x1 + margin_left;
		int y1c = (y1 + y2) / 2;
		int x2c = x2 - margin_right;
		int y2c = (y1 + y2) / 2 - 5;

		mouth1.x = x1;
		mouth1.y = y1;
		mouth1.width = r->width;
		mouth1.height = (r->height) - 26;

		cvRectangle(img,
			cvPoint(x1, y1),
			cvPoint(x2, y2),
			CV_RGB(255, 255, 255), 1, 8, 0);
		hasMouth = true;

		Mat crop(img);
		crop = crop(mouth1);

		ssfn.str("");
		filename = "";
		ssfn << fd << "mouth" << filenumber << ".jpg";
		filename = ssfn.str();

		fn = new char[filename.size() + 1];
		fn[filename.size()] = 0;
		memcpy(fn, filename.c_str(), filename.size());

		imwrite(fn, crop);
		//namedWindow(fn, CV_WINDOW_NORMAL && CV_WINDOW_AUTOSIZE);
		imshow(fn, crop);
	}
	return (hasMouth) ? 1 : 0;
}
