CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

test: macEyeDetection.cpp
	g++ $(CFLAGS) $(LIBS) -o test macEyeDetection.cpp
# CFLAGS = `pkg-config --cflags opencv`
# LIBS = `pkg-config --libs opencv`

# eyedetection: eyedetection.cpp
# 	g++ -v $(CFLAGS) -o eyedetection eyedetection.cpp

# # eyedetection: eyedetection.o
# 	# ld -v -demangle -static -arch x86_64 -macosx_version_min 10.10.0 -o eyedetection.o OpenCVBuild/lib/libopencv_calib3d.a OpenCVBuild/lib/libopencv_contrib.a OpenCVBuild/lib/libopencv_core.a OpenCVBuild/lib/libopencv_features2d.a OpenCVBuild/lib/libopencv_flann.a OpenCVBuild/lib/libopencv_gpu.a OpenCVBuild/lib/libopencv_highgui.a OpenCVBuild/lib/libopencv_imgproc.a OpenCVBuild/lib/libopencv_legacy.a OpenCVBuild/lib/libopencv_ml.a OpenCVBuild/lib/libopencv_nonfree.a OpenCVBuild/lib/libopencv_objdetect.a OpenCVBuild/lib/libopencv_ocl.a OpenCVBuild/lib/libopencv_photo.a OpenCVBuild/lib/libopencv_stitching.a OpenCVBuild/lib/libopencv_superres.a OpenCVBuild/lib/libopencv_ts.a OpenCVBuild/lib/libopencv_video.a OpenCVBuild/lib/libopencv_videostab.a 