################################################################################
# Copyright 2020 - NVIDIA Corporation
# SPDX-License-Identifier: MIT
################################################################################

CXX=g++

APP:= deepstream-pose-estimation-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= deepstream_pose_estimation_app.cpp munkres_algorithm.cpp post_process.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(patsubst %.c,%.o, $(patsubst %.cpp,%.o, $(SRCS)))

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/opt/nvidia/deepstream/deepstream/sources/apps/apps-common/includes -I/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_utils -lm \
        -lpthread -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)


