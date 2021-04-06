// Copyright 2020 - NVIDIA Corporation
// SPDX-License-Identifier: MIT

/*
 *  Modified by MACNICA Inc.
 */ 

//#include "post_process.cpp"
#include "post_process.hpp"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>

#define EPS 1e-6

#define MAX_DISPLAY_LEN 64

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

#define OUTPUT_FILE "Pose_Estimation.mp4"

#define CAP_WIDTH 640
#define CAP_HEIGHT 480

#define DEBUG_PRINT(msg)  g_print("%s:%d [%s] %s\n", __FILE__, __LINE__, __func__, msg)

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

gint frame_number = 0;

static Vec2D<int> topology{
    {0, 1, 15, 13},
    {2, 3, 13, 11},
    {4, 5, 16, 14},
    {6, 7, 14, 12},
    {8, 9, 11, 12},
    {10, 11, 5, 7},
    {12, 13, 6, 8},
    {14, 15, 7, 9},
    {16, 17, 8, 10},
    {18, 19, 1, 2},
    {20, 21, 0, 1},
    {22, 23, 0, 2},
    {24, 25, 1, 3},
    {26, 27, 2, 4},
    {28, 29, 3, 5},
    {30, 31, 4, 6},
    {32, 33, 17, 0},
    {34, 35, 17, 5},
    {36, 37, 17, 6},
    {38, 39, 17, 11},
    {40, 41, 17, 12}};

/*Method to parse information returned from the model*/
std::tuple<Vec2D<int>, Vec3D<float>>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
  Vec1D<int> counts;
  Vec3D<int> peaks;

  float threshold = 0.1;
  int window_size = 5;
  int max_num_parts = 2;
  int num_integral_samples = 7;
  float link_threshold = 0.1;
  int max_num_objects = 100;

  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  void *paf_data = tensor_meta->out_buf_ptrs_host[1];
  NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;

  /* Finding peaks within a given window */
  find_peaks(counts, peaks, cmap_data, cmap_dims, threshold, window_size, max_num_parts);
  /* Non-Maximum Suppression */
  Vec3D<float> refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size);
  /* Create a Bipartite graph to assign detected body-parts to a unique person in the frame */
  Vec3D<float> score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples);
  /* Assign weights to all edges in the bipartite graph generated */
  Vec3D<int> connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts);
  /* Connecting all the Body Parts and Forming a Human Skeleton */
  Vec2D<int> objects = connect_parts(connections, topology, counts, max_num_objects);
  return {objects, refined_peaks};
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
  int K = topology.size();
  int count = objects.size();
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  for (auto &object : objects)
  {
    int C = object.size();
    for (int j = 0; j < C; j++)
    {
      int k = object[j];
      if (k >= 0)
      {
        auto &peak = normalized_peaks[j][k];
        int x = peak[1] * MUXER_OUTPUT_WIDTH;
        int y = peak[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
        cparams.xc = x;
        cparams.yc = y;
        cparams.radius = 8;
        cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
        cparams.has_bg_color = 1;
        cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_circles++;
      }
    }

    for (int k = 0; k < K; k++)
    {
      int c_a = topology[k][2];
      int c_b = topology[k][3];
      if (object[c_a] >= 0 && object[c_b] >= 0)
      {
        auto &peak0 = normalized_peaks[c_a][object[c_a]];
        auto &peak1 = normalized_peaks[c_b][object[c_b]];
        int x0 = peak0[1] * MUXER_OUTPUT_WIDTH;
        int y0 = peak0[0] * MUXER_OUTPUT_HEIGHT;
        int x1 = peak1[1] * MUXER_OUTPUT_WIDTH;
        int y1 = peak1[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
        lparams.x1 = x0;
        lparams.x2 = x1;
        lparams.y1 = y0;
        lparams.y2 = y1;
        lparams.line_width = 3;
        lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_lines++;
      }
    }
  }
}

/* pgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
         l_user = l_user->next)
    {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
      {
        NvDsInferTensorMeta *tensor_meta =
            (NvDsInferTensorMeta *)user_meta->user_meta_data;
        Vec2D<int> objects;
        Vec3D<float> normalized_peaks;
        tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
        create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
      }
    }

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
          Vec2D<int> objects;
          Vec3D<float> normalized_peaks;
          tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
          create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number =  %d", frame_number);
    //offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    txt_params->font_params.font_name = (char *)"Mono";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

gboolean
link_element_to_tee_src_pad(GstElement *tee, GstElement *sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(tee), "src_%u");
  tee_src_pad = gst_element_request_pad(tee, padtemplate, NULL, NULL);

  if (!tee_src_pad)
  {
    g_printerr("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad(sinkelem, "sink");
  if (!sinkpad)
  {
    g_printerr("Failed to get sink pad from '%s'",
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }

  if (gst_pad_link(tee_src_pad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link '%s' and '%s'", GST_ELEMENT_NAME(tee),
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }
  ret = TRUE;

done:
  if (tee_src_pad)
  {
    gst_object_unref(tee_src_pad);
  }
  if (sinkpad)
  {
    gst_object_unref(sinkpad);
  }
  return ret;
}

GstElement  *
make_element_and_link(
  const gchar *factoryname, const gchar *name, GstBin *bin, GstElement *src
)
{
  GstElement  *element;
  gboolean    flag;
    
  element = gst_element_factory_make(factoryname, name);
    
  flag = gst_bin_add(bin, element);
  g_assert(flag == TRUE);
    
  if (src != NULL) {
    flag = gst_element_link(src, element);
    g_assert(flag == TRUE);
  }
    
  return (element);
}

GstElement  *
make_caps_and_link(
  const gchar *caps_str, const gchar *name, GstBin *bin, GstElement *src
)
{
  GstElement  *element;
  GstCaps     *caps;
    
  caps = gst_caps_from_string(caps_str);
  element = make_element_and_link("capsfilter", name, bin, src);
  g_object_set(G_OBJECT(element), "caps", caps, NULL);
  gst_caps_unref(caps);
    
  return (element);
}

GstElement *
construct_camera_source_bin(
  GstBin *bin, const gchar *device, gint width, gint height
)
{
  GstElement *element = NULL;
  GstCaps *caps = NULL;
  gchar *caps_str = NULL;

  element = make_element_and_link("v4l2src", NULL, bin, element);
  g_object_set(G_OBJECT(element), "device", device, NULL);

  caps_str = g_strdup_printf("video/x-raw, width=%d, height=%d, format=YUY2", width, height);
  element = make_caps_and_link(caps_str, NULL, bin, element);
  g_free(caps_str);

  element = make_element_and_link("videoconvert", NULL, bin, element);

  element = make_caps_and_link("video/x-raw, format=NV12", NULL, bin, element);

  element = make_element_and_link("nvvideoconvert", NULL, bin, element);

  element = make_caps_and_link("video/x-raw(memory:NVMM), format=NV12", NULL, bin, element);

  return (element);
}

void
on_qtdemux_pad_added(GstElement *element, GstPad *pad, gpointer data)
{
  GstElement *queue = (GstElement *)data;
  GstPad *sink_pad = NULL;
  gchar *name = NULL;

  name = gst_pad_get_name(pad);
  g_print("Pad name: %s\n", name);
  if (strcmp(name, "video_0") != 0) {
    g_free(name);
    g_print("not video_0 pad\n");
    return;
  }
  g_free(name);

  sink_pad = gst_element_get_static_pad(queue, "sink");
  if (sink_pad == NULL) {
      g_printerr("sink pad error\n");
      return;
  }

  if (gst_pad_link(pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr("pad link error\n");
    return;
  }
  gst_object_unref(GST_OBJECT(sink_pad));
}

GstElement *
construct_file_source_bin(GstBin *bin, const gchar *file_path)
{
  GstElement *element = NULL;
  GstElement *qtdemux = NULL;
  guint pos;

  element = make_element_and_link("filesrc", NULL, bin, element);
  g_object_set(G_OBJECT(element), "location", file_path, NULL);

  pos = strlen(file_path) - 3;
  if (g_ascii_strncasecmp(file_path + pos, "mov", 80) == 0
      || g_ascii_strncasecmp(file_path + pos, "mp4", 80) == 0) {
    qtdemux = make_element_and_link("qtdemux", NULL, bin, element);
    element = make_element_and_link("queue", NULL, bin, NULL);
    g_signal_connect(
      qtdemux, "pad-added", G_CALLBACK(on_qtdemux_pad_added), element);
  }
  else {
    element = make_element_and_link("queue", NULL, bin, element);
  }

  element = make_element_and_link("h264parse", NULL, bin, element);

  element = make_element_and_link("nvv4l2decoder", NULL, bin, element);

  return (element);
}

GstElement *
construct_inference_bin(
  GstBin *bin, GstElement *source_elements[], gint num_sources, gboolean is_live,
  GstPad **p_pgie_src_pad, GstPad **p_osd_sink_pad
)
{
  GstElement *element = NULL;
  GstPadTemplate *pad_template = NULL;
  GstPad *sink_pad = NULL;
  GstPad *src_pad = NULL;
  gchar *str = NULL;

  element = make_element_and_link("nvstreammux", NULL, bin, element);
  g_object_set(G_OBJECT(element), 
    "width", 1920, 
    "height", 1080, 
    "batch-size", 1, 
    "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
    NULL
  );
  if (is_live) {
    g_object_set(G_OBJECT(element), "live-source", TRUE, NULL);
  }

  for (gint i = 0;i < num_sources;i++) {
    src_pad = gst_element_get_static_pad(
      source_elements[i], "src"
    );
    if (src_pad == NULL) {
        g_printerr("src pad error\n");
        return (NULL);
    }
    
    str = g_strdup_printf("sink_%u", i);
    sink_pad = gst_element_get_request_pad(element, str);
    g_free(str);

    if (gst_pad_link(src_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("pad link error\n");
        return (NULL);
    }
    gst_object_unref(GST_OBJECT(sink_pad));
    gst_object_unref(GST_OBJECT(src_pad));
  }

  element = make_element_and_link("nvinfer", NULL, bin, element);
  g_object_set(G_OBJECT(element), 
    "output-tensor-meta", TRUE,
    "config-file-path", "deepstream_pose_estimation_config.txt", 
    NULL
  );
  *p_pgie_src_pad = gst_element_get_static_pad(element, "src");

  element = make_element_and_link("nvvideoconvert", NULL, bin, element);

  element = make_element_and_link("nvdsosd", NULL, bin, element);
  *p_osd_sink_pad = gst_element_get_static_pad(element, "sink");

  element = make_element_and_link("tee", NULL, bin, element);
  
  return (element);
}

GstElement *
construct_display_bin(
  GstBin *bin, GstElement *tee_element
)
{
  GstElement *element = NULL;
  GstElement *element0 = NULL;

#ifdef PLATFORM_TEGRA
  element = make_element_and_link("nvegltransform", NULL, bin, element);
  element0 = element;
  element = make_element_and_link("nveglglessink", NULL, bin, element);
#else
  element = make_element_and_link("nveglglessink", NULL, bin, element);
  element0 = element;
#endif

  if (!link_element_to_tee_src_pad(tee_element, element0))
  {
    g_printerr("Could not link tee to nvsink\n");
    return (NULL);
  }

  return (element);
}

GstElement *
construct_file_sink_bin(
  GstBin *bin, GstElement *tee_element, const gchar *file_path 
)
{
  GstElement *element = NULL;
  GstCaps *caps = NULL;

  element = make_element_and_link("queue", NULL, bin, element);

  if (!link_element_to_tee_src_pad(tee_element, element))
  {
    g_printerr("Could not link tee to queue\n");
    return (NULL);
  }

  element = make_element_and_link("nvvideoconvert", NULL, bin, element);

  element = make_caps_and_link(
    "video/x-raw(memory:NVMM), format=I420", NULL, bin, element
  );

  element = make_element_and_link("nvv4l2h264enc", NULL, bin, element);

  element = make_element_and_link("h264parse", NULL, bin, element);

  element = make_element_and_link("qtmux", NULL, bin, element);

  element = make_element_and_link("filesink", NULL, bin, element);
  g_object_set(G_OBJECT(element), "location", file_path, NULL);

  return (element);
}

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstBus *bus = NULL;
  GstElement *pipeline = NULL;
  GstElement *vid_src = NULL;
  GstElement *tee = NULL;
  GstElement *nvsink = NULL;
  GstElement *element_list[8];
  GstPad *pgie_src_pad = NULL;
  GstPad *osd_sink_pad = NULL;
  gboolean is_live = FALSE;
  guint bus_watch_id;
  gchar input_path[80];
  gchar output_path[80];

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* get the input path and the output path */
  g_strlcpy(input_path, "/dev/video0", sizeof input_path);
  memset(output_path, 0, sizeof output_path);
  if (argc >= 2) {
    g_strlcpy(input_path, argv[1], sizeof input_path);
    if (argc >= 3) {
      g_strlcpy(output_path, argv[2], sizeof output_path);
      g_strlcat(output_path, OUTPUT_FILE, sizeof output_path);
      g_print("output file %s\n", output_path);
    }
  }
  g_print("input file %s\n", input_path);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

  if (strncmp(input_path, "/dev/video", strlen("/dev/video")) == 0) {
    is_live = TRUE;
    vid_src = construct_camera_source_bin(
      GST_BIN(pipeline), input_path, CAP_WIDTH, CAP_HEIGHT
    );
  }
  else {
    is_live = FALSE;
    vid_src = construct_file_source_bin(
      GST_BIN(pipeline), input_path
    );
  }

  element_list[0] = vid_src;
  tee = construct_inference_bin(
    GST_BIN(pipeline), element_list, 1, is_live,
    &pgie_src_pad, &osd_sink_pad
  );

  if (output_path[0] != 0 && !is_live) {
    construct_file_sink_bin(GST_BIN(pipeline), tee, output_path);
  }

  nvsink = construct_display_bin(GST_BIN(pipeline), tee);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  if (!pgie_src_pad)
    g_print("Unable to get pgie src pad\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_pad_buffer_probe, (gpointer)nvsink, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, (gpointer)nvsink, NULL);

  /* Set the pipeline to "playing" state */
  g_print("Now playing...\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline),
    GST_DEBUG_GRAPH_SHOW_ALL, "pipeline_graph");

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
