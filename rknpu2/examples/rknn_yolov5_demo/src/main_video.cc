// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#define BUILD_VIDEO_RTSP 

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"

#include "rknn_api.h"
#include "postprocess.h"

#include "utils/mpp_decoder.h"
#include "utils/mpp_encoder.h"
#include "utils/drawing.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#if defined(BUILD_VIDEO_RTSP)
#include "mk_mediakit.h"
#endif

#include <mutex>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_CHANNELS 16
#define MAX_IMAGE_SIZE (1920*1080*3/2) // 最大支持1080p YUV420SP图像

typedef struct {
    int width;
    int height;
    int format;
    size_t size;
    bool updated;
    sem_t sem;
} SharedImageHeader;

typedef struct {
    SharedImageHeader header;
    unsigned char data[];
} SharedImage;
// 在文件顶部添加全局互斥锁
static std::mutex rga_mutex;


#define OUT_VIDEO_PATH "out.h264"

typedef struct
{
  std::string path;
  int chn_num;
  int img_count;
}chn_info;

typedef struct
{
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  FILE *out_fp;
  MppDecoder *decoder;
  MppEncoder *encoder;
  int mk_rtsp_code;
  chn_info chn_img;
  int shm_fd[MAX_CHANNELS];  // 每个通道的共享内存文件描述符
  SharedImage* shared_img[MAX_CHANNELS]; // 映射的共享内存指针
} rknn_app_context_t;

typedef struct
{
  int width;
  int height;
  int width_stride;
  int height_stride;
  int format;
  char *virt_addr;
  int fd;
} image_frame_t;

// 线程参数结构体，添加线程编号成员
typedef struct {
    const char *model_name;
    const char *video_name;
    int video_type;
    int thread_id; // 线程编号
} ThreadArgs;

// 线程函数声明
void* process_video_thread(void* arg);

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *read_file_data(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int write_data_to_file(const char *path, char *data, unsigned int size)
{
  FILE *fp;

  fp = fopen(path, "w");
  if (fp == NULL)
  {
    printf("open error: %s", path);
    return -1;
  }

  fwrite(data, 1, size, fp);
  fflush(fp);

  fclose(fp);
  return 0;
}

static int init_model(const char *model_path, rknn_app_context_t *app_ctx)
{
  int ret;
  rknn_context ctx;

  /* Create the neural network */
  printf("Loading mode...\n");
  int model_data_size = 0;
  unsigned char *model_data = read_file_data(model_path, &model_data_size);
  if (model_data == NULL)
  {
    return -1;
  }

  ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  if (model_data)
  {
    free(model_data);
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0)
  {
    printf("rknn_query RKNN_QUERY_SDK_VERSION error ret=%d\n", ret);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(rknn_input_output_num));
  if (ret < 0)
  {
    printf("rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", app_ctx->io_num.n_input, app_ctx->io_num.n_output);

  rknn_tensor_attr *input_attrs = (rknn_tensor_attr *)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
  memset(input_attrs, 0, app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
  for (int i = 0; i < app_ctx->io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr *output_attrs = (rknn_tensor_attr *)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
  memset(output_attrs, 0, app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
  for (int i = 0; i < app_ctx->io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_query RKNN_QUERY_OUTPUT_ATTR error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(output_attrs[i]));
  }

  app_ctx->input_attrs = input_attrs;
  app_ctx->output_attrs = output_attrs;
  app_ctx->rknn_ctx = ctx;

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    printf("model is NCHW input fmt\n");
    app_ctx->model_channel = input_attrs[0].dims[1];
    app_ctx->model_height = input_attrs[0].dims[2];
    app_ctx->model_width = input_attrs[0].dims[3];
  }
  else
  {
    printf("model is NHWC input fmt\n");
    app_ctx->model_height = input_attrs[0].dims[1];
    app_ctx->model_width = input_attrs[0].dims[2];
    app_ctx->model_channel = input_attrs[0].dims[3];
  }
  printf("model input height=%d, width=%d, channel=%d\n", app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

  return 0;
}

static int release_model(rknn_app_context_t *app_ctx)
{
  if (app_ctx->rknn_ctx != 0)
  {
    rknn_destroy(app_ctx->rknn_ctx);
  }
  free(app_ctx->input_attrs);
  free(app_ctx->output_attrs);
  deinitPostProcess();
  return 0;
}

static int inference_model(rknn_app_context_t *app_ctx, image_frame_t *img, detect_result_group_t *detect_result)
{
  int ret;
  rknn_context ctx = app_ctx->rknn_ctx;
  int model_width = app_ctx->model_width;
  int model_height = app_ctx->model_height;
  int model_channel = app_ctx->model_channel;

  struct timeval start_time, stop_time;
  const float nms_threshold = NMS_THRESH;
  const float box_conf_threshold = BOX_THRESH;
  // You may not need resize when src resulotion equals to dst resulotion
  void *resize_buf = nullptr;
  // init rga context
  rga_buffer_t src;
  rga_buffer_t dst;
  im_rect src_rect;
  im_rect dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));

  printf("input image %dx%d stride %dx%d format=%d\n", img->width, img->height, img->width_stride, img->height_stride, img->format);

  float scale_w = (float)model_width / img->width;
  float scale_h = (float)model_height / img->height;

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = model_width * model_height * model_channel;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  //打印当前线程编号
  printf("resize with RGA ,---------------pthread_self :%lu\n", (int)pthread_self());

  resize_buf = malloc(model_width * model_height * model_channel);
  memset(resize_buf, 0, model_width * model_height * model_channel);

  {
    std::lock_guard<std::mutex> lock(rga_mutex);
    src = wrapbuffer_virtualaddr((void *)img->virt_addr, img->width, img->height, img->format, img->width_stride, img->height_stride);
    dst = wrapbuffer_virtualaddr((void *)resize_buf, model_width, model_height, RK_FORMAT_RGB_888);
    ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      return -1;
    }
    IM_STATUS STATUS = imresize(src, dst);
  }

  inputs[0].buf = resize_buf;

  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, app_ctx->io_num.n_input, inputs);

  rknn_output outputs[app_ctx->io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < app_ctx->io_num.n_output; i++)
  {
    outputs[i].index = i;
    outputs[i].want_float = 0;
  }

  ret = rknn_run(ctx, NULL);
  if (ret < 0)
  {
    printf("rknn run error %d\n", ret);
    exit(-1);
  }

  ret = rknn_outputs_get(ctx, app_ctx->io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  printf("\n\n\n\npost process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n\n\n\n\n", box_conf_threshold, nms_threshold);

  std::vector<float> out_scales;
  std::vector<int32_t> out_zps;
  for (int i = 0; i < app_ctx->io_num.n_output; ++i)
  {
    out_scales.push_back(app_ctx->output_attrs[i].scale);
    out_zps.push_back(app_ctx->output_attrs[i].zp);
  }
  BOX_RECT pads;
  memset(&pads, 0, sizeof(BOX_RECT));

  post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, model_height, model_width,
               box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, detect_result);
  ret = rknn_outputs_release(ctx, app_ctx->io_num.n_output, outputs);

  if (resize_buf)
  {
    free(resize_buf);
  }
  return 0;
}



void mpp_decoder_frame_callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data)
{

  rknn_app_context_t *ctx = (rknn_app_context_t *)userdata;

  int ret = 0;
  static int frame_index = 0;
  frame_index++;

  void *mpp_frame = NULL;
  int mpp_frame_fd = 0;
  void *mpp_frame_addr = NULL;
  int enc_data_size;

  rga_buffer_t origin;
  rga_buffer_t src;

  if (ctx->encoder == NULL)
  {
    MppEncoder *mpp_encoder = new MppEncoder();
    MppEncoderParams enc_params;
    memset(&enc_params, 0, sizeof(MppEncoderParams));
    enc_params.width = width;
    enc_params.height = height;
    enc_params.hor_stride = width_stride;
    enc_params.ver_stride = height_stride;
    enc_params.fmt = MPP_FMT_YUV420SP;
    // enc_params.type = MPP_VIDEO_CodingHEVC;
    // Note: rk3562只能支持h264格式的视频流
    enc_params.type = MPP_VIDEO_CodingAVC;
    mpp_encoder->Init(enc_params, NULL);

    ctx->encoder = mpp_encoder;
  }



  int enc_buf_size = ctx->encoder->GetFrameSize();
  char *enc_data = (char *)malloc(enc_buf_size);

  image_frame_t img;
  img.width = width;
  img.height = height;
  img.width_stride = width_stride;
  img.height_stride = height_stride;
  img.fd = fd;
  img.virt_addr = (char *)data;
  img.format = RK_FORMAT_YCbCr_420_SP;
  detect_result_group_t detect_result;
  memset(&detect_result, 0, sizeof(detect_result_group_t));

  //完整的视频处理流程通常是：rtsp解码264→处理yuv（直接CV or 推理后CV）→编码264→rtsp推流(传输 或 保存)  or  rtsp解码264→处理yuv→ 本地渲染播放
  
  // mpp_frame = ctx->encoder->GetInputFrameBuffer();
  // mpp_frame_fd = ctx->encoder->GetInputFrameBufferFd(mpp_frame);
  // mpp_frame_addr = ctx->encoder->GetInputFrameBufferAddr(mpp_frame);

#if 1
  {
    // 使用 共享内存 rk yolo 和 ax clip 两个进行之间 传递 8-16路 img 数据
    //  获取当前通道号
    int chn_num = ctx->chn_img.chn_num;

    // 加锁保护共享内存访问
    sem_wait(&ctx->shared_img[chn_num]->header.sem);

    // 填充共享内存数据
    ctx->shared_img[chn_num]->header.width = width;
    ctx->shared_img[chn_num]->header.height = height;
    ctx->shared_img[chn_num]->header.format = format;
    ctx->shared_img[chn_num]->header.size = width * height * 3 / 2;
    //memcpy(ctx->shared_img[chn_num]->data, data, ctx->shared_img[chn_num]->header.size);
    // 使用RGA硬件加速拷贝YUV数据
    {
        std::lock_guard<std::mutex> lock(rga_mutex);
        rga_buffer_t src = wrapbuffer_virtualaddr((void *)data, width, height, RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst = wrapbuffer_virtualaddr((void *)ctx->shared_img[chn_num]->data, width, height, RK_FORMAT_YCbCr_420_SP);
        IM_STATUS status = imcopy(src, dst);
        if (status != IM_STATUS_SUCCESS) {
            printf("RGA copy failed: %s\n", imStrError(status));
        }
    }
    
    ctx->shared_img[chn_num]->header.updated = true;
    printf("-------------------RGA copy updated: %d \n",chn_num);
    // 解锁
    sem_post(&ctx->shared_img[chn_num]->header.sem);
  }
#endif

  // 如果某一路有配置推理模型
  //if(0)
  if (ctx->rknn_ctx > 0)
  {
    // 从输入的图片CVmat中，推理做目标检测获取目标检测框：detect_result
    ret = inference_model(ctx, &img, &detect_result);
    if (ret != 0)
    {
      printf("inference model fail\n");
      goto RET;
    }

  #if 0
    // Copy To another buffer avoid to modify mpp decoder buffer
    // 保护RGA拷贝操作
    {
      std::lock_guard<std::mutex> lock(rga_mutex);
      origin = wrapbuffer_fd(fd, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
      src = wrapbuffer_fd(mpp_frame_fd, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
      imcopy(origin, src);
    }

    // Draw objects
    for (int i = 0; i < detect_result.count; i++)
    {
      detect_result_t *det_result = &(detect_result.results[i]);
      printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
             det_result->box.right, det_result->box.bottom, det_result->prop);
      int x1 = det_result->box.left;
      int y1 = det_result->box.top;
      int x2 = det_result->box.right;
      int y2 = det_result->box.bottom;
      draw_rectangle_yuv420sp((unsigned char *)mpp_frame_addr, width_stride, height_stride, x1, y1, x2 - x1 + 1, y2 - y1 + 1, 0x00FF0000, 4);
    }

    img.virt_addr = (char *)mpp_frame_addr;
#endif

  }
    
#if 0
  {
      //每存储25帧图片，新建一个当前时间（ 以月日分秒为规则，如0626151609）的文件夹，再存储25帧图片
    if (ctx->chn_img.img_count % 25 == 0)
    {
      char dir_name[128];
      // 获取当前时间
      time_t now = time(NULL);
      struct tm tm_now;
      localtime_r(&now, &tm_now);  // 使用线程安全的localtime_r，并将结果复制到局部变量

    // 生成月日分秒格式的目录名 (如0626151609)
      snprintf(dir_name, sizeof(dir_name), "/userdata/clip/thread_%d/%02d%02d%02d%02d%02d",
              ctx->chn_img.chn_num,
              tm_now.tm_mon + 1,    // 月 (0-11 → 1-12)
              tm_now.tm_mday,       // 日 
              tm_now.tm_hour,       // 时
              tm_now.tm_min,        // 分
              tm_now.tm_sec);       // 秒
            
      if (mkdir(dir_name, 0777) == -1 && errno != EEXIST) {
          printf("create directory %s error\n", dir_name);
          goto RET;
      }
      printf("create directory -----------------------------------------------------------%s success\n", dir_name);

      ctx->chn_img.path = dir_name;
      ctx->chn_img.img_count = 0;
    }

    // 将img 中RK_FORMAT_YCbCr_420_SP格式的图 保存到 ctx->chn_img.path 路径中，按照bmp方式保存
    char img_path[128]; 
    snprintf(img_path, sizeof(img_path), "%s/frame_%d.bmp", ctx->chn_img.path.c_str(), ctx->chn_img.img_count);
    
    //CV保存jpeg图片 太消耗 CPU，换位rga算子
    // 使用RGA进行YUV420SP到BGR转换
    #if 1
    cv::Mat bgr_img(img.height, img.width, CV_8UC3);
    {
        std::lock_guard<std::mutex> lock(rga_mutex);
        rga_buffer_t src = wrapbuffer_virtualaddr((void *)img.virt_addr, img.width, img.height, RK_FORMAT_YCbCr_420_SP, img.width_stride, img.height_stride);
        rga_buffer_t dst = wrapbuffer_virtualaddr((void *)bgr_img.data, img.width, img.height, RK_FORMAT_BGR_888);
        IM_STATUS status = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
        if (status != IM_STATUS_SUCCESS) {
            printf("RGA convert color failed: %s\n", imStrError(status));
            goto RET;
        }
    }
    #endif

    // 将YUV420SP转换为BGR格式
    // cv::Mat yuv_img(img.height * 3/2, img.width, CV_8UC1, img.virt_addr);
    // cv::Mat bgr_img;
    // cv::cvtColor(yuv_img, bgr_img, cv::COLOR_YUV2BGR_NV12);
    
    //保存为 PNG、BMP 等格式，这些格式支持无损保存原始像素数据，不用压缩
    //BMP 6M左右，CPU消耗很少，但是消耗大量IO时间片，CPU空转等待（需要改为异步保存），导致拉流卡主，磁盘很快爆掉
    //25p*6m*8路 =1000MB 每秒，异步磁盘吃不消
    if (!cv::imwrite(img_path, bgr_img)) {
       printf("Failed to save image: %s\n", img_path);
    }

    ctx->chn_img.img_count++;

  }

#endif

#if 0
  // Encode to file，将CV后的图片，重新编码为264存储到文件
  // Write header on first frame
  if (frame_index == 1)
  {
    enc_data_size = ctx->encoder->GetHeader(enc_data, enc_buf_size);
    fwrite(enc_data, 1, enc_data_size, ctx->out_fp);
  }
  memset(enc_data, 0, enc_buf_size);
  enc_data_size = ctx->encoder->Encode(mpp_frame, enc_data, enc_buf_size);
  fwrite(enc_data, 1, enc_data_size, ctx->out_fp);
#endif

RET:
  if (enc_data != nullptr)
  {
    free(enc_data);
  }
}

int process_video_file(rknn_app_context_t *ctx, const char *path)
{
  int video_size;
  char *video_data = (char *)read_file_data(path, &video_size);
  char *video_data_end = video_data + video_size;
  printf("read video size=%d\n", video_size);

  const int SIZE = 8192;
  char *video_data_ptr = video_data;

  do
  {
    int pkt_eos = 0;
    int size = SIZE;
    if (video_data_ptr + size >= video_data_end)
    {
      pkt_eos = 1;
      size = video_data_end - video_data_ptr;
    }

    ctx->decoder->Decode((uint8_t *)video_data_ptr, size, pkt_eos);

    video_data_ptr += size;

    if (video_data_ptr >= video_data_end)
    {
      printf("reset decoder\n");
      break;
    }

    // LOGD("video_data_ptr=%p video_data_end=%p", video_data_ptr, video_data_end);
    // usleep(10*1000);
  } while (1);

  return 0;
}

void API_CALL on_track_frame_out(void *user_data, mk_frame frame)
{
  rknn_app_context_t *ctx = (rknn_app_context_t *)user_data;
  printf("on_track_frame_out ctx=%p,thread_id=%lu\n", ctx,pthread_self());
  const char *data = mk_frame_get_data(frame);
  size_t size = mk_frame_get_data_size(frame);
  printf("decoder=%p\n", ctx->decoder);
  ctx->decoder->Decode((uint8_t *)data, size, 0);
}

void API_CALL on_mk_play_event_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[],
                                    int track_count)
{
  //rknn_app_context_t *ctx = (rknn_app_context_t *)user_data;
  if (err_code == 0)
  {
    // 打印线程信息
    printf("play success! thread_id=%lu\n", pthread_self());
    int i;
    for (i = 0; i < track_count; ++i)
    {
      if (mk_track_is_video(tracks[i]))
      {
        log_info("got video track: %s", mk_track_codec_name(tracks[i]));
        // 监听track数据回调
        mk_track_add_delegate(tracks[i], on_track_frame_out, user_data);
      }
    }
  }
  else
  {
    printf("play failed: %d %s", err_code, err_msg);
  }
}

void API_CALL on_mk_shutdown_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[], int track_count)
{
  rknn_app_context_t *ctx = (rknn_app_context_t *)user_data;
  ctx->mk_rtsp_code = err_code;
 
  printf("\n\n\n\n\nplay interrupted: %d %s \n\n\n\n\n\n\n", err_code, err_msg);
}

int process_video_rtsp(rknn_app_context_t *ctx, const char *url)
{
  mk_config config;
  memset(&config, 0, sizeof(mk_config));
  config.log_mask = LOG_CONSOLE;
  mk_env_init(&config);
  
  int retry_count = 1024;  // 最大重试次数
  while(retry_count-- > 0) {
    mk_player player = mk_player_create();
    ctx->mk_rtsp_code = 0;  // 重置错误码
    
    mk_player_set_on_result(player, on_mk_play_event_func, ctx);
    mk_player_set_on_shutdown(player, on_mk_shutdown_func, ctx);
    mk_player_play(player, url);

    while(ctx->mk_rtsp_code <= 0) {
      usleep(1000*1000);  // 每100ms检查一次，循环中
    }

    if(ctx->mk_rtsp_code > 0) {
      mk_player_release(player);
      printf("RTSP connection failed, retrying... (%d attempts left),thread id: %lu\n", retry_count, (int)pthread_self());
      usleep(1*1000*1000);  // 等待1秒后重试
    } else {
      break;  // 成功则退出循环
    }
  }

  if(retry_count <= 0) {
    printf("Max retry attempts reached, giving up.-------------------------------------------------------\n");
    printf("\n\n\n\n\n\n\nexit------------------------------------thread id: %lu\n\n\n\n\n\n\n\n", (int)pthread_self());
    exit(-1);
  }

}

// 清理共享内存资源
void cleanup_shared_memory(rknn_app_context_t* ctx, int chn_num) {
    if (ctx->shared_img[chn_num]) {
        size_t mem_size = sizeof(SharedImageHeader) + MAX_IMAGE_SIZE;
        munmap(ctx->shared_img[chn_num], mem_size);
        ctx->shared_img[chn_num] = nullptr;
    }
    if (ctx->shm_fd[chn_num] != -1) {
        close(ctx->shm_fd[chn_num]);
        ctx->shm_fd[chn_num] = -1;
    }
}

// 初始化共享内存
void init_shared_memory(rknn_app_context_t* ctx, int chn_num) {
    char shm_name[64];
    snprintf(shm_name, sizeof(shm_name), "/rk_yolo_shm_%d", chn_num);
    
    // 创建或打开共享内存
    ctx->shm_fd[chn_num] = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (ctx->shm_fd[chn_num] == -1) {
        perror("shm_open failed");
        return;
    }

    // 设置共享内存大小
    size_t mem_size = sizeof(SharedImageHeader) + MAX_IMAGE_SIZE;
    if (ftruncate(ctx->shm_fd[chn_num], mem_size) == -1) {
        perror("ftruncate failed");
        close(ctx->shm_fd[chn_num]);
        return;
    }

    // 映射共享内存
    ctx->shared_img[chn_num] = (SharedImage*)mmap(NULL, mem_size, 
                                               PROT_READ | PROT_WRITE, 
                                               MAP_SHARED, ctx->shm_fd[chn_num], 0);
    if (ctx->shared_img[chn_num] == MAP_FAILED) {
        perror("mmap failed");
        close(ctx->shm_fd[chn_num]);
        return;
    }

  // 初始化信号量
    if (sem_init(&ctx->shared_img[chn_num]->header.sem, 1, 1) == -1) {
        perror("sem_init failed");
        fprintf(stderr, "信号量初始化失败，错误号: %d\n", errno);
        munmap(ctx->shared_img[chn_num], mem_size);
        close(ctx->shm_fd[chn_num]);
        return;
    }

    // 测试信号量
    if (sem_post(&ctx->shared_img[chn_num]->header.sem) == -1) {
        perror("信号量测试失败");
        munmap(ctx->shared_img[chn_num], mem_size);
        close(ctx->shm_fd[chn_num]);
        exit(0);
    }
}


// 线程函数定义
void* process_video_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int ret;

    //打印线程信息
    printf("thread %d start,thread_id=%lu\n", args->thread_id,pthread_self());

    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));

    if (args->thread_id >= 1 && args->thread_id <= 4) {
        ret = init_model(args->model_name, &app_ctx);
        if (ret != 0) {
            printf("Thread %d: init model fail\n", args->thread_id);
            return nullptr;
        }
    }

    if (app_ctx.decoder == NULL) {
        MppDecoder *decoder = new MppDecoder();
        decoder->Init(args->video_type, 30, &app_ctx);
        decoder->SetCallback(mpp_decoder_frame_callback);
        app_ctx.decoder = decoder;
    }

    if (app_ctx.out_fp == NULL) {
        char out_video_path[256];
       
        #if 0
        // 为每个线程生成不同的输出文件名
        snprintf(out_video_path, sizeof(out_video_path), "out_thread_%d.h264", args->thread_id);
        FILE *fp = fopen(out_video_path, "w");
        if (fp == NULL) {
            printf("Thread %d: open %s error\n", args->thread_id, out_video_path);
            return nullptr;
        }
        app_ctx.out_fp = fp;
        #endif

        //各线程分别创建到1-16 号16个通道文件夹
        app_ctx.chn_img.chn_num =  args->thread_id;
        char dir_name[64];  
        snprintf(dir_name, sizeof(dir_name), "/userdata/clip/thread_%d", args->thread_id);
        if (mkdir(dir_name, 0777) == -1 && errno != EEXIST) {
            printf("Thread %d: create directory %s error\n", args->thread_id, dir_name);
        }
        
        init_shared_memory(&app_ctx, args->thread_id);
    }

    printf("Thread %d: app_ctx=%p decoder=%p\n", args->thread_id, &app_ctx, app_ctx.decoder);

    if (strncmp(args->video_name, "rtsp", 4) == 0) {
#if defined(BUILD_VIDEO_RTSP)
        process_video_rtsp(&app_ctx, args->video_name);
#else
        printf("Thread %d: rtsp no support\n", args->thread_id);
#endif
    } else {
        process_video_file(&app_ctx, args->video_name);
    }

    printf("Thread %d: waiting finish\n", args->thread_id);
    usleep(3 * 1000 * 1000);


    cleanup_shared_memory(&app_ctx, args->thread_id);

    // release
    fflush(app_ctx.out_fp);
    fclose(app_ctx.out_fp);

    if (app_ctx.decoder != nullptr) {
        delete (app_ctx.decoder);
        app_ctx.decoder = nullptr;
    }
    if (app_ctx.encoder != nullptr) {
        delete (app_ctx.encoder);
        app_ctx.encoder = nullptr;
    }

    if (args->thread_id >= 1 && args->thread_id <= 4) {
        release_model(&app_ctx);
    }

    return nullptr;
}


/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
  int status = 0;

  //单进程8线程，是由于单个进程内的mpp_decoder_frame_callback 共享资源竞争和上下文切换开销过大导致的，拉流线程卡主，流量上不去。
  //  //ctx->decoder->Decode((uint8_t *)data, size, 0); 去掉这一行，16路的流量就可以按照相机的静态码率拉到
  //--需要做全链路异步化改造： 拉流异步化 ，解码异步化，推理异步化……
  const int thread_num =8;
  pthread_t threads[thread_num];
  ThreadArgs args[thread_num];

  if (argc != 4)
  {
    printf("Usage: %s <rknn_model> <video_path_prefix> <video_type 264/265> \n", argv[0]);
    return -1;
  }

  char *model_name = (char *)argv[1];
  char *video_path_prefix = argv[2];
  int video_type = atoi(argv[3]);

  for (int i = 0; i < thread_num; ++i) {
      args[i].model_name = model_name;
      args[i].video_name = strdup(video_path_prefix);
      args[i].video_type = video_type;
      args[i].thread_id = i + 1; // 设置线程编号

      if (pthread_create(&threads[i], NULL, process_video_thread, &args[i]) != 0) {
          printf("Failed to create thread %d\n", i);
          return -1;
      }

      sleep(1);
  }

  // 等待所有线程结束
  for (int i = 0; i < thread_num; ++i) {
      pthread_join(threads[i], NULL);
      free((void*)args[i].video_name);
  }

 
  return 0;
}

