#!/usr/bin/env python3
import cv2
import subprocess
import numpy as np
import os
import time
import acl
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ais_bench.infer.interface import InferSession
from ais_bench.infer.common.utils import logger_print

# 保留时间测量函数
def measure_time(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger_print(f"{func.__name__} 耗时: {execution_time:.4f} 秒")
        return result
    return wrapper


class ImageModelInference:
    """模型推理核心类，处理预处理、推理和后处理"""
    def __init__(self, model_path, input_size=(1280, 1280), device_id=0, stride=32):
        self.model_path = model_path
        self.input_size = input_size  # (H, W)
        self.device_id = device_id
        self.stride = stridewei西
        self.acl_initialized = False
        self.context = None
        
        # 预定义绘图字体参数（减少重复计算）
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 2
        # 预计算标签高度（固定字体下高度不变）
        self.label_height = cv2.getTextSize("0", self.font, self.font_scale, self.font_thickness)[0][1] + 5
        
        # 初始化ACL环境
        self.init_acl()
        
        # 加载模型
        self.session = InferSession(device_id, model_path)
        logger_print(f"模型已加载: {model_path}")

    def init_acl(self):
        """初始化ACL资源"""
        try:
            ret = acl.rt.get_context()
            if ret[0] is not None:
                self.acl_initialized = True
                self.context = ret[0]
                logger_print("使用现有ACL上下文")
                return
        except Exception as e:
            logger_print(f"检查ACL上下文失败: {str(e)}")
            
        ret = acl.init()
        if ret != 0:
            raise Exception(f"ACL初始化失败，错误码: {ret}")
        self.acl_initialized = True
        
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise Exception(f"设置设备 {self.device_id} 失败，错误码: {ret}")
        
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0 or self.context is None:
            raise Exception(f"创建上下文失败，错误码: {ret}")

    def letterbox(self, im, new_shape=(1280, 1280), color=(114, 114, 114)):
        """图像缩放和填充（优化尺寸计算，减少冗余操作）"""
        shape = im.shape[:2]  # 原始尺寸 [H, W]
        if shape == new_shape:
            return im  # 尺寸匹配直接返回，减少计算
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W, H)
        
        # 计算填充量（确保为stride倍数）
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw % self.stride, dh % self.stride
        dw //= 2
        dh //= 2
        
        # 缩放图像（仅当尺寸不匹配时）
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 添加填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        # 仅在必要时强制调整（减少冗余resize）
        if im.shape[:2] != new_shape:
            im = cv2.resize(im, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return im

    @measure_time
    def preprocess(self, cv_image):
        """图像预处理（减少日志，优化内存操作）"""
        original_shape = cv_image.shape[:2]
        
        # 缩放填充
        letterboxed_img = self.letterbox(cv_image, new_shape=self.input_size)
        
        # 颜色转换+归一化（合并操作减少中间变量）
        rgb_img = cv2.cvtColor(letterboxed_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0
        
        # 维度转换（确保内存连续，加速推理）
        input_data = np.transpose(normalized_img, (2, 0, 1))[np.newaxis, :, :, :]
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        
        return input_data, original_shape
    
    @measure_time
    def inference(self, input_data):
        """执行模型推理（简化输入处理）"""
        if not isinstance(input_data, list):
            input_data = [input_data]
        return self.session.infer(input_data, mode='static')

    def non_max_suppression(self, prediction, iou_threshold=0.25):
        """使用OpenCV内置NMS加速（C++实现，替代Python循环）"""
        if prediction.size == 0:
            return np.zeros((0, 6), dtype=prediction.dtype)
        
        # 提取框坐标(x1,y1,x2,y2)、置信度、类别ID
        boxes = prediction[:, :4].astype(np.float32)
        scores = prediction[:, 4]
        class_ids = prediction[:, 5].astype(np.int32)
        
        keep = []
        # 按类别分组NMS，避免跨类别抑制
        for class_id in np.unique(class_ids):
            mask = (class_ids == class_id)
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            if len(class_boxes) == 0:
                continue
            
            # 转换为OpenCV格式(x,y,w,h)
            x, y, x2, y2 = class_boxes[:, 0], class_boxes[:, 1], class_boxes[:, 2], class_boxes[:, 3]
            opencv_boxes = np.column_stack([x, y, x2 - x, y2 - y])  # (x,y,w,h)
            
            # 执行NMS
            indices = cv2.dnn.NMSBoxes(
                bboxes=opencv_boxes.tolist(),
                scores=class_scores.tolist(),
                score_threshold=0.01,
                nms_threshold=iou_threshold
            )
            if len(indices) > 0:
                keep.extend(np.where(mask)[0][indices.flatten()])
        
        return prediction[keep]

    @measure_time
    def postprocess_and_draw(self, outputs, original_image, original_shape):
        """后处理+绘图（向量化操作，减少循环）"""
        # 解析输出
        detections = outputs[0].reshape(-1, 85)
        if len(detections) == 0:
            return original_image  # 无检测框直接返回
        
        # 0. 计算类别置信度（向量化）
        confidences = detections[:, 4]
        class_scores = detections[:, 5:]
        class_confidences = confidences[:, None] * class_scores
        max_class_confidences = np.max(class_confidences, axis=1)
        max_class_indices = np.argmax(class_confidences, axis=1)
        
        # 1. 过滤低置信度框（向量化）
        valid_mask = (max_class_confidences > 0.85)
        high_conf_detections = detections[valid_mask]
        high_conf_class_indices = max_class_indices[valid_mask]
        high_conf_class_confidences = max_class_confidences[valid_mask]
        if len(high_conf_detections) == 0:
            return original_image
        
        # 2. 中心坐标转边角坐标（向量化）
        boxes_xywh = high_conf_detections[:, :4]
        boxes_x1y1x2y2 = np.empty_like(boxes_xywh)
        boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        
        # 3. 准备NMS输入并执行
        nms_boxes = np.hstack([boxes_x1y1x2y2, 
                              high_conf_class_confidences[:, None], 
                              high_conf_class_indices[:, None]])
        nms_detections = self.non_max_suppression(nms_boxes, 0.25)
        if len(nms_detections) == 0:
            return original_image
        
        # 4. 坐标转换参数
        input_height, input_width = self.input_size
        orig_height, orig_width = original_shape
        r = min(input_height / orig_height, input_width / orig_width)
        pad_x = (input_width - orig_width * r) / 2
        pad_y = (input_height - orig_height * r) / 2
        
        # 5. 批量计算原图坐标（向量化）
        x1, y1, x2, y2 = nms_detections[:, 0], nms_detections[:, 1], nms_detections[:, 2], nms_detections[:, 3]
        real_x1 = ((x1 - pad_x) / r).astype(np.int32)
        real_y1 = ((y1 - pad_y) / r).astype(np.int32)
        real_x2 = ((x2 - pad_x) / r).astype(np.int32)
        real_y2 = ((y2 - pad_y) / r).astype(np.int32)
        confs = nms_detections[:, 4]
        cls_ids = nms_detections[:, 5].astype(np.int32)
        
        # 6. 过滤小目标（向量化）
        min_box_area = 0.001 * orig_width * orig_height
        box_areas = (real_x2 - real_x1) * (real_y2 - real_y1)
        valid_mask = (box_areas >= min_box_area)
        if not np.any(valid_mask):
            return original_image
        
        # 提取有效框
        real_x1, real_y1 = real_x1[valid_mask], real_y1[valid_mask]
        real_x2, real_y2 = real_x2[valid_mask], real_y2[valid_mask]
        confs, cls_ids = confs[valid_mask], cls_ids[valid_mask]
        
        # 7. 绘图（优化循环，减少重复计算）
        result_image = np.ascontiguousarray(original_image, dtype=np.uint8)  # 确保内存连续加速绘图
        for i in range(len(real_x1)):
            rx1, ry1, rx2, ry2 = real_x1[i], real_y1[i], real_x2[i], real_y2[i]
            conf = confs[i]
            cls_id = cls_ids[i]
            
            # 绘制边界框
            cv2.rectangle(result_image, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            
            # 绘制标签（预计算字体参数，减少getTextSize调用）
            label = f"Class {cls_id}: {conf:.2f}"
            label_width = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0][0]
            # 标签背景
            cv2.rectangle(
                result_image, 
                (rx1, ry1 - self.label_height), 
                (rx1 + label_width, ry1), 
                (0, 255, 0), 
                -1
            )
            # 标签文本
            cv2.putText(
                result_image, label, 
                (rx1, ry1 - 2), 
                self.font, self.font_scale, (255, 255, 255), self.font_thickness
            )
        
        return result_image


    def destroy(self):
        """释放资源"""
        if hasattr(self, 'session') and self.session:
            self.session.free_resource()
        
        if self.acl_initialized:
            if self.context:
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
            acl.finalize()


class ROS2ModelInferenceRTSP(Node):
    def __init__(self):
        super().__init__('ros2_model_inference_rtsp')
        
        # 参数配置
        self.declare_parameter('model_path', 'yolov7-e6-bs1.om')
        self.declare_parameter('input_size', [1280, 1280])
        self.declare_parameter('device_id', 0)
        self.declare_parameter('stride', 32)
        self.declare_parameter('sub_image_topic', '/rs_camera/rgb')
        self.declare_parameter('rtsp_url', 'rtsp://192.168.137.100:554/live/test')
        self.declare_parameter('fps', 15)
        
        # 获取参数
        self.model_path = self.get_parameter('model_path').value
        self.input_size = tuple(self.get_parameter('input_size').value)
        self.device_id = self.get_parameter('device_id').value
        self.stride = self.get_parameter('stride').value
        self.rtsp_url = self.get_parameter('rtsp_url').value
        self.fps = self.get_parameter('fps').value
        
        # 初始化组件
        self.bridge = CvBridge()
        self.ffmpeg_process = None
        self.ffmpeg_ready = False  # 标记FFmpeg是否就绪
        
        # 初始化推理器
        self.inferencer = ImageModelInference(
            model_path=self.model_path,
            input_size=self.input_size,
            device_id=self.device_id,
            stride=self.stride
        )
        
        # 创建ROS订阅者
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter('sub_image_topic').value,
            self.image_callback,
            10
        )
        
        # 启动FFmpeg进程
        self.start_ffmpeg()
        
        self.get_logger().info(f"节点已启动，订阅话题: {self.get_parameter('sub_image_topic').value}")
        self.get_logger().info(f"RTSP推流地址: {self.rtsp_url}")
    
    def start_ffmpeg(self):
        """启动FFmpeg推流进程（增加健康检查）"""
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.terminate()
        
        command = [
            'ffmpeg',
            '-y', '-loglevel', 'error',  # 关闭FFmpeg日志减少IO
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.input_size[1]}x{self.input_size[0]}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.rtsp_url
        ]
        
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.ffmpeg_ready = True

    def image_callback(self, msg):
        """图像回调（减少日志，优化异常处理）"""
        try:
            # 图像转换（直接指定格式，减少判断）
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 预处理
            input_data, original_shape = self.inferencer.preprocess(cv_image)
            
            # 推理
            outputs = self.inferencer.inference(input_data)
            
            # 后处理并绘制
            result_image = self.inferencer.postprocess_and_draw(outputs, cv_image, original_shape)
            
            # 推流
            self.push_to_rtsp(result_image)
            
        except Exception as e:
            self.get_logger().error(f"处理出错: {str(e)}")
            if not self.ffmpeg_ready:
                self.start_ffmpeg()
    
    def push_to_rtsp(self, image):
        """推流优化（减少尺寸检查，避免重复resize）"""
        try:
            if not self.ffmpeg_ready or self.ffmpeg_process.poll() is not None:
                self.start_ffmpeg()
            
            # 仅在尺寸不匹配时resize
            if image.shape[:2] != (self.input_size[0], self.input_size[1]):
                image = cv2.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # 写入数据（确保内存连续）
            self.ffmpeg_process.stdin.write(image.tobytes())
            
        except Exception as e:
            self.get_logger().error(f"推流失败: {str(e)}")
            self.ffmpeg_ready = False
    
    def destroy_node(self):
        """释放资源"""
        self.inferencer.destroy()
        
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait(timeout=2)
        
        super().destroy_node()
        self.get_logger().info("节点已关闭")


def main(args=None):
    rclpy.init(args=args)
    node = ROS2ModelInferenceRTSP()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
