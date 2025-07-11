#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess
import numpy as np

class ImageToRTSPPublisher(Node):
    def __init__(self):
        super().__init__('image_to_rtsp_publisher')
        
        # 参数声明
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('rtsp_url', 'rtsp://192.168.1.200/live/test')
        self.declare_parameter('fps', 25)
        self.declare_parameter('resolution', [640, 480])
        
        # 获取参数值
        input_topic = self.get_parameter('input_topic').value
        self.rtsp_url = self.get_parameter('rtsp_url').value
        self.fps = self.get_parameter('fps').value
        self.width, self.height = self.get_parameter('resolution').value
        
        # 初始化CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # 创建图像订阅者
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        # 启动FFmpeg进程
        self.start_ffmpeg_process()
        
        self.get_logger().info(f"节点已启动，订阅话题: {input_topic}")
        self.get_logger().info(f"RTSP推流地址: {self.rtsp_url}")
    
    def start_ffmpeg_process(self):
        """启动FFmpeg推流进程"""
        command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-loglevel', 'error',  # 减少日志输出
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',  # 分辨率
            '-r', str(self.fps),  # 帧率
            '-i', '-',  # 输入来自管道
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # 快速编码，适合实时流
            '-tune', 'zerolatency',  # 零延迟优化
            '-pix_fmt', 'yuv420p',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # 使用TCP传输更可靠
            self.rtsp_url
        ]
        
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def image_callback(self, msg):
        """处理接收到的图像消息并推流"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 调整图像大小以匹配FFmpeg设置
            if cv_image.shape[1] != self.width or cv_image.shape[0] != self.height:
                cv_image = cv2.resize(cv_image, (self.width, self.height))
            
            # 示例：在图像中心画一个圆
            cv2.circle(cv_image, (self.width//2, self.height//2), 50, (0, 0, 255), 5)
            
            # 将图像写入FFmpeg进程
            self.ffmpeg_process.stdin.write(cv_image.tobytes())
            
        except Exception as e:
            self.get_logger().error(f"处理图像时出错: {str(e)}")
    
    def destroy_node(self):
        """节点关闭时释放资源"""
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageToRTSPPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()