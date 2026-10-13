from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
 return LaunchDescription([
  DeclareLaunchArgument('engine_path'),
  DeclareLaunchArgument('input_topic',default_value='/object/points_camera'),
  DeclareLaunchArgument('expected_frame',default_value='camera_depth_optical_frame'),
  Node(package='pointr_cpp',executable='pointr_node',name='pointr_node',output='screen',
       parameters=[{'engine_path':LaunchConfiguration('engine_path'),
                    'input_topic':LaunchConfiguration('input_topic'),
                    'expected_frame':LaunchConfiguration('expected_frame')}])])
