#include "pointr/pointr_trt.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
class PoinTrNode : public rclcpp::Node {
 std::unique_ptr<pointr::PoinTrTRT> model_;
 rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
 rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
 std::string expected_frame_;
public:
 PoinTrNode():Node("pointr_node"){
  auto engine=declare_parameter<std::string>("engine_path","");
  auto input=declare_parameter<std::string>("input_topic","/object/points_camera");
  auto output=declare_parameter<std::string>("output_topic","/pointr/completed_points");
  expected_frame_=declare_parameter<std::string>("expected_frame","camera_depth_optical_frame");
  if(engine.empty())throw std::runtime_error("Set engine_path to a locally built TensorRT engine");
  model_=std::make_unique<pointr::PoinTrTRT>(engine);
  pub_=create_publisher<sensor_msgs::msg::PointCloud2>(output,rclcpp::SensorDataQoS());
  sub_=create_subscription<sensor_msgs::msg::PointCloud2>(input,rclcpp::SensorDataQoS().keep_last(1),
   [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
    try{
     if(!expected_frame_.empty()&&msg->header.frame_id!=expected_frame_){
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,"Rejecting frame '%s'; expected camera optical frame '%s'",msg->header.frame_id.c_str(),expected_frame_.c_str());return;}
     bool xyz32=true;
     for(const auto* name:{"x","y","z"}){bool found=false;for(auto& f:msg->fields)if(f.name==name&&f.datatype==sensor_msgs::msg::PointField::FLOAT32&&f.count==1)found=true;xyz32&=found;}
     if(!xyz32)throw std::runtime_error("PointCloud2 requires float32 XYZ fields");
     sensor_msgs::PointCloud2ConstIterator<float> x(*msg,"x"),y(*msg,"y"),z(*msg,"z");
     std::vector<pointr::Point> p;for(;x!=x.end();++x,++y,++z)if(std::isfinite(*x)&&std::isfinite(*y)&&std::isfinite(*z))p.push_back({*x,*y,*z});
     if(p.size()<64)throw std::runtime_error("Require at least 64 valid object points");
     auto result=model_->infer(p);
     sensor_msgs::msg::PointCloud2 out;out.header=msg->header;
     sensor_msgs::PointCloud2Modifier modify(out);modify.setPointCloud2FieldsByString(1,"xyz");modify.resize(result.points.size());
     sensor_msgs::PointCloud2Iterator<float> ox(out,"x"),oy(out,"y"),oz(out,"z");
     for(auto q:result.points){*ox=q.x;*oy=q.y;*oz=q.z;++ox;++oy;++oz;}out.is_dense=true;pub_->publish(out);
    }catch(const std::exception& e){RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),3000,"PoinTr: %s",e.what());}
   });
 }
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<PoinTrNode>());}catch(const std::exception& e){std::cerr<<e.what()<<'\n';rclcpp::shutdown();return 1;}rclcpp::shutdown();}
